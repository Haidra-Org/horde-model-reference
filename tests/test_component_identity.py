"""Tests for the per-checkpoint component-identity sidecar.

These pin the sidecar's contract for its cross-repo consumers (hordelib, the worker): an ensure/read round
trip caches the identity, idempotent ensures do not rewrite, a changed checkpoint invalidates the cache, an
optionally extracted VAE is content-addressed (so identical VAEs deduplicate) and hashes equal to the recorded
embedded identity, and a non-safetensors container is refused without writing anything. Synthetic safetensors
files are built in-test (torch-free) with the same helper the component_hash tests use.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from pydantic import ValidationError

from horde_model_reference.component_hash import (
    ComponentKind,
    UnsupportedContainerError,
    hash_standalone_component_file,
)
from horde_model_reference.component_identity import (
    ComponentIdentitySidecar,
    EmbeddedComponentIdentity,
    ensure_sidecar,
    read_sidecar,
    sidecar_path_for,
)

# A monolithic-checkpoint tensor set: one UNet tensor, a two-tensor VAE, and one text-encoder tensor. The VAE
# tensors mirror the component_hash conformance vector so the extracted-file hash is deterministic.
_UNET = ("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44)))
_VAE = [
    ("first_stage_model.decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
    ("first_stage_model.encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
]
_TEXT_ENCODER = ("conditioner.embedders.0.weight", "F32", (1, 2), bytes(range(50, 58)))


def _build_safetensors(
    tensors: list[tuple[str, str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
) -> bytes:
    """Assemble a minimal valid safetensors container from ``(name, dtype, shape, data)`` tuples."""
    header: dict[str, object] = {}
    buffer = bytearray()
    for name, dtype, shape, data in tensors:
        begin = len(buffer)
        buffer += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, len(buffer)]}
    if metadata is not None:
        header["__metadata__"] = metadata
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + bytes(buffer)


def _write(tmp_path: Path, name: str, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> Path:
    path = tmp_path / name
    path.write_bytes(_build_safetensors(tensors))
    return path


def _monolithic(tmp_path: Path, name: str = "ckpt.safetensors") -> Path:
    """Write a monolithic checkpoint embedding a UNet tensor, a VAE, and a text encoder."""
    return _write(tmp_path, name, [_UNET, *_VAE, _TEXT_ENCODER])


def _tensor_bytes(*tensors: tuple[str, str, tuple[int, ...], bytes]) -> int:
    return sum(len(data) for _, _, _, data in tensors)


def test_ensure_read_round_trip(tmp_path: Path) -> None:
    """Ensure writes a sidecar that read returns as an equal model with both components detected."""
    ckpt = _monolithic(tmp_path)
    produced = ensure_sidecar(ckpt)

    assert sidecar_path_for(ckpt).exists()
    assert produced.ckpt_file_name == "ckpt.safetensors"
    assert produced.ckpt_size_bytes == ckpt.stat().st_size
    assert set(produced.embedded) == {ComponentKind.VAE.value, ComponentKind.TEXT_ENCODERS.value}
    assert read_sidecar(ckpt) == produced


def test_tensor_byte_accounting(tmp_path: Path) -> None:
    """Per-component and total tensor bytes are summed from the header, and residual is the remainder."""
    ckpt = _monolithic(tmp_path)
    sidecar = ensure_sidecar(ckpt)

    assert sidecar.total_tensor_bytes == _tensor_bytes(_UNET, *_VAE, _TEXT_ENCODER)
    assert sidecar.embedded[ComponentKind.VAE.value].tensor_bytes == _tensor_bytes(*_VAE)
    assert sidecar.embedded[ComponentKind.TEXT_ENCODERS.value].tensor_bytes == _tensor_bytes(_TEXT_ENCODER)
    assert sidecar.residual_tensor_bytes == _tensor_bytes(_UNET)


def test_idempotent_ensure_does_not_rewrite(tmp_path: Path) -> None:
    """A second ensure on a fresh sidecar returns an equal model without rewriting the file."""
    ckpt = _monolithic(tmp_path)
    first = ensure_sidecar(ckpt)
    sidecar_file = sidecar_path_for(ckpt)
    mtime_before = sidecar_file.stat().st_mtime_ns

    second = ensure_sidecar(ckpt)

    assert second == first
    assert sidecar_file.stat().st_mtime_ns == mtime_before


def test_stale_sidecar_is_ignored_and_recomputed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Growing the checkpoint invalidates the sidecar (read returns None, warns), and ensure recomputes it."""
    ckpt = _monolithic(tmp_path)
    original = ensure_sidecar(ckpt)

    # Append a tensor: the file grows, so the recorded size no longer matches.
    ckpt.write_bytes(_build_safetensors([_UNET, *_VAE, _TEXT_ENCODER, ("extra.w", "F16", (2,), bytes(range(4)))]))
    assert ckpt.stat().st_size != original.ckpt_size_bytes

    with caplog.at_level("WARNING"):
        assert read_sidecar(ckpt) is None
    assert any("stale" in message.lower() for message in caplog.messages)

    recomputed = ensure_sidecar(ckpt)
    assert recomputed.ckpt_size_bytes == ckpt.stat().st_size
    assert recomputed != original
    assert read_sidecar(ckpt) == recomputed


def test_malformed_sidecar_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A corrupt sidecar file reads as None with a warning rather than raising."""
    ckpt = _monolithic(tmp_path)
    sidecar_path_for(ckpt).write_text("{ not valid json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert read_sidecar(ckpt) is None
    assert any("malformed" in message.lower() for message in caplog.messages)


def test_vae_extraction_writes_content_addressed_file(tmp_path: Path) -> None:
    """Requesting extraction materialises the VAE under the extraction dir with its hash-derived name."""
    ckpt = _monolithic(tmp_path)
    extraction_dir = tmp_path / "vae"

    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=extraction_dir)

    vae = sidecar.embedded[ComponentKind.VAE.value]
    assert vae.extracted_file_name is not None
    expected_name = f"vae-{vae.content_hash[:16]}.safetensors"
    assert vae.extracted_file_name == expected_name
    extracted = extraction_dir / expected_name
    assert extracted.exists() and extracted.stat().st_size > 0


def test_extracted_vae_hash_matches_recorded_identity(tmp_path: Path) -> None:
    """The standalone hash of the extracted VAE equals the sidecar's recorded embedded VAE content hash."""
    ckpt = _monolithic(tmp_path)
    extraction_dir = tmp_path / "vae"

    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=extraction_dir)
    vae = sidecar.embedded[ComponentKind.VAE.value]
    assert vae.extracted_file_name is not None

    extracted = extraction_dir / vae.extracted_file_name
    assert hash_standalone_component_file(extracted) == vae.content_hash


def test_reensure_repairs_deleted_extraction(tmp_path: Path) -> None:
    """Deleting the extracted VAE and re-ensuring restores it without rewriting from scratch elsewhere."""
    ckpt = _monolithic(tmp_path)
    extraction_dir = tmp_path / "vae"

    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=extraction_dir)
    extracted = extraction_dir / sidecar.embedded[ComponentKind.VAE.value].extracted_file_name  # type: ignore[operator]
    extracted.unlink()
    assert not extracted.exists()

    repaired = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=extraction_dir)
    assert extracted.exists() and extracted.stat().st_size > 0
    assert repaired.embedded[ComponentKind.VAE.value].extracted_file_name == extracted.name


def test_shared_vae_deduplicates_extraction(tmp_path: Path) -> None:
    """Two checkpoints embedding the same VAE payload extract to the same content-addressed filename."""
    ckpt_a = _monolithic(tmp_path, "a.safetensors")
    # A different UNet and text encoder, but the identical VAE tensors.
    ckpt_b = _write(
        tmp_path,
        "b.safetensors",
        [
            ("model.diffusion_model.x", "F16", (3,), bytes(range(60, 66))),
            *_VAE,
            ("conditioner.embedders.0.weight", "F16", (2, 2), bytes(range(9, 17))),
        ],
    )
    extraction_dir = tmp_path / "vae"

    sidecar_a = ensure_sidecar(ckpt_a, extract_vae=True, extraction_dir=extraction_dir)
    sidecar_b = ensure_sidecar(ckpt_b, extract_vae=True, extraction_dir=extraction_dir)

    name_a = sidecar_a.embedded[ComponentKind.VAE.value].extracted_file_name
    name_b = sidecar_b.embedded[ComponentKind.VAE.value].extracted_file_name
    assert name_a == name_b
    assert list(extraction_dir.glob("vae-*.safetensors")) == [extraction_dir / name_a]  # type: ignore[operator]


def test_extract_vae_without_dir_raises(tmp_path: Path) -> None:
    """Requesting extraction without an extraction dir is a usage error and writes no sidecar."""
    ckpt = _monolithic(tmp_path)
    with pytest.raises(ValueError, match="extraction_dir"):
        ensure_sidecar(ckpt, extract_vae=True)
    assert not sidecar_path_for(ckpt).exists()


def test_extract_vae_on_checkpoint_without_vae_is_noop(tmp_path: Path) -> None:
    """A checkpoint with no VAE omits the kind and needs no extraction even when extraction is requested."""
    ckpt = _write(tmp_path, "ckpt.safetensors", [_UNET, _TEXT_ENCODER])
    extraction_dir = tmp_path / "vae"

    sidecar = ensure_sidecar(ckpt, extract_vae=True, extraction_dir=extraction_dir)

    assert ComponentKind.VAE.value not in sidecar.embedded
    assert not extraction_dir.exists() or list(extraction_dir.iterdir()) == []


def test_no_component_checkpoint_omits_kinds(tmp_path: Path) -> None:
    """A checkpoint with no shareable component records no embedded kinds and residual equals the total."""
    ckpt = _write(tmp_path, "unet_only.safetensors", [_UNET])
    sidecar = ensure_sidecar(ckpt)

    assert sidecar.embedded == {}
    assert sidecar.residual_tensor_bytes == sidecar.total_tensor_bytes == _tensor_bytes(_UNET)


def test_unsupported_container_raises_and_writes_nothing(tmp_path: Path) -> None:
    """A non-safetensors container (e.g. a pickle) raises and leaves no sidecar behind."""
    ckpt = tmp_path / "pickled.ckpt"
    ckpt.write_bytes(b"\x80\x04\x95garbage-not-a-safetensors-header")

    with pytest.raises(UnsupportedContainerError):
        ensure_sidecar(ckpt)
    assert not sidecar_path_for(ckpt).exists()


def test_read_sidecar_absent_returns_none(tmp_path: Path) -> None:
    """read_sidecar returns None when no sidecar has been written."""
    ckpt = _monolithic(tmp_path)
    assert read_sidecar(ckpt) is None


def test_embedded_identity_is_frozen() -> None:
    """The identity models are immutable value objects."""
    identity = EmbeddedComponentIdentity(content_hash="abc", tensor_bytes=4)
    with pytest.raises(ValidationError):
        identity.content_hash = "def"  # type: ignore[misc]
    sidecar = ComponentIdentitySidecar(ckpt_file_name="x", ckpt_size_bytes=1, total_tensor_bytes=1)
    with pytest.raises(ValidationError):
        sidecar.ckpt_size_bytes = 2  # type: ignore[misc]
