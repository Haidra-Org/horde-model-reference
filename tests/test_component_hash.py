"""Tests for the torch-free tensor-region component hash.

These pin the hash's contract so consumers in other repos (hordelib, the worker) can reproduce it: a fixed
synthetic input hashes to a fixed digest (the cross-repo conformance vector), the hash is independent of
header ordering and container metadata but sensitive to dtype and bytes, and an embedded VAE hashes equal to
the same VAE extracted to a standalone file.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from horde_model_reference.component_hash import (
    ComponentKind,
    NoComponentTensorsError,
    UnsupportedContainerError,
    component_kind_for_purpose,
    hash_embedded_component_file,
    hash_standalone_component_file,
)

# Fixed digest of the canonical standalone input below. Any change here is a wire-format change that must be
# mirrored in every consumer that recomputes the hash.
_CONFORMANCE_DIGEST = "099674a60c4e178cecbc4c7fa70f970e7aa6f27f543d1e604539d56d00aee931"

_CANONICAL_STANDALONE: list[tuple[str, str, tuple[int, ...], bytes]] = [
    ("decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
    ("encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
]


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


def _write(tmp_path: Path, name: str, tensors: list[tuple[str, str, tuple[int, ...], bytes]], **kw: object) -> Path:
    path = tmp_path / name
    path.write_bytes(_build_safetensors(tensors, **kw))  # type: ignore[arg-type]
    return path


def test_conformance_vector(tmp_path: Path) -> None:
    """The canonical synthetic input hashes to the pinned cross-repo digest."""
    path = _write(tmp_path, "vae.safetensors", _CANONICAL_STANDALONE)
    assert hash_standalone_component_file(path) == _CONFORMANCE_DIGEST


def test_header_order_independent(tmp_path: Path) -> None:
    """Reordering tensors in the header does not change the hash."""
    forward = _write(tmp_path, "a.safetensors", _CANONICAL_STANDALONE)
    reversed_ = _write(tmp_path, "b.safetensors", list(reversed(_CANONICAL_STANDALONE)))
    assert hash_standalone_component_file(forward) == hash_standalone_component_file(reversed_)


def test_container_metadata_ignored(tmp_path: Path) -> None:
    """A ``__metadata__`` entry does not change the hash."""
    plain = _write(tmp_path, "a.safetensors", _CANONICAL_STANDALONE)
    with_meta = _write(tmp_path, "b.safetensors", _CANONICAL_STANDALONE, metadata={"format": "pt", "note": "x"})
    assert hash_standalone_component_file(plain) == hash_standalone_component_file(with_meta)


def test_dtype_is_significant(tmp_path: Path) -> None:
    """Changing a tensor's dtype changes the hash."""
    base = _write(tmp_path, "a.safetensors", _CANONICAL_STANDALONE)
    altered = _write(
        tmp_path,
        "b.safetensors",
        [("decoder.conv_in.weight", "F32", (2, 2), bytes(range(1, 9))), _CANONICAL_STANDALONE[1]],
    )
    assert hash_standalone_component_file(base) != hash_standalone_component_file(altered)


def test_bytes_are_significant(tmp_path: Path) -> None:
    """Changing a tensor's bytes changes the hash."""
    base = _write(tmp_path, "a.safetensors", _CANONICAL_STANDALONE)
    altered = _write(
        tmp_path,
        "b.safetensors",
        [("decoder.conv_in.weight", "F16", (2, 2), bytes(range(2, 10))), _CANONICAL_STANDALONE[1]],
    )
    assert hash_standalone_component_file(base) != hash_standalone_component_file(altered)


@pytest.mark.parametrize("vae_prefix", ["first_stage_model.", "vae."])
def test_embedded_vae_matches_standalone(tmp_path: Path, vae_prefix: str) -> None:
    """An embedded VAE hashes equal to the same VAE extracted to a standalone file."""
    standalone = _write(tmp_path, "vae.safetensors", _CANONICAL_STANDALONE)
    embedded_tensors = [
        ("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44))),
        (f"{vae_prefix}decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
        ("conditioner.embedders.0.y", "F32", (1,), bytes(range(50, 54))),
        (f"{vae_prefix}encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
    ]
    checkpoint = _write(tmp_path, "ckpt.safetensors", embedded_tensors)
    assert hash_embedded_component_file(checkpoint, ComponentKind.VAE) == hash_standalone_component_file(standalone)


def test_embedded_text_encoder_is_hashed(tmp_path: Path) -> None:
    """Embedded text encoders hash consistently across checkpoints, independent of UNet/VAE and stable."""
    te = [
        ("conditioner.embedders.0.weight", "F32", (1, 2), bytes(range(16, 24))),
        ("conditioner.embedders.1.weight", "F16", (2, 2), bytes(range(1, 9))),
    ]
    checkpoint = _write(
        tmp_path,
        "ckpt.safetensors",
        [("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44))), *te],
    )
    te_hash = hash_embedded_component_file(checkpoint, ComponentKind.TEXT_ENCODERS)
    # A different checkpoint carrying the same text encoders (different UNet) hashes equal.
    other_unet = ("model.diffusion_model.x", "F16", (3,), bytes(range(60, 66)))
    other = _write(tmp_path, "other.safetensors", [other_unet, *te])
    assert hash_embedded_component_file(other, ComponentKind.TEXT_ENCODERS) == te_hash


def test_embedded_vae_missing_raises(tmp_path: Path) -> None:
    """A checkpoint with no VAE tensors raises."""
    checkpoint = _write(tmp_path, "ckpt.safetensors", [("model.diffusion_model.x", "F16", (2,), bytes(range(4)))])
    with pytest.raises(NoComponentTensorsError):
        hash_embedded_component_file(checkpoint, ComponentKind.VAE)


def test_non_safetensors_rejected(tmp_path: Path) -> None:
    """A non-safetensors container (e.g. a pickle) is refused."""
    path = tmp_path / "pickled.ckpt"
    path.write_bytes(b"\x80\x04\x95garbage-not-a-safetensors-header")
    with pytest.raises(UnsupportedContainerError):
        hash_standalone_component_file(path)


def test_purpose_to_kind() -> None:
    """``file_purpose`` strings map to the right component kind."""
    assert component_kind_for_purpose("vae") is ComponentKind.VAE
    assert component_kind_for_purpose("text_encoders") is ComponentKind.TEXT_ENCODERS
    assert component_kind_for_purpose("text_encoder") is ComponentKind.TEXT_ENCODERS
    assert component_kind_for_purpose("unet") is None
    assert component_kind_for_purpose(None) is None
