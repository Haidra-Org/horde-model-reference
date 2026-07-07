"""Tests for extracting a monolithic checkpoint's embedded VAE to a standalone file (torch-free).

The core invariant is that a sliced standalone VAE hashes equal to the source checkpoint's embedded VAE:
the lane can therefore materialise a standalone VAE from a monolithic checkpoint and the reference's recorded
embedded hash still identifies it. The tensor bytes are copied verbatim, and the write is atomic.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from horde_model_reference.component_hash import (
    ComponentKind,
    NoComponentTensorsError,
    extract_embedded_vae_file,
    hash_embedded_component_file,
    hash_standalone_component_file,
)


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


def _read_safetensors(path: Path) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    """Parse a safetensors file into ``name -> (dtype, shape, data)``, dropping any ``__metadata__``."""
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    data = raw[8 + header_len :]
    parsed: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        parsed[name] = (entry["dtype"], tuple(entry["shape"]), data[begin:end])
    return parsed


def _checkpoint(tmp_path: Path, vae_prefix: str) -> Path:
    """Write a monolithic checkpoint embedding a UNet tensor, a text-encoder tensor, and a two-tensor VAE."""
    tensors = [
        ("model.diffusion_model.x", "F16", (2,), bytes(range(40, 44))),
        (f"{vae_prefix}decoder.conv_in.weight", "F16", (2, 2), bytes(range(1, 9))),
        ("conditioner.embedders.0.y", "F32", (1,), bytes(range(50, 54))),
        (f"{vae_prefix}encoder.conv_out.weight", "F32", (1, 2), bytes(range(16, 24))),
    ]
    path = tmp_path / "ckpt.safetensors"
    path.write_bytes(_build_safetensors(tensors))
    return path


@pytest.mark.parametrize("vae_prefix", ["first_stage_model.", "vae."])
def test_extracted_vae_hash_equals_embedded(tmp_path: Path, vae_prefix: str) -> None:
    """The extracted standalone VAE hashes equal to the source checkpoint's embedded VAE hash."""
    source = _checkpoint(tmp_path, vae_prefix)
    dest = tmp_path / "vae.safetensors"
    returned = extract_embedded_vae_file(source, dest)
    embedded = hash_embedded_component_file(source, ComponentKind.VAE)
    assert returned == embedded
    assert hash_standalone_component_file(dest) == embedded


def test_extracted_file_has_only_bare_vae_keys(tmp_path: Path) -> None:
    """Extraction strips the checkpoint prefix and drops the UNet and text-encoder tensors."""
    source = _checkpoint(tmp_path, "first_stage_model.")
    dest = tmp_path / "vae.safetensors"
    extract_embedded_vae_file(source, dest)
    assert set(_read_safetensors(dest)) == {"decoder.conv_in.weight", "encoder.conv_out.weight"}


def test_extracted_bytes_match_source(tmp_path: Path) -> None:
    """The extracted VAE tensor bytes, dtypes and shapes are copied verbatim from the checkpoint."""
    source = _checkpoint(tmp_path, "first_stage_model.")
    dest = tmp_path / "vae.safetensors"
    extract_embedded_vae_file(source, dest)
    tensors = _read_safetensors(dest)
    assert tensors["decoder.conv_in.weight"] == ("F16", (2, 2), bytes(range(1, 9)))
    assert tensors["encoder.conv_out.weight"] == ("F32", (1, 2), bytes(range(16, 24)))


def test_extract_missing_vae_raises(tmp_path: Path) -> None:
    """A checkpoint with no VAE tensors raises rather than writing an empty file."""
    source = tmp_path / "ckpt.safetensors"
    source.write_bytes(_build_safetensors([("model.diffusion_model.x", "F16", (2,), bytes(range(4)))]))
    dest = tmp_path / "vae.safetensors"
    with pytest.raises(NoComponentTensorsError):
        extract_embedded_vae_file(source, dest)
    assert not dest.exists()


def test_no_temp_file_left_behind(tmp_path: Path) -> None:
    """A successful extraction leaves the destination and no sibling temporary file."""
    source = _checkpoint(tmp_path, "first_stage_model.")
    dest = tmp_path / "vae.safetensors"
    extract_embedded_vae_file(source, dest)
    assert dest.exists()
    assert not (tmp_path / "vae.safetensors.tmp").exists()
