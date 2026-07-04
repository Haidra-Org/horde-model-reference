"""Torch-free, container-independent content hashing of model components (VAE, text encoders).

Cross-process component sharing needs one stable answer to "are these two VAEs (or text encoders) the
same weights?", computed by the reference tooling (which populates the hashes offline). If two checkpoints
embed the same VAE, their recorded hashes must match so the reference can group them and promote the shared
component to the canonical set. The identity is defined here once and imported by every consumer.

The hash is *container-independent*: it folds in each tensor's canonical name, dtype, shape and raw bytes,
but never the safetensors container metadata (``__metadata__``) or the header's key ordering. The same
weights re-serialised into a different file therefore hash identically, while a genuinely different tensor
(e.g. an fp16 vs fp32 VAE) does not.

Two forms are hashed, because a component can appear either as its own file or embedded in a checkpoint:

* **standalone** (:func:`hash_standalone_component`): the file *is* the component (a split-file VAE or
  text-encoder). Every tensor is hashed with its bare key.
* **embedded** (:func:`hash_embedded_component`): the component's tensors are selected out of a monolithic
  checkpoint by their known key prefix, and that prefix is stripped so the result equals the standalone
  hash of the same weights extracted to their own file.

Embedded extraction is supported for the **VAE only**. In a monolithic checkpoint ComfyUI does not merely
prefix the text-encoder weights, it renames and structurally reshuffles them at load time (SD1.5/SDXL fold
``conditioner.embedders.*`` into ``clip_l``/``clip_g`` and convert the OpenCLIP layout, which fuses/splits
tensors), so a torch-free file hash cannot reproduce the standalone form without reimplementing that
loader. Embedded text-encoder hashing therefore raises :class:`UnsupportedComponentError`; text-encoder
sharing is served by the split-file families (Flux/Qwen/Z-Image), whose encoders are already standalone.

This module stays inside :mod:`horde_model_reference`'s torch-free boundary (enforced by
``tests/test_torch_free_imports.py``): it parses the safetensors header with the stdlib and hashes byte
ranges directly, never importing torch, numpy or the ``safetensors`` package, and never materialising a
tensor. Because only the requested byte ranges are read, the same code hashes a local file or a remote one
over HTTP range requests, so the offline pass never downloads a whole multi-gigabyte checkpoint.

Only the safetensors container is supported. A pickle ``.ckpt`` cannot be hashed here without executing
arbitrary code, so :class:`UnsupportedContainerError` is raised for it; callers treat "no hash" as "not
shareable" and fall back to a normal load. That keeps this module both torch-free and pickle-safe.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from strenum import StrEnum

if TYPE_CHECKING:
    import requests

__all__ = [
    "ComponentKind",
    "ComponentTensor",
    "HttpRangeRegionReader",
    "LocalFileRegionReader",
    "NoComponentTensorsError",
    "RangeNotSupportedError",
    "RegionReader",
    "UnsupportedComponentError",
    "UnsupportedContainerError",
    "component_kind_for_purpose",
    "hash_embedded_component",
    "hash_embedded_component_file",
    "hash_embedded_component_url",
    "hash_standalone_component",
    "hash_standalone_component_file",
    "hash_standalone_component_url",
]

_HASH_ALGORITHM = "sha256"
_READ_CHUNK_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 256 * 1024 * 1024
"""Safetensors headers are small (a JSON tensor index); a value far larger than any real header is refused
rather than trusted, so a non-safetensors file cannot drive an unbounded allocation."""


class ComponentKind(StrEnum):
    """A shareable model component whose weights can be content-hashed and deduplicated."""

    VAE = "vae"
    TEXT_ENCODERS = "text_encoders"


_PURPOSE_TO_KIND: dict[str, ComponentKind] = {
    "vae": ComponentKind.VAE,
    "text_encoders": ComponentKind.TEXT_ENCODERS,
    "text_encoder": ComponentKind.TEXT_ENCODERS,
}
"""Maps a :attr:`DownloadRecord.file_purpose` string (which has a singular ``text_encoder`` alias) to the
canonical :class:`ComponentKind`. Kept aligned with ``on_disk_layout.COMPONENT_PURPOSE_FOLDERS``."""


_EMBEDDED_VAE_PREFIXES: tuple[str, ...] = ("first_stage_model.", "vae.")
"""Key prefixes under which a monolithic checkpoint stores its VAE (``first_stage_model.`` on SD1.5/SDXL,
``vae.`` on Flux/Qwen combined dicts). ComfyUI selects the VAE by stripping exactly this prefix
(``comfy/supported_models_base.py`` ``vae_key_prefix``; ``comfy/sd.py`` ``state_dict_prefix_replace``), and
the strip is a pure leading-substring removal with no key renaming, so stripping it here yields the same
bare ``encoder.``/``decoder.`` keys a standalone VAE file carries."""


class UnsupportedContainerError(ValueError):
    """Raised when a file is not a parseable safetensors container (e.g. a pickle ``.ckpt``)."""


class UnsupportedComponentError(ValueError):
    """Raised when embedded extraction is requested for a component kind it is not supported for."""


class NoComponentTensorsError(LookupError):
    """Raised when a container holds no tensors for the requested component."""


class RangeNotSupportedError(RuntimeError):
    """Raised when a remote host ignores an HTTP ``Range`` request and would return the whole file.

    Reading a tensor past the header from a full-body response would hash the wrong bytes, so this is
    surfaced rather than silently producing a corrupt hash.
    """


@dataclass(frozen=True)
class ComponentTensor:
    """One selected tensor: its canonical (prefix-stripped) name plus where its bytes live and their shape."""

    canonical_name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    """Byte offset of the tensor's data, relative to the start of the data segment (after the header)."""
    end: int
    """Exclusive end byte offset of the tensor's data, relative to the start of the data segment."""


class RegionReader(Protocol):
    """A source of byte ranges from a safetensors container (a local file or a remote URL).

    ``iter_region`` is the primitive so arbitrarily large tensor regions stream in bounded memory; ``read``
    is the small-range convenience used for the header.
    """

    def iter_region(self, offset: int, length: int) -> Iterator[bytes]:
        """Yield the ``length`` bytes starting at absolute ``offset``, in chunks."""
        ...

    def read(self, offset: int, length: int) -> bytes:
        """Return the ``length`` bytes starting at absolute ``offset``."""
        ...


class LocalFileRegionReader:
    """Reads byte ranges from an open local file. Use as a context manager to release the handle."""

    def __init__(self, path: str | Path) -> None:
        """Open *path* for reading byte ranges."""
        self._path = Path(path)
        self._handle = self._path.open("rb")

    def __enter__(self) -> LocalFileRegionReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()

    def iter_region(self, offset: int, length: int) -> Iterator[bytes]:
        """Yield ``length`` bytes from ``offset`` in chunks (see :class:`RegionReader`)."""
        self._handle.seek(offset)
        remaining = length
        while remaining > 0:
            chunk = self._handle.read(min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

    def read(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes from ``offset`` (see :class:`RegionReader`)."""
        self._handle.seek(offset)
        return self._handle.read(length)


class HttpRangeRegionReader:
    """Reads byte ranges from a remote safetensors file via HTTP ``Range`` requests.

    One request per region keeps memory bounded regardless of tensor size. Intended for the offline
    reference pass, which can therefore hash a component without downloading the whole checkpoint.
    """

    def __init__(self, url: str, session: requests.Session | None = None, *, timeout: float = 30.0) -> None:
        """Bind to *url*; reuse *session* when given, else own a private one closed on exit."""
        import requests

        self._url = url
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self._timeout = timeout

    def __enter__(self) -> HttpRangeRegionReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the session when this reader created it."""
        if self._owns_session:
            self._session.close()

    def iter_region(self, offset: int, length: int) -> Iterator[bytes]:
        """Yield ``length`` bytes from ``offset`` via a ranged GET (see :class:`RegionReader`)."""
        if length <= 0:
            return
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        with self._session.get(self._url, headers=headers, stream=True, timeout=self._timeout) as response:
            response.raise_for_status()
            if offset > 0 and response.status_code != 206:
                raise RangeNotSupportedError(
                    f"Host returned status {response.status_code} for a Range request to {self._url}.",
                )
            remaining = length
            for chunk in response.iter_content(chunk_size=_READ_CHUNK_BYTES):
                if not chunk:
                    continue
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                remaining -= len(chunk)
                yield chunk
                if remaining <= 0:
                    break

    def read(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes from ``offset`` via a ranged GET (see :class:`RegionReader`)."""
        return b"".join(self.iter_region(offset, length))


def component_kind_for_purpose(file_purpose: str | None) -> ComponentKind | None:
    """Return the :class:`ComponentKind` for a ``DownloadRecord.file_purpose``, or None if not shareable."""
    if file_purpose is None:
        return None
    return _PURPOSE_TO_KIND.get(file_purpose)


def _parse_safetensors_header(reader: RegionReader) -> tuple[dict[str, object], int]:
    """Return the parsed safetensors header and the absolute byte offset where the data segment begins.

    The safetensors layout is an 8-byte little-endian header length, that many bytes of JSON mapping each
    tensor name to ``{"dtype", "shape", "data_offsets"}`` (plus an optional ``__metadata__`` entry), then
    the raw tensor buffer. A length that is absurd for a header, or a header that is not the expected JSON
    object, marks a non-safetensors file rather than a valid one.
    """
    length_bytes = reader.read(0, 8)
    if len(length_bytes) < 8:
        raise UnsupportedContainerError("File is too small to be a safetensors container.")
    (header_len,) = struct.unpack("<Q", length_bytes)
    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        raise UnsupportedContainerError(f"Implausible safetensors header length: {header_len}.")
    header_bytes = reader.read(8, header_len)
    if len(header_bytes) < header_len:
        raise UnsupportedContainerError("Truncated safetensors header.")
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as parse_error:
        raise UnsupportedContainerError("Safetensors header is not valid JSON.") from parse_error
    if not isinstance(header, dict):
        raise UnsupportedContainerError("Safetensors header is not a JSON object.")
    return header, 8 + header_len


def _tensor_from_entry(canonical_name: str, entry: object) -> ComponentTensor | None:
    """Build a :class:`ComponentTensor` from a header entry, or None when the entry is not a tensor."""
    if not isinstance(entry, dict):
        return None
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
        return None
    if len(offsets) != 2:
        return None
    return ComponentTensor(
        canonical_name=canonical_name,
        dtype=dtype,
        shape=tuple(int(dim) for dim in shape),
        begin=int(offsets[0]),
        end=int(offsets[1]),
    )


def _select_standalone_tensors(header: dict[str, object]) -> list[ComponentTensor]:
    """Select every tensor of a standalone component file, keeping its bare key as canonical."""
    selected: list[ComponentTensor] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        tensor = _tensor_from_entry(name, entry)
        if tensor is not None:
            selected.append(tensor)
    return selected


def _select_embedded_vae_tensors(header: dict[str, object]) -> list[ComponentTensor]:
    """Select the VAE tensors from a monolithic checkpoint, stripping the checkpoint prefix to bare keys."""
    selected: list[ComponentTensor] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        prefix = next((candidate for candidate in _EMBEDDED_VAE_PREFIXES if name.startswith(candidate)), None)
        if prefix is None:
            continue
        tensor = _tensor_from_entry(name[len(prefix) :], entry)
        if tensor is not None:
            selected.append(tensor)
    return selected


def _fold_tensor_metadata(digest: hashlib._Hash, tensor: ComponentTensor) -> None:
    """Fold a tensor's identity metadata into *digest* with length prefixes so the encoding is unambiguous.

    Every variable-length field is length-prefixed, so no combination of names, dtypes and shapes can alias
    another; the byte payload is streamed separately, immediately after.
    """
    name_bytes = tensor.canonical_name.encode("utf-8")
    dtype_bytes = tensor.dtype.encode("utf-8")
    digest.update(struct.pack("<Q", len(name_bytes)))
    digest.update(name_bytes)
    digest.update(struct.pack("<Q", len(dtype_bytes)))
    digest.update(dtype_bytes)
    digest.update(struct.pack("<Q", len(tensor.shape)))
    for dim in tensor.shape:
        digest.update(struct.pack("<Q", dim))
    digest.update(struct.pack("<Q", tensor.end - tensor.begin))


def _hash_selected(reader: RegionReader, data_start: int, tensors: list[ComponentTensor]) -> str:
    """Hash *tensors* in ascending canonical-name order so header ordering never affects the result."""
    digest = hashlib.new(_HASH_ALGORITHM)
    for tensor in sorted(tensors, key=lambda item: item.canonical_name):
        _fold_tensor_metadata(digest, tensor)
        for chunk in reader.iter_region(data_start + tensor.begin, tensor.end - tensor.begin):
            digest.update(chunk)
    return digest.hexdigest()


def hash_standalone_component(reader: RegionReader) -> str:
    """Return the normalized content hash of a standalone component file read through *reader*.

    Raises:
        UnsupportedContainerError: The source is not a parseable safetensors container.
        NoComponentTensorsError: The container holds no tensors.
    """
    header, data_start = _parse_safetensors_header(reader)
    tensors = _select_standalone_tensors(header)
    if not tensors:
        raise NoComponentTensorsError("Standalone component container holds no tensors.")
    return _hash_selected(reader, data_start, tensors)


def hash_embedded_component(reader: RegionReader, kind: ComponentKind) -> str:
    """Return the normalized content hash of *kind* extracted from a monolithic checkpoint read via *reader*.

    Only :attr:`ComponentKind.VAE` is supported; the result equals :func:`hash_standalone_component` of the
    same VAE extracted to its own file.

    Raises:
        UnsupportedComponentError: *kind* has no supported embedded extraction (any text encoder).
        UnsupportedContainerError: The source is not a parseable safetensors container.
        NoComponentTensorsError: The checkpoint holds no tensors for *kind*.
    """
    if kind is not ComponentKind.VAE:
        raise UnsupportedComponentError(
            f"Embedded extraction is not supported for {kind}; it is served by split-file component files.",
        )
    header, data_start = _parse_safetensors_header(reader)
    tensors = _select_embedded_vae_tensors(header)
    if not tensors:
        raise NoComponentTensorsError(f"No embedded {kind} tensors found in checkpoint.")
    return _hash_selected(reader, data_start, tensors)


def hash_standalone_component_file(path: str | Path) -> str:
    """Content-hash a standalone component file on local disk. See :func:`hash_standalone_component`."""
    with LocalFileRegionReader(path) as reader:
        return hash_standalone_component(reader)


def hash_standalone_component_url(url: str, session: requests.Session | None = None) -> str:
    """Content-hash a remote standalone component file via HTTP range reads. See :func:`hash_standalone_component`."""
    with HttpRangeRegionReader(url, session=session) as reader:
        return hash_standalone_component(reader)


def hash_embedded_component_file(path: str | Path, kind: ComponentKind) -> str:
    """Content-hash *kind* embedded in a local checkpoint file. See :func:`hash_embedded_component`."""
    with LocalFileRegionReader(path) as reader:
        return hash_embedded_component(reader, kind)


def hash_embedded_component_url(url: str, kind: ComponentKind, session: requests.Session | None = None) -> str:
    """Content-hash *kind* embedded in a remote checkpoint via HTTP ranges. See :func:`hash_embedded_component`."""
    with HttpRangeRegionReader(url, session=session) as reader:
        return hash_embedded_component(reader, kind)
