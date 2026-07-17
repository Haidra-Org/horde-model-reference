"""Per-checkpoint sidecar recording the identity of the components a monolithic checkpoint embeds.

A monolithic checkpoint (SD1.5/SDXL ``.safetensors``) carries its VAE and text encoders inside the same file
as the UNet. Consumers (hordelib, the worker) need a cheap, torch-free answer to "which shareable components
does this checkpoint embed, and by what content identity?" so they can group checkpoints that share a VAE or
text encoder and materialise the shared component once. Recomputing that per process would re-read the
checkpoint every start; instead it is computed once and cached next to the file as a small JSON sidecar.

The sidecar records, per :class:`~horde_model_reference.component_hash.ComponentKind`, the component's
container-independent content hash (:mod:`horde_model_reference.component_hash`) and the byte size of its
tensor data, plus the checkpoint's total tensor bytes. From those a consumer derives the UNet-ish residual
(``total - vae - text_encoders``) without a second pass, exposed as
:attr:`ComponentIdentitySidecar.residual_tensor_bytes`.

Identity and sidecar naming are content-addressed so caching is self-consistent across machines:

* the sidecar lives at ``<checkpoint-name>.component-identity.json`` (the full checkpoint filename plus a fixed
  suffix), so it sits beside its checkpoint and never collides with another checkpoint's sidecar;
* an optionally extracted standalone VAE is named ``vae-<first-16-hex-of-content-hash>.safetensors``, so two
  checkpoints embedding byte-identical VAE weights extract to the same filename and deduplicate for free.

Cost and safety mirror :mod:`horde_model_reference.component_hash`: only the safetensors header is parsed and
only the component tensor byte ranges are read (never the whole checkpoint), no torch/numpy/``safetensors``
import occurs, and every write (sidecar and extracted VAE) is atomic (temp file then replace) so an interrupted
run never leaves a corrupt artifact. A pickle ``.ckpt`` cannot be parsed torch-free, so
:class:`~horde_model_reference.component_hash.UnsupportedContainerError` is raised rather than silently
swallowed; callers decide the fallback.

This is a library module: no CLI, no environment reads, no dependency on
:class:`~horde_model_reference.model_reference_manager.ModelReferenceManager`.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

# The header parse and tensor selection are reused from component_hash rather than reimplemented: parsing the
# checkpoint once and feeding the same selected tensors into both the hash and the byte-size sums keeps this
# module in lockstep with the identity component_hash defines (same prefixes, same ordering) and avoids a
# second read of a multi-gigabyte checkpoint.
from horde_model_reference.component_hash import (
    ComponentKind,
    LocalFileRegionReader,
    _hash_selected,
    _parse_safetensors_header,
    _select_embedded_tensors,
    _select_standalone_tensors,
    extract_embedded_vae,
)

__all__ = [
    "ComponentIdentitySidecar",
    "EmbeddedComponentIdentity",
    "ensure_sidecar",
    "read_sidecar",
    "sidecar_path_for",
]

_SIDECAR_SUFFIX = ".component-identity.json"
"""Suffix appended to the full checkpoint filename to name its sidecar (``model.safetensors`` yields
``model.safetensors.component-identity.json``), so the sidecar is unambiguous even when two checkpoints in a
folder differ only by extension."""

_CURRENT_SCHEMA_VERSION = 1


class EmbeddedComponentIdentity(BaseModel):
    """The identity of one component (VAE or text encoders) embedded in a monolithic checkpoint."""

    model_config = ConfigDict(frozen=True)

    content_hash: str
    """The container-independent content hash of this component's tensors (see
    :func:`horde_model_reference.component_hash.hash_embedded_component`)."""
    tensor_bytes: int
    """Sum of this component's tensor data byte sizes, from the safetensors header's ``data_offsets``."""
    extracted_file_name: str | None = None
    """VAE only: the standalone file materialised under the extraction (``vae/``) folder, or ``None`` when no
    extraction was requested or the kind is not extractable (text encoders are never extracted)."""


class ComponentIdentitySidecar(BaseModel):
    """Cached component identity for a single monolithic checkpoint, written beside the checkpoint file."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = _CURRENT_SCHEMA_VERSION
    """Version of this sidecar's structure, so a consumer can detect and re-derive an outdated layout."""
    ckpt_file_name: str
    """The checkpoint's file name (not a full path), recorded for provenance when the sidecar is read alone."""
    ckpt_size_bytes: int
    """The checkpoint's size in bytes when the sidecar was computed; a mismatch marks the sidecar stale."""
    total_tensor_bytes: int
    """Sum of every tensor's data byte size in the checkpoint (all tensors, not just embedded components)."""
    embedded: dict[str, EmbeddedComponentIdentity] = Field(default_factory=dict)
    """Detected embedded components keyed by :class:`~horde_model_reference.component_hash.ComponentKind` value
    (``"vae"``, ``"text_encoders"``); a kind with no detectable tensors is simply absent."""

    @property
    def residual_tensor_bytes(self) -> int:
        """Tensor bytes not attributed to any detected embedded component (the UNet-ish remainder).

        Derived as ``total_tensor_bytes`` minus every embedded component's ``tensor_bytes``; equals the total
        when the checkpoint embeds no shareable component.
        """
        accounted = sum(component.tensor_bytes for component in self.embedded.values())
        return self.total_tensor_bytes - accounted


def sidecar_path_for(ckpt_path: Path) -> Path:
    """Return the sidecar path for *ckpt_path* (``<name>.component-identity.json`` beside the checkpoint)."""
    return ckpt_path.with_name(ckpt_path.name + _SIDECAR_SUFFIX)


def _extraction_file_name(content_hash: str) -> str:
    """Return the content-addressed standalone-VAE filename for a VAE with *content_hash*."""
    return f"vae-{content_hash[:16]}.safetensors"


def _parse_sidecar_file(path: Path) -> ComponentIdentitySidecar | None:
    """Load and validate a sidecar file, returning None when it is missing or malformed (no staleness check)."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return ComponentIdentitySidecar.model_validate_json(raw)
    except PydanticValidationError:
        return None


def _write_sidecar_atomic(ckpt_path: Path, sidecar: ComponentIdentitySidecar) -> None:
    """Write *sidecar* beside *ckpt_path* atomically (temp file then replace) so no partial file is left."""
    path = sidecar_path_for(ckpt_path)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(sidecar.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _extract_vae_named(reader: LocalFileRegionReader, content_hash: str, extraction_dir: Path) -> str:
    """Materialise the checkpoint's VAE under *extraction_dir* with its content-addressed name; return that name.

    The write is skipped when a nonzero file already exists at the content-addressed path: identical VAE weights
    always yield the same filename, so an existing copy is byte-equivalent by construction and re-extraction
    would be wasted work. The extraction itself is atomic (see
    :func:`horde_model_reference.component_hash.extract_embedded_vae`).
    """
    extraction_dir.mkdir(parents=True, exist_ok=True)
    dest = extraction_dir / _extraction_file_name(content_hash)
    if not (dest.exists() and dest.stat().st_size > 0):
        extract_embedded_vae(reader, dest)
    return dest.name


def _compute_sidecar(
    ckpt_path: Path,
    ckpt_size_bytes: int,
    *,
    extract_vae: bool,
    extraction_dir: Path | None,
) -> ComponentIdentitySidecar:
    """Parse *ckpt_path* once and build its sidecar, optionally extracting the embedded VAE."""
    embedded: dict[str, EmbeddedComponentIdentity] = {}
    with LocalFileRegionReader(ckpt_path) as reader:
        header, data_start = _parse_safetensors_header(reader)
        total_tensor_bytes = sum(tensor.end - tensor.begin for tensor in _select_standalone_tensors(header))
        for kind in ComponentKind:
            tensors = _select_embedded_tensors(header, kind)
            if not tensors:
                continue
            content_hash = _hash_selected(reader, data_start, tensors)
            tensor_bytes = sum(tensor.end - tensor.begin for tensor in tensors)
            extracted_file_name: str | None = None
            if kind is ComponentKind.VAE and extract_vae:
                # extraction_dir is guaranteed non-None here by ensure_sidecar's precondition check.
                assert extraction_dir is not None
                extracted_file_name = _extract_vae_named(reader, content_hash, extraction_dir)
            embedded[kind.value] = EmbeddedComponentIdentity(
                content_hash=content_hash,
                tensor_bytes=tensor_bytes,
                extracted_file_name=extracted_file_name,
            )
    return ComponentIdentitySidecar(
        ckpt_file_name=ckpt_path.name,
        ckpt_size_bytes=ckpt_size_bytes,
        total_tensor_bytes=total_tensor_bytes,
        embedded=embedded,
    )


def _ensure_vae_extraction(
    ckpt_path: Path,
    sidecar: ComponentIdentitySidecar,
    extraction_dir: Path,
) -> ComponentIdentitySidecar:
    """Ensure the recorded VAE is materialised on disk, returning an updated sidecar when a change was needed.

    Returns *sidecar* unchanged (same instance) when the content-addressed VAE file already exists and the
    sidecar already records its name; otherwise (re)extracts and returns a copy with ``extracted_file_name``
    set. The VAE entry is assumed present (checked by the caller).
    """
    vae = sidecar.embedded[ComponentKind.VAE.value]
    expected_name = _extraction_file_name(vae.content_hash)
    dest = extraction_dir / expected_name
    if vae.extracted_file_name == expected_name and dest.exists() and dest.stat().st_size > 0:
        return sidecar
    extraction_dir.mkdir(parents=True, exist_ok=True)
    if not (dest.exists() and dest.stat().st_size > 0):
        with LocalFileRegionReader(ckpt_path) as reader:
            extract_embedded_vae(reader, dest)
    updated_vae = vae.model_copy(update={"extracted_file_name": expected_name})
    updated_embedded = dict(sidecar.embedded)
    updated_embedded[ComponentKind.VAE.value] = updated_vae
    return sidecar.model_copy(update={"embedded": updated_embedded})


def read_sidecar(ckpt_path: Path) -> ComponentIdentitySidecar | None:
    """Return the cached sidecar for *ckpt_path*, or None when it is absent, malformed, or stale.

    A malformed sidecar (unreadable or failing validation) and a stale one (its recorded ``ckpt_size_bytes`` no
    longer matches the checkpoint on disk) both return None and log a warning, so a consumer treats either as
    "no cached identity" and can recompute via :func:`ensure_sidecar`.
    """
    path = sidecar_path_for(ckpt_path)
    if not path.exists():
        return None
    sidecar = _parse_sidecar_file(path)
    if sidecar is None:
        logger.warning(f"Ignoring malformed component-identity sidecar: {path}")
        return None
    try:
        current_size = ckpt_path.stat().st_size
    except OSError:
        logger.warning(f"Cannot stat checkpoint for sidecar staleness check: {ckpt_path}")
        return None
    if sidecar.ckpt_size_bytes != current_size:
        logger.warning(
            f"Ignoring stale component-identity sidecar for {ckpt_path.name}: "
            f"recorded {sidecar.ckpt_size_bytes} bytes, on disk {current_size} bytes.",
        )
        return None
    return sidecar


def ensure_sidecar(
    ckpt_path: Path,
    *,
    extract_vae: bool = False,
    extraction_dir: Path | None = None,
) -> ComponentIdentitySidecar:
    """Return *ckpt_path*'s component-identity sidecar, computing and writing it when absent or stale.

    Idempotent: a fresh sidecar (its recorded size matching the checkpoint) is returned without rewriting. When
    *extract_vae* is set and the checkpoint embeds a VAE whose extracted file is missing (or whose name was not
    yet recorded), the VAE is (re)extracted and the sidecar updated in place. A checkpoint that embeds no VAE is
    returned unchanged even when *extract_vae* is set (there is nothing to extract).

    Args:
        ckpt_path: The monolithic checkpoint to identify.
        extract_vae: Whether to materialise the embedded VAE as a standalone file. Requires *extraction_dir*.
        extraction_dir: Directory the standalone VAE is written into; required when *extract_vae* is True (no
            location is guessed).

    Returns:
        The checkpoint's component-identity sidecar.

    Raises:
        ValueError: *extract_vae* is True but *extraction_dir* is None.
        UnsupportedContainerError: *ckpt_path* is not a parseable safetensors container (e.g. a pickle ``.ckpt``).
    """
    if extract_vae and extraction_dir is None:
        raise ValueError("extraction_dir is required when extract_vae=True.")

    current_size = ckpt_path.stat().st_size

    sidecar_file = sidecar_path_for(ckpt_path)
    existing: ComponentIdentitySidecar | None = None
    if sidecar_file.exists():
        parsed = _parse_sidecar_file(sidecar_file)
        if parsed is not None and parsed.ckpt_size_bytes == current_size:
            existing = parsed

    if existing is not None:
        if not extract_vae or ComponentKind.VAE.value not in existing.embedded:
            return existing
        assert extraction_dir is not None
        repaired = _ensure_vae_extraction(ckpt_path, existing, extraction_dir)
        if repaired is not existing:
            _write_sidecar_atomic(ckpt_path, repaired)
        return repaired

    sidecar = _compute_sidecar(
        ckpt_path,
        current_size,
        extract_vae=extract_vae,
        extraction_dir=extraction_dir,
    )
    _write_sidecar_atomic(ckpt_path, sidecar)
    return sidecar
