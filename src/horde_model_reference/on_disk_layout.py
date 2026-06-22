"""Torch-free knowledge of where canonical model weights live on disk.

This is the single place that answers, without importing torch or ComfyUI (as was previously done in hordelib):
under which directory the model category folders live, which folder a category uses, where each file a record
declares actually sits (including component files that route to sibling folders), and whether those files are
present. Third-party consumers of :mod:`horde_model_reference` get a documented, explicit on-disk layout API
rather than having to reverse-engineer worker- or hordelib-specific conventions.

The weights root is distinct from the reference-JSON location managed by
:class:`horde_model_reference.path_consts.HordeModelReferencePaths`: reference JSON lives under
``{cache_home}/horde_model_reference/`` while the weights live under the resolved weights root (e.g.
``<root>/compvis`` for image-generation checkpoints).

Presence is an EXISTENCE-ONLY check (no checksum hashing): fast but "unverified". A present file could still
be corrupt; :mod:`horde_model_reference.download_engine` remains the authority on integrity. ``Path.exists``
follows symlinks, so a model symlinked onto another disk counts as present, while a broken symlink does not.

Multiple roots are supported so a deployment can spread files across disks: presence and path resolution
search ``[root, *extra_roots]`` and the first existing copy wins, but resolution always *targets* the
primary ``root`` when no copy exists yet (so new downloads remain deterministic).
"""

from __future__ import annotations

import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from horde_model_reference.meta_consts import (
    MODEL_REFERENCE_CATEGORY,
    get_category_descriptor,
    get_weights_marker_folders,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = [
    "COMPONENT_PURPOSE_FOLDERS",
    "PresenceSummary",
    "category_folder",
    "component_relative_path",
    "file_paths_for",
    "free_bytes_for",
    "is_present",
    "presence_summary",
    "resolve_weights_root",
]

_DEFAULT_CACHE_HOME = "models"


COMPONENT_PURPOSE_FOLDERS: dict[str, str] = {
    "vae": "vae",
    "text_encoders": "text_encoders",
    "text_encoder": "text_encoders",
}
"""Multi-file model components whose ``file_purpose`` routes them to a sibling folder.

A model such as Qwen-Image ships its unet, VAE and text-encoder as separate files. ComfyUI's component
loaders look for the VAE in ``<root>/vae`` and the text-encoder in ``<root>/text_encoders``, not in the
owning category's folder (e.g. ``<root>/compvis``). Keys are ``DownloadRecord.file_purpose`` values; values
are the destination folder names. Anything not listed here (e.g. ``unet``/checkpoints) stays in the
category's own folder.
"""


def component_relative_path(file_name: str, file_purpose: str | None) -> Path:
    """Return a path for *file_name* relative to its category folder, honouring component routing.

    Components with a recognised ``file_purpose`` (see :data:`COMPONENT_PURPOSE_FOLDERS`) are redirected to
    the matching sibling folder via a ``../<folder>`` prefix so the component loaders find them; every other
    file stays in the category's own folder. The ``..``-relative form resolves correctly for any category
    folder that lives directly under the weights root.
    """
    if file_purpose:
        folder = COMPONENT_PURPOSE_FOLDERS.get(file_purpose)
        if folder:
            return Path("..") / folder / file_name
    return Path(file_name)


def resolve_weights_root(
    cache_home: str | os.PathLike[str] | None = None,
    *,
    marker_subdirs: tuple[str, ...] | None = None,
) -> Path:
    """Return the directory under which model category folders live, torch-free.

    Starts at *cache_home* (or ``$AIWORKER_CACHE_HOME``, or ``"models"``), then breadth-first searches for a
    directory that contains every folder in *marker_subdirs* (the signature of an existing model tree, by
    default the registry's weights-marker folders such as ``compvis`` and ``clip``); falls back to the base
    directory when none is found.

    Args:
        cache_home: Explicit base directory. When ``None``, ``$AIWORKER_CACHE_HOME`` is used (default
            ``"models"``). Pass an explicit value to avoid depending on the environment.
        marker_subdirs: Folder names that must all be present to identify the model tree. Defaults to
            :func:`horde_model_reference.meta_consts.get_weights_marker_folders`.

    Returns:
        The resolved weights-root directory.
    """
    markers = get_weights_marker_folders() if marker_subdirs is None else marker_subdirs
    if cache_home is not None:
        base = os.fspath(cache_home)
    else:
        base = os.environ.get("AIWORKER_CACHE_HOME", _DEFAULT_CACHE_HOME)
    marker_set = set(markers)
    queue: deque[str] = deque([base])
    while queue:
        dirpath = queue.popleft()
        try:
            dirnames = next(os.walk(dirpath))[1]
        except StopIteration:
            continue
        if marker_set.issubset(dirnames):
            return Path(dirpath)
        for dirname in dirnames:
            queue.append(os.path.join(dirpath, dirname))
    return Path(base)


def category_folder(category: MODEL_REFERENCE_CATEGORY | str) -> str | None:
    """Return the on-disk weights folder for *category*, or None when it has none or is unknown."""
    try:
        return get_category_descriptor(category).on_disk_folder_name
    except KeyError:
        return None


def _category_relative_paths(record: GenericModelRecord) -> list[Path]:
    """Return each declared file's path relative to the weights root (empty when undeterminable)."""
    category = record.category
    if category is None:
        return []
    folder = category_folder(category)
    if folder is None:
        return []
    return [
        Path(folder) / component_relative_path(download.file_name, download.file_purpose)
        for download in record.config.download
    ]


def file_paths_for(
    record: GenericModelRecord,
    root: Path,
    *,
    extra_roots: Sequence[Path] = (),
) -> list[Path]:
    """Return the resolved on-disk path of every file *record* declares (empty when undeterminable).

    For each declared file the search order is ``[root, *extra_roots]`` and the first existing copy is
    returned; when no copy exists, the path under the primary *root* is returned (the download target).
    """
    relative_paths = _category_relative_paths(record)
    search_roots = (root, *extra_roots)
    resolved: list[Path] = []
    for relative_path in relative_paths:
        candidates = [Path(os.path.normpath(search_root / relative_path)) for search_root in search_roots]
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        resolved.append(existing if existing is not None else candidates[0])
    return resolved


def is_present(
    record: GenericModelRecord,
    root: Path,
    *,
    extra_roots: Sequence[Path] = (),
) -> bool:
    """Return whether every file *record* declares exists under any of the roots (existence-only).

    Searches ``[root, *extra_roots]`` for each file; a model whose files are spread across disks counts as
    present. ``Path.exists`` follows symlinks, so a symlinked file counts only when its target exists.
    """
    relative_paths = _category_relative_paths(record)
    if not relative_paths:
        return False
    search_roots = (root, *extra_roots)
    return all(
        any(Path(os.path.normpath(search_root / relative_path)).exists() for search_root in search_roots)
        for relative_path in relative_paths
    )


@dataclass(frozen=True)
class PresenceSummary:
    """Which of a requested set of model names are on disk, missing, or absent from the reference.

    The single canonical answer to "of these configured models, which are present?", so consumers (a
    worker's download plan, its live readiness count, its download-trigger missing-set) all derive
    presence the same way instead of each re-deriving it (and drifting). ``present`` + ``missing`` +
    ``unknown`` partition the requested names, preserving input order within each bucket.
    """

    present: tuple[str, ...]
    """Names whose every declared file exists on disk (existence-only, not integrity-checked)."""
    missing: tuple[str, ...]
    """Names that have a reference record but are not (yet) fully on disk."""
    unknown: tuple[str, ...]
    """Requested names with no record in the reference (so presence cannot be determined)."""

    @property
    def num_present(self) -> int:
        """How many requested models are present on disk."""
        return len(self.present)

    @property
    def num_requested(self) -> int:
        """Total requested names (present + missing + unknown)."""
        return len(self.present) + len(self.missing) + len(self.unknown)


def presence_summary(
    reference: Mapping[str, GenericModelRecord],
    names: Iterable[str],
    root: Path,
    *,
    extra_roots: Sequence[Path] = (),
) -> PresenceSummary:
    """Partition *names* into present / missing / unknown against *reference* and the on-disk *root*.

    A name with no record in *reference* lands in ``unknown`` (its presence is undeterminable, distinct
    from a known model that is simply not downloaded yet). Every other name is routed through
    :func:`is_present` so the per-record routing (component sibling folders, multi-root search) is applied
    exactly once, here.
    """
    present: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    for name in names:
        record = reference.get(name)
        if record is None:
            unknown.append(name)
        elif is_present(record, root, extra_roots=extra_roots):
            present.append(name)
        else:
            missing.append(name)
    return PresenceSummary(present=tuple(present), missing=tuple(missing), unknown=tuple(unknown))


def free_bytes_for(root: Path) -> int | None:
    """Return free bytes on the volume holding *root* (cwd when it does not exist yet), or None."""
    probe = root if root.exists() else Path.cwd()
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None
