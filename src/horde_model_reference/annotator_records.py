"""Bridge the torch-free annotator catalog into ``controlnet_annotator`` model-reference records.

:mod:`horde_model_reference.annotator_catalog` is the verified, dependency-light list of the annotator
checkpoint *files* the horde-exposed ``comfyui_controlnet_aux`` preprocessors load. This module turns that
list into proper :class:`~horde_model_reference.model_reference_records.ControlNetAnnotatorModelRecord`
instances so annotators are first-class reference data: mirrored, hash-backfilled and read exactly like every
other category, with no bespoke side-channel.

One record groups the file(s) a single preprocessor needs (files are grouped by their shared ``preprocessors``
set), keyed by ``annotator_<primary control type>``. Each file becomes a
:class:`~horde_model_reference.model_reference_records.DownloadRecord` whose ``file_name`` is rooted at the
controlnet folder's ``annotators/`` subdirectory, so the engine writes it exactly where the package looks and
the package skips its own HuggingFace download.
"""

from __future__ import annotations

from horde_model_reference.annotator_catalog import ANNOTATOR_FILES, AnnotatorFile
from horde_model_reference.download_engine import UNKNOWN_SHA256_SENTINEL
from horde_model_reference.model_reference_records import (
    ControlNetAnnotatorModelRecord,
    DownloadRecord,
    GenericModelRecordConfig,
)

__all__ = ["ANNOTATOR_ON_DISK_SUBFOLDER", "annotator_model_name", "annotator_records"]

ANNOTATOR_ON_DISK_SUBFOLDER = "annotators"
"""Subdirectory under the controlnet weights folder where annotator checkpoints live.

The on-disk path the ``controlnet`` category resolves for a file is ``controlnet/<file_name>``; prefixing each
annotator's ``file_name`` with this subfolder reproduces ``comfyui_controlnet_aux``'s expected
``controlnet/annotators/<repo>/<subfolder>/<filename>`` layout (see
:func:`horde_model_reference.on_disk_layout.annotator_root`)."""


def annotator_model_name(entry: AnnotatorFile) -> str:
    """Return the stable record name for the annotator group *entry* belongs to.

    Derived from the entry's primary horde control type (e.g. ``"annotator_openpose"``). Files that share a
    preprocessor share a control type, so every file in a group yields the same name.
    """
    primary_control_type = entry.control_types[0] if entry.control_types else "misc"
    return f"annotator_{primary_control_type}"


def _download_for(entry: AnnotatorFile) -> DownloadRecord:
    """Build the :class:`DownloadRecord` for one annotator file, placed under the controlnet ``annotators/`` dir."""
    return DownloadRecord(
        file_name=f"{ANNOTATOR_ON_DISK_SUBFOLDER}/{entry.relative_path}",
        file_url=entry.origin_url,
        sha256sum=entry.sha256 or UNKNOWN_SHA256_SENTINEL,
    )


def annotator_records(
    catalog: tuple[AnnotatorFile, ...] = ANNOTATOR_FILES,
) -> dict[str, ControlNetAnnotatorModelRecord]:
    """Return the ``controlnet_annotator`` records derived from *catalog*, keyed by model name.

    Files are grouped by their shared ``preprocessors`` set (one record per preprocessor); within a record the
    files are the ``config.download`` entries in catalog order. The record's ``control_types`` /
    ``preprocessors`` carry the group's metadata so consumers can map a control type to its annotator.
    """
    grouped: dict[str, list[AnnotatorFile]] = {}
    for entry in catalog:
        grouped.setdefault(annotator_model_name(entry), []).append(entry)

    records: dict[str, ControlNetAnnotatorModelRecord] = {}
    for name, entries in grouped.items():
        first = entries[0]
        records[name] = ControlNetAnnotatorModelRecord(
            name=name,
            control_types=list(first.control_types),
            preprocessors=list(first.preprocessors),
            config=GenericModelRecordConfig(download=[_download_for(entry) for entry in entries]),
        )
    return records
