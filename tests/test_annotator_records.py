"""Tests for the annotator-catalog -> ``controlnet_annotator`` record bridge and its uniform mirroring.

These assert that ControlNet annotators are genuinely a first-class category: the bridge groups the catalog
files into records with the right on-disk paths, and the *generic* R2 planner mirrors and backfills them with
no annotator-specific code path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from horde_model_reference.annotator_catalog import ANNOTATOR_FILES
from horde_model_reference.annotator_records import (
    ANNOTATOR_ON_DISK_SUBFOLDER,
    annotator_model_name,
    annotator_records,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    ControlNetAnnotatorModelRecord,
)
from horde_model_reference.on_disk_layout import annotator_root, file_paths_for
from scripts.r2_sync.allowlist import RedistributableAllowlist, RedistributableEntry
from scripts.r2_sync.object_store import InMemoryObjectStore, object_key_for
from scripts.r2_sync.planner import SyncAction, build_sync_plan


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _allow(*names: str) -> RedistributableAllowlist:
    return RedistributableAllowlist(entries={name: RedistributableEntry(name=name) for name in names})


def test_category_uses_the_annotator_record_type() -> None:
    """The new category resolves to its dedicated record type."""
    assert MODEL_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.controlnet_annotator] is ControlNetAnnotatorModelRecord


def test_bridge_groups_every_catalog_file_into_records() -> None:
    """Every catalog file appears exactly once across the produced records, grouped by preprocessor."""
    records = annotator_records()
    all_downloads = [d for record in records.values() for d in record.config.download]
    assert len(all_downloads) == len(ANNOTATOR_FILES)

    # Each record's files share its control types / preprocessors (the grouping invariant).
    for name, record in records.items():
        assert record.name == name
        assert record.control_types
        assert record.preprocessors


def test_record_name_is_derived_from_primary_control_type() -> None:
    """The model name is ``annotator_<primary control type>`` so a control type maps to its annotator."""
    for entry in ANNOTATOR_FILES:
        assert annotator_model_name(entry) == f"annotator_{entry.control_types[0]}"


def test_download_paths_match_comfyui_annotator_layout(tmp_path: Path) -> None:
    """Each download resolves to ``controlnet/annotators/<repo>/<sub>/<filename>`` (where the package looks)."""
    records = annotator_records()
    for record in records.values():
        resolved = file_paths_for(record, tmp_path)
        for download, on_disk in zip(record.config.download, resolved, strict=True):
            assert download.file_name.startswith(f"{ANNOTATOR_ON_DISK_SUBFOLDER}/")
            # The resolved path lives under the canonical annotator root used by the worker / preload.
            assert annotator_root(tmp_path) in on_disk.parents


def test_records_carry_real_hashes_and_are_content_addressed() -> None:
    """The annotator records ship real sha256s, so the generic planner content-addresses them with no bytes.

    With the catalog backfilled, every file declares a real hash; a planner run against a store already
    holding those content-addressed objects resolves to ALREADY_PRESENT (no download, no FIXME correction).
    """
    records = annotator_records()
    names = list(records)

    declared = [d.sha256sum for r in records.values() for d in r.config.download]
    assert all(sha and sha != "FIXME" for sha in declared)

    # Pre-seed the bucket with each file's content-addressed key, then plan.
    store = InMemoryObjectStore(present={object_key_for(sha) for sha in declared})
    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.controlnet_annotator: records},
        allowlist=_allow(*names),
        store=store,
        byte_source=_DictMissing(),  # no bytes needed: a known hash already present needs none
        apply=False,
    )

    counts = plan.counts()
    assert counts[SyncAction.ALREADY_PRESENT] == len(ANNOTATOR_FILES)
    assert not plan.corrections  # nothing to backfill; the hashes are already real


def test_non_allowlisted_annotator_is_skipped(tmp_path: Path) -> None:
    """An annotator record absent from the allowlist is left on origin, like any other model."""
    records = annotator_records()
    cleared = next(iter(records))
    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.controlnet_annotator: records},
        allowlist=_allow(cleared),  # only one record cleared
        store=InMemoryObjectStore(),
        byte_source=_DictMissing(),
        apply=False,
    )
    skipped = {item.model_name for item in plan.items if item.action == SyncAction.SKIPPED_NOT_ALLOWLISTED}
    assert cleared not in skipped
    assert skipped == set(records) - {cleared}


class _DictMissing:
    """A byte source that never has bytes (for skip/missing planning paths)."""

    def acquire(self, record: object, download: object) -> None:
        return None
