"""Behavior of the legacy instruct-format migration heuristic against the real text reference."""

from pathlib import Path

import pytest

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix
from horde_model_reference.text_guidance import TextGuidanceStatus
from horde_model_reference.text_guidance_migration import build_legacy_migration_change_set
from horde_model_reference.text_guidance_store import TextGuidanceStore


def _canonical_records(manager: ModelReferenceManager) -> dict[str, dict[str, object]]:
    """Return canonical text records without the projected backend-prefixed duplicates."""
    records = manager.get_raw_model_reference_json(MODEL_REFERENCE_CATEGORY.text_generation) or {}
    return {
        model_name: record
        for model_name, record in records.items()
        if isinstance(record, dict) and not has_legacy_text_backend_prefix(model_name)
    }


def _instruct_format(record: dict[str, object]) -> str | None:
    """Return the trimmed legacy label carried by a record, if it has one."""
    label = record.get("instruct_format")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return None


def test_migration_publishes_every_labeled_model_and_converges(
    model_reference_manager: ModelReferenceManager,
    tmp_path: Path,
) -> None:
    """Seeding from the live reference documents each labeled model once and then proposes nothing."""
    records = _canonical_records(model_reference_manager)
    labeled = {name: record for name, record in records.items() if _instruct_format(record) is not None}
    if not labeled:
        pytest.skip("No converted text_generation reference with instruct_format labels is available locally.")

    store = TextGuidanceStore(root_path=tmp_path / "guidance")
    preview = build_legacy_migration_change_set(
        records,
        existing_profiles=[],
        current_assignment=store.get_assignment,
    )
    assert preview.change_set is not None
    assert preview.source_model_count == len(labeled)

    store.apply_change_set(
        preview.change_set,
        canonical_model_names=set(records),
        editor_id="migration-test",
    )

    distinct_labels = {str(_instruct_format(record)).casefold() for record in labeled.values()}
    assert len(store.list_profiles()) == len(distinct_labels)
    for model_name, record in labeled.items():
        resolved = store.resolve(model_name, legacy_instruct_format=_instruct_format(record))
        assert resolved.summary.status is TextGuidanceStatus.PUBLISHED
        assert resolved.primary_profile is not None

    rerun = build_legacy_migration_change_set(
        records,
        existing_profiles=store.list_profiles(include_deprecated=True),
        current_assignment=store.get_assignment,
    )
    assert rerun.change_set is None
