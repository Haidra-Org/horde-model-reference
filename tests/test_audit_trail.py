from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference import AuditSettings, ReplicateMode, horde_model_reference_settings
from horde_model_reference.audit.events import AuditDomain, AuditOperation
from horde_model_reference.audit.writer import AuditTrailWriter
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import horde_model_reference_paths

LEGACY_DOMAIN = AuditDomain("legacy")
V2_DOMAIN = AuditDomain("v2")
CREATE_OPERATION = AuditOperation("create")
UPDATE_OPERATION = AuditOperation("update")
DELETE_OPERATION = AuditOperation("delete")


def _read_events(category_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not category_dir.exists():
        return events

    for file_path in sorted(category_dir.glob("audit-*.jsonl")):
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    return events


def test_audit_trail_writer_rotates_files(tmp_path: Path) -> None:
    """AuditTrailWriter should rotate files once the size threshold is exceeded."""
    audit_root = tmp_path / "audit"
    writer = AuditTrailWriter(root_path=audit_root, max_file_size_bytes=150)

    for index in range(12):
        writer.append_event(
            domain=LEGACY_DOMAIN,
            category="test_category",
            model_name=f"model_{index}",
            operation=CREATE_OPERATION,
            logical_user_id="user-id",
        )

    category_dir = audit_root / str(LEGACY_DOMAIN) / "test_category"
    files = sorted(category_dir.glob("audit-*.jsonl"))
    assert len(files) >= 2, "Expected rotation to create multiple segment files"

    events = _read_events(category_dir)
    assert len(events) == 12
    assert [event["event_id"] for event in events] == list(range(1, 13))


@pytest.mark.usefixtures("legacy_canonical_mode")
def test_filesystem_backend_emits_audit_events_for_crud(primary_base: Path, tmp_path: Path) -> None:
    """FileSystemBackend should emit audit events for create/update/delete operations."""
    audit_writer = AuditTrailWriter(root_path=tmp_path / "audit")
    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
        skip_startup_metadata_population=True,
        audit_writer=audit_writer,
    )

    category = MODEL_REFERENCE_CATEGORY.image_generation
    model_name = "audit_test_model"

    create_payload = {
        "name": model_name,
        "description": "initial",
    }
    backend.update_model_legacy(category, model_name, create_payload, logical_user_id="u-123")

    update_payload = {
        "name": model_name,
        "description": "updated",
        "extra": "value",
    }
    backend.update_model_legacy(category, model_name, update_payload, logical_user_id="u-123")

    backend.delete_model_legacy(category, model_name, logical_user_id="u-123")

    category_dir = tmp_path / "audit" / str(LEGACY_DOMAIN) / category.value
    events = _read_events(category_dir)

    assert len(events) == 3

    create_event, update_event, delete_event = events

    assert create_event["operation"] == str(CREATE_OPERATION)
    assert create_event["payload"]["after"]["description"] == "initial"

    assert update_event["operation"] == str(UPDATE_OPERATION)
    assert "delta" in update_event["payload"]
    delta = update_event["payload"]["delta"]
    assert delta["description"]["old"] == "initial"
    assert delta["description"]["new"] == "updated"

    assert delete_event["operation"] == str(DELETE_OPERATION)
    assert delete_event["payload"]["before"]["description"] == "updated"


@pytest.mark.usefixtures("v2_canonical_mode")
def test_filesystem_backend_emits_v2_audit_events(primary_base: Path, tmp_path: Path) -> None:
    """V2 writes should produce audit events mirroring legacy behavior."""
    audit_writer = AuditTrailWriter(root_path=tmp_path / "audit")
    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
        skip_startup_metadata_population=True,
        audit_writer=audit_writer,
    )

    category = MODEL_REFERENCE_CATEGORY.image_generation
    model_name = "audit_test_model_v2"

    create_payload = {
        "name": model_name,
        "description": "initial",
    }
    backend.update_model(
        category,
        model_name,
        create_payload,
        logical_user_id="u-123",
        request_id="job-create",
    )

    update_payload = {
        "name": model_name,
        "description": "updated",
        "extra": "value",
    }
    backend.update_model(
        category,
        model_name,
        update_payload,
        logical_user_id="u-123",
        request_id="job-update",
    )

    backend.delete_model(
        category,
        model_name,
        logical_user_id="u-123",
        request_id="job-delete",
    )

    category_dir = tmp_path / "audit" / str(V2_DOMAIN) / category.value
    events = _read_events(category_dir)

    assert len(events) == 3

    create_event, update_event, delete_event = events

    assert create_event["operation"] == str(CREATE_OPERATION)
    assert create_event["request_id"] == "job-create"
    assert create_event["payload"]["after"]["description"] == "initial"

    assert update_event["operation"] == str(UPDATE_OPERATION)
    assert update_event["request_id"] == "job-update"
    delta = update_event["payload"]["delta"]
    assert delta["description"]["old"] == "initial"
    assert delta["description"]["new"] == "updated"

    assert delete_event["operation"] == str(DELETE_OPERATION)
    assert delete_event["request_id"] == "job-delete"
    assert delete_event["payload"]["before"]["description"] == "updated"


def test_audit_path_uses_relative_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(horde_model_reference_paths, "base_path", tmp_path)
    monkeypatch.setattr(
        horde_model_reference_settings,
        "audit",
        AuditSettings(relative_subdir="custom-audit", root_path_override=None),
    )

    expected = tmp_path / "custom-audit"
    assert horde_model_reference_paths.audit_path == expected


def test_audit_path_honors_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override_path = tmp_path / "outside" / "logs"
    monkeypatch.setattr(
        horde_model_reference_settings,
        "audit",
        AuditSettings(root_path_override=str(override_path)),
    )

    assert horde_model_reference_paths.audit_path == override_path.resolve()
