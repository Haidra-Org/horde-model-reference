from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from horde_model_reference import CanonicalFormat, horde_model_reference_settings
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.diff_service import PendingChangeDiffService
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.models import PendingChangeRecord


@dataclass
class _BackendStub:
    """Stub backend that returns legacy JSON keyed by category."""

    legacy_data: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any]]

    def get_legacy_json(self, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any] | None:
        return self.legacy_data.get(category)


@dataclass
class _ManagerStub:
    state_by_model: dict[str, dict[str, Any] | None]
    backend: _BackendStub

    def get_raw_model_json(self, *, category: MODEL_REFERENCE_CATEGORY, model_name: str) -> dict[str, Any] | None:
        return self.state_by_model.get(model_name)


@dataclass
class _QueueStub:
    records: dict[int, PendingChangeRecord]

    def get_change(self, change_id: int) -> PendingChangeRecord | None:
        return self.records.get(change_id)


def test_bulk_diff_reports_total_requests_even_when_errors() -> None:
    """Counts requested change_ids and returns structured errors when some ids are missing."""
    record = PendingChangeRecord(
        change_id=1,
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_name="model-1",
        operation=AuditOperation.UPDATE,
        payload={"name": "new"},
        requested_by="user",
        requested_username="user",
    )

    manager = _ManagerStub(
        state_by_model={"model-1": {"name": "old"}},
        backend=_BackendStub(
            legacy_data={
                MODEL_REFERENCE_CATEGORY.image_generation: {"model-1": {"name": "old"}},
            },
        ),
    )
    queue = _QueueStub(records={1: record})

    service = PendingChangeDiffService(manager=manager, queue_service=queue)  # type: ignore
    result = service.compute_bulk_diffs([1, 2])

    assert result.total == 2
    assert len(result.diffs) == 1
    assert len(result.errors) == 1

    diff = result.diffs[0]
    assert diff.change_id == 1
    assert diff.net_operation == "modified"
    assert diff.fields_modified == ["name"]
    assert not diff.fields_added and not diff.fields_removed

    missing = result.errors[0]
    assert missing["change_id"] == 2
    assert missing["error_type"] == "NotFound"


def test_fetch_current_state_uses_v2_when_canonical_format_is_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """When canonical_format is 'v2', diff should use get_raw_model_json (v2 path)."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)

    record = PendingChangeRecord(
        change_id=1,
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_name="model-1",
        operation=AuditOperation.UPDATE,
        payload={"name": "v2-new"},
        requested_by="user",
        requested_username="user",
    )

    manager = _ManagerStub(
        state_by_model={"model-1": {"name": "v2-old"}},
        backend=_BackendStub(
            legacy_data={
                MODEL_REFERENCE_CATEGORY.image_generation: {"model-1": {"name": "legacy-old"}},
            },
        ),
    )
    queue = _QueueStub(records={1: record})

    service = PendingChangeDiffService(manager=manager, queue_service=queue)  # type: ignore
    result = service.compute_change_diff(1)

    assert result is not None
    # Should use v2 data ("v2-old"), not legacy data ("legacy-old")
    assert result.current_state == {"name": "v2-old"}
    assert result.net_operation == "modified"
    assert result.fields_modified == ["name"]
