from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.pending_queue.apply import (
    PendingChangeBackendError,
    PendingChangeNotFoundError,
    PendingChangePayloadError,
    PendingChangeStateError,
    apply_pending_changes,
)
from horde_model_reference.pending_queue.models import MarkAppliedResult, PendingChangeRecord, PendingChangeStatus
from horde_model_reference.pending_queue.service import PendingQueueService


@dataclass
class _DummyBackend:
    """Backend stub that records update/delete operations."""

    fail_on_models: set[str] | None = None

    def __post_init__(self) -> None:
        self.updated: list[tuple[MODEL_REFERENCE_CATEGORY, str, dict[str, Any]]] = []
        self.deleted: list[tuple[MODEL_REFERENCE_CATEGORY, str]] = []

    def supports_writes(self) -> bool:
        return True

    def supports_legacy_writes(self) -> bool:
        return True

    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        payload: dict[str, Any],
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        if self.fail_on_models and model_name in self.fail_on_models:
            raise RuntimeError("backend update failure")
        self.updated.append((category, model_name, payload))

    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        if self.fail_on_models and model_name in self.fail_on_models:
            raise RuntimeError("backend delete failure")
        self.deleted.append((category, model_name))

    def update_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        payload: dict[str, Any],
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        # Delegate to update_model for testing purposes
        self.update_model(category, model_name, payload, logical_user_id=logical_user_id, request_id=request_id)

    def delete_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        # Delegate to delete_model for testing purposes
        self.delete_model(category, model_name, logical_user_id=logical_user_id, request_id=request_id)


@dataclass
class _DummyManager:
    backend: _DummyBackend


class _DummyQueueService:
    def __init__(self, records: list[PendingChangeRecord], *, fail_mark_ids: set[int] | None = None) -> None:
        self.records = {record.change_id: record for record in records}
        self.applied_ids: list[int] = []
        self.fail_mark_ids = fail_mark_ids or set()

    def get_change(self, change_id: int) -> PendingChangeRecord | None:
        return self.records.get(change_id)

    def mark_applied(
        self,
        *,
        change_id: int,
        applied_by: str,
        applied_username: str,
        job_id: str | None = None,
    ) -> MarkAppliedResult:
        record = self.records.get(change_id)
        if record is None:
            raise ValueError("missing record")
        if change_id in self.fail_mark_ids:
            raise RuntimeError("mark_applied failure")
        updated = record.model_copy(
            update={
                "status": PendingChangeStatus.APPLIED,
                "applied_by": applied_by,
                "applied_username": applied_username,
                "applied_job_id": job_id,
            }
        )
        self.records[change_id] = updated
        self.applied_ids.append(change_id)
        return MarkAppliedResult(record=updated, batch_split=None)

    def reserve_for_apply(self, *, change_id: int, reservation_id: str) -> PendingChangeRecord:
        record = self.records.get(change_id)
        if record is None:
            raise ValueError("missing record")
        if record.status is not PendingChangeStatus.APPROVED:
            raise ValueError("record not approved")
        existing = record.applied_job_id
        if existing is not None and existing != reservation_id:
            raise ValueError("already reserved")
        updated = record.model_copy(update={"applied_job_id": reservation_id})
        self.records[change_id] = updated
        return updated

    def clear_apply_reservation(self, *, change_id: int, reservation_id: str) -> None:
        record = self.records.get(change_id)
        if record is None:
            return
        if record.applied_job_id != reservation_id:
            return
        self.records[change_id] = record.model_copy(update={"applied_job_id": None})


def _approved_record(
    change_id: int,
    *,
    operation: AuditOperation,
    payload: dict[str, Any] | None,
    model_name: str | None = None,
    batch_id: int = 1,
) -> PendingChangeRecord:
    return PendingChangeRecord(
        change_id=change_id,
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_name=model_name or f"model_{change_id}",
        operation=operation,
        payload=payload,
        requested_by="user",
        requested_username="user",
        status=PendingChangeStatus.APPROVED,
        batch_id=batch_id,
    )


def test_apply_pending_changes_applies_all_records() -> None:
    """Applies every change when all records are approved and valid."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "one"}),
        _approved_record(2, operation=AuditOperation.DELETE, payload=None),
    ])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-1",
    )

    assert [record.change_id for record in result.applied_records] == [1, 2]
    assert queue_service_stub.applied_ids == [1, 2]
    assert backend.updated and backend.deleted
    assert result.failed_change_id is None
    assert result.failed_error is None


def test_apply_pending_changes_stops_on_first_error() -> None:
    """Stops iteration when a later change is not ready for apply."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    records = [
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "ok"}),
        PendingChangeRecord(
            change_id=2,
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            model_name="model_2",
            operation=AuditOperation.UPDATE,
            payload={"name": "pending"},
            requested_by="user",
            requested_username="user",
            status=PendingChangeStatus.PENDING,
            batch_id=1,  # Has batch_id but is still PENDING - should fail on status check
        ),
    ]
    queue_service_stub = _DummyQueueService(records)

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-2",
    )

    assert [record.change_id for record in result.applied_records] == [1]
    assert result.failed_change_id == 2
    assert isinstance(result.failed_error, PendingChangeStateError)
    assert queue_service_stub.applied_ids == [1]


def test_apply_pending_changes_handles_missing_change() -> None:
    """Returns failure metadata when a change cannot be found."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "exists"})
    ])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-3",
        enforce_batch_cohesion=False,  # Skip batch validation to test per-change error handling
    )

    assert not result.applied_records
    assert result.failed_change_id == 2
    assert isinstance(result.failed_error, PendingChangeNotFoundError)


def test_apply_pending_changes_reports_backend_failure() -> None:
    """Surfaces backend errors without applying subsequent changes."""
    backend = _DummyBackend(fail_on_models={"model_2"})
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "one"}),
        _approved_record(2, operation=AuditOperation.UPDATE, payload={"name": "fails"}),
    ])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-4",
    )

    assert [record.change_id for record in result.applied_records] == [1]
    assert queue_service_stub.applied_ids == [1]
    assert result.failed_change_id == 2
    assert isinstance(result.failed_error, PendingChangeBackendError)
    assert backend.updated == [
        (
            MODEL_REFERENCE_CATEGORY.image_generation,
            "model_1",
            {"name": "one"},
        )
    ]


def test_apply_pending_changes_reports_delete_failure() -> None:
    """Propagates backend delete errors as PendingChangeBackendError."""
    backend = _DummyBackend(fail_on_models={"model_2"})
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.DELETE, payload=None),
        _approved_record(2, operation=AuditOperation.DELETE, payload=None),
    ])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-5",
    )

    assert [record.change_id for record in result.applied_records] == [1]
    assert queue_service_stub.applied_ids == [1]
    assert result.failed_change_id == 2
    assert isinstance(result.failed_error, PendingChangeBackendError)
    assert backend.deleted == [
        (
            MODEL_REFERENCE_CATEGORY.image_generation,
            "model_1",
        )
    ]


def test_apply_pending_changes_handles_mark_applied_failure() -> None:
    """Treats queue mark failures as backend errors and halts sequencing."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService(
        [
            _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "ok"}),
            _approved_record(2, operation=AuditOperation.UPDATE, payload={"name": "next"}),
        ],
        fail_mark_ids={1},
    )

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 2],
        applied_by="approver",
        applied_username="approver",
        job_id="job-6",
    )

    assert not queue_service_stub.applied_ids
    assert [record.change_id for record in result.applied_records] == []
    assert result.failed_change_id == 1
    assert isinstance(result.failed_error, PendingChangeBackendError)
    assert backend.updated == [
        (
            MODEL_REFERENCE_CATEGORY.image_generation,
            "model_1",
            {"name": "ok"},
        )
    ]


def test_apply_pending_changes_missing_payload_surfaces_error() -> None:
    """Rejects UPDATE/CREATE changes that lack payload content."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([_approved_record(1, operation=AuditOperation.UPDATE, payload=None)])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1],
        applied_by="approver",
        applied_username="approver",
        job_id="job-7",
    )

    assert not result.applied_records
    assert queue_service_stub.applied_ids == []
    assert isinstance(result.failed_error, PendingChangePayloadError)
    assert result.failed_change_id == 1


def test_apply_pending_changes_duplicate_ids_trigger_state_error() -> None:
    """Handles duplicate IDs by applying once then failing on second occurrence."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "dupe"})
    ])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1, 1],
        applied_by="approver",
        applied_username="approver",
        job_id="job-8",
    )

    assert [record.change_id for record in result.applied_records] == [1]
    assert queue_service_stub.applied_ids == [1]
    assert result.failed_change_id == 1
    assert isinstance(result.failed_error, PendingChangeStateError)


def test_apply_pending_changes_invalid_operation_raises_backend_error() -> None:
    """Surfaces unsupported audit operations as backend errors (QA-crafted records)."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([
        _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "ok"})
    ])
    bad_operation = cast(AuditOperation, "bogus_op")
    queue_service_stub.records[1].__dict__["operation"] = bad_operation

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1],
        applied_by="approver",
        applied_username="approver",
        job_id="job-9",
    )

    assert not result.applied_records
    assert queue_service_stub.applied_ids == []
    assert isinstance(result.failed_error, PendingChangeBackendError)
    assert result.failed_change_id == 1


def test_apply_pending_changes_respects_existing_reservation() -> None:
    """Prevent duplicate backend writes when a change is already reserved."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    reserved_record = _approved_record(1, operation=AuditOperation.UPDATE, payload={"name": "locked"})
    reserved_record.__dict__["applied_job_id"] = "other-job"
    queue_service_stub = _DummyQueueService([reserved_record])

    result = apply_pending_changes(
        manager=cast(ModelReferenceManager, manager_stub),
        queue_service=cast(PendingQueueService, queue_service_stub),
        change_ids=[1],
        applied_by="approver",
        applied_username="approver",
        job_id="new-job",
    )

    assert not backend.updated
    assert not queue_service_stub.applied_ids
    assert result.failed_change_id == 1
    assert isinstance(result.failed_error, PendingChangeStateError)


def test_apply_pending_changes_requires_change_ids() -> None:
    """Validates the helper rejects empty change id sequences."""
    backend = _DummyBackend()
    manager_stub = _DummyManager(backend=backend)
    queue_service_stub = _DummyQueueService([])

    with pytest.raises(ValueError):
        apply_pending_changes(
            manager=cast(ModelReferenceManager, manager_stub),
            queue_service=cast(PendingQueueService, queue_service_stub),
            change_ids=[],
            applied_by="approver",
            applied_username="approver",
        )
