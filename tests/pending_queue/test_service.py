from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from horde_model_reference.audit import AuditTrailWriter
from horde_model_reference.audit.events import AuditDomain, AuditOperation, AuditPayload
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.models import PendingChangeStatus, PendingQueueFilter
from horde_model_reference.pending_queue.service import PendingQueueService
from horde_model_reference.pending_queue.store import PendingQueueStore

_TEST_REQUESTOR_ID = "requestor"
_TEST_REQUESTOR_USERNAME = "Requester"
_TEST_APPROVER_ID = "approver"
_TEST_APPROVER_USERNAME = "Approver"


@dataclass
class _StubAuditWriter:
    events: list[dict[str, Any]]

    def append_event(
        self,
        *,
        domain: AuditDomain,
        category: str,
        model_name: str,
        operation: AuditOperation,
        logical_user_id: str,
        payload: AuditPayload | None = None,
        request_id: str | None = None,
        timestamp: int | None = None,
    ) -> None:
        self.events.append(
            {
                "domain": domain,
                "category": category,
                "model_name": model_name,
                "operation": operation,
                "logical_user_id": logical_user_id,
                "payload": payload,
                "request_id": request_id,
                "timestamp": timestamp,
            }
        )


@pytest.fixture()
def pending_queue_service(tmp_path: Path) -> tuple[PendingQueueService, _StubAuditWriter]:
    """Provide an isolated queue service backed by a temp directory."""
    queue_root = tmp_path / "pending_queue"
    queue_root.mkdir()
    store = PendingQueueStore(root_path=queue_root)
    audit_stub = _StubAuditWriter(events=[])
    service = PendingQueueService(store=store, audit_writer=cast(AuditTrailWriter, audit_stub))
    return service, audit_stub


def _enqueue(
    service: PendingQueueService,
    *,
    model_name: str,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation,
    operation: AuditOperation = AuditOperation.CREATE,
    payload: dict[str, object] | None = None,
) -> int:
    record = service.enqueue_change(
        category=category,
        model_name=model_name,
        operation=operation,
        payload=payload,
        requestor_id=_TEST_REQUESTOR_ID,
        requestor_username=_TEST_REQUESTOR_USERNAME,
    )
    return record.change_id


def test_list_changes_filters_and_pagination(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """List API behaviors remain stable directly at the service level."""
    service, _ = pending_queue_service
    pending_id = _enqueue(
        service,
        model_name="filter_pending",
        category=MODEL_REFERENCE_CATEGORY.image_generation,
    )
    approved_id = _enqueue(
        service,
        model_name="filter_approved",
        category=MODEL_REFERENCE_CATEGORY.audio_generation,
    )
    second_pending_id = _enqueue(
        service,
        model_name="filter_second",
        category=MODEL_REFERENCE_CATEGORY.image_generation,
    )

    service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="approve-one",
        approved_ids=[approved_id],
        rejected_ids=None,
        reject_reason=None,
    )

    page = service.list_changes(
        queue_filter=PendingQueueFilter(statuses={PendingChangeStatus.PENDING}),
    )
    assert {record.change_id for record in page.items} == {pending_id, second_pending_id}

    page = service.list_changes(
        queue_filter=PendingQueueFilter(categories={MODEL_REFERENCE_CATEGORY.audio_generation}),
    )
    assert [record.change_id for record in page.items] == [approved_id]

    page = service.list_changes(queue_filter=PendingQueueFilter(model_name="second"))
    assert [record.change_id for record in page.items] == [second_pending_id]

    page = service.list_changes(offset=1, limit=1)
    assert page.total == 3
    assert len(page.items) == 1
    assert page.items[0].change_id in {approved_id, second_pending_id}


def test_process_batch_updates_records_and_audits(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """Approvals/rejections update records and emit audit entries."""
    service, audit_writer = pending_queue_service
    approved_id = _enqueue(service, model_name="batch_approve")
    rejected_id = _enqueue(service, model_name="batch_reject")

    result = service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="review",
        approved_ids=[approved_id],
        rejected_ids=[rejected_id],
        reject_reason="needs work",
    )

    assert result.batch_title == "review"
    assert {record.change_id for record in result.approved} == {approved_id}
    assert {record.change_id for record in result.rejected} == {rejected_id}

    approved_record = service.get_change(approved_id)
    assert approved_record is not None
    assert approved_record.status is PendingChangeStatus.APPROVED
    assert approved_record.batch_id == result.batch_id
    assert approved_record.approved_by == _TEST_APPROVER_ID

    rejected_record = service.get_change(rejected_id)
    assert rejected_record is not None
    assert rejected_record.status is PendingChangeStatus.REJECTED
    assert rejected_record.reject_reason == "needs work"

    actions = [event["payload"].after["action"] for event in audit_writer.events]
    assert actions.count("approve") == 1
    assert actions.count("reject") == 1


def test_process_batch_requires_reason_when_rejecting(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """Rejections without a reason should be rejected early."""
    service, _ = pending_queue_service
    rejected_id = _enqueue(service, model_name="reject_missing_reason")

    with pytest.raises(ValueError):
        service.process_batch(
            approver_id=_TEST_APPROVER_ID,
            approver_username=_TEST_APPROVER_USERNAME,
            batch_title="reject-only",
            approved_ids=None,
            rejected_ids=[rejected_id],
            reject_reason=None,
        )


def test_process_batch_reject_only_skips_batch_allocation(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """Reject-only reviews should not bump the batch counter or assign batch ids."""
    service, _ = pending_queue_service
    rejected_id = _enqueue(service, model_name="reject-only")

    result = service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="reject-only",
        approved_ids=None,
        rejected_ids=[rejected_id],
        reject_reason="nope",
    )

    rejected_record = service.get_change(rejected_id)
    assert rejected_record is not None
    assert rejected_record.batch_id is None
    assert result.batch_id is None


def test_mark_applied_transitions_and_audits(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """Approved records transition to APPLIED along with audit output."""
    service, audit_writer = pending_queue_service
    change_id = _enqueue(service, model_name="apply_me")

    service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="approve",
        approved_ids=[change_id],
        rejected_ids=None,
        reject_reason=None,
    )

    result = service.mark_applied(
        change_id=change_id,
        applied_by=_TEST_APPROVER_ID,
        applied_username=_TEST_APPROVER_USERNAME,
        job_id="job-123",
    )

    assert result.record.status is PendingChangeStatus.APPLIED
    assert result.record.applied_by == _TEST_APPROVER_ID
    assert result.record.applied_job_id == "job-123"

    actions = [event["payload"].after["action"] for event in audit_writer.events]
    assert "apply" in actions


def test_mark_applied_requires_approved_status(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """mark_applied should enforce approval state before mutating."""
    service, _ = pending_queue_service
    change_id = _enqueue(service, model_name="apply_without_approval")

    with pytest.raises(ValueError):
        service.mark_applied(
            change_id=change_id,
            applied_by=_TEST_APPROVER_ID,
            applied_username=_TEST_APPROVER_USERNAME,
            job_id="job-321",
        )


def test_mark_applied_accepts_applying_status(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """mark_applied should accept records in APPLYING state (set during reservation)."""
    service, _ = pending_queue_service
    change_id = _enqueue(service, model_name="applying_model")

    service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="approve",
        approved_ids=[change_id],
        rejected_ids=None,
        reject_reason=None,
    )

    # Simulate the reservation step (APPROVED → APPLYING)
    service.reserve_for_apply(change_id=change_id, reservation_id="job-1")

    result = service.mark_applied(
        change_id=change_id,
        applied_by=_TEST_APPROVER_ID,
        applied_username=_TEST_APPROVER_USERNAME,
        job_id="job-1",
    )

    assert result.record.status is PendingChangeStatus.APPLIED


def test_scan_stuck_applying_reverts_records(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """scan_stuck_applying reverts APPLYING records to APPROVED."""
    service, _ = pending_queue_service
    change_id = _enqueue(service, model_name="stuck_model")

    service.process_batch(
        approver_id=_TEST_APPROVER_ID,
        approver_username=_TEST_APPROVER_USERNAME,
        batch_title="approve",
        approved_ids=[change_id],
        rejected_ids=None,
        reject_reason=None,
    )

    service.reserve_for_apply(change_id=change_id, reservation_id="crashed-job")

    # Simulate restart
    reverted = service.scan_stuck_applying()
    assert len(reverted) == 1
    assert reverted[0].change_id == change_id
    assert reverted[0].status is PendingChangeStatus.APPROVED
    assert reverted[0].applied_job_id is None


def test_scan_stuck_applying_returns_empty_when_none(
    pending_queue_service: tuple[PendingQueueService, _StubAuditWriter],
) -> None:
    """scan_stuck_applying returns empty when no APPLYING records exist."""
    service, _ = pending_queue_service
    assert service.scan_stuck_applying() == []
