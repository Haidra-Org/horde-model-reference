from __future__ import annotations

from horde_model_reference.audit.events import AuditDomain, AuditEvent, AuditOperation, AuditPayload
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.audit_view import PendingQueueAuditDataset
from horde_model_reference.pending_queue.models import PendingChangeStatus


def _event(
    *, event_id: int, action: str, change_id: int | None, payload_extra: dict[str, object] | None = None
) -> AuditEvent:
    payload = {"action": action}
    if change_id is not None:
        payload["change_id"] = change_id
    if payload_extra:
        payload.update(payload_extra)
    model_name = str(change_id) if change_id is not None else "queue"
    return AuditEvent.new(
        event_id=event_id,
        domain=AuditDomain.LEGACY,
        category="pending_queue",
        model_name=model_name,
        operation=AuditOperation.UPDATE,
        logical_user_id="user",
        payload=AuditPayload.from_create(payload),
    )


def test_batch_split_reassigns_changes_to_new_batch() -> None:
    """Batch split audit events should move remaining approvals to the new batch id."""
    events = [
        _event(
            event_id=1,
            action="enqueue",
            change_id=1,
            payload_extra={
                "category": MODEL_REFERENCE_CATEGORY.image_generation.value,
                "operation": AuditOperation.UPDATE.value,
                "model": "model-1",
            },
        ),
        _event(
            event_id=2,
            action="enqueue",
            change_id=2,
            payload_extra={
                "category": MODEL_REFERENCE_CATEGORY.image_generation.value,
                "operation": AuditOperation.UPDATE.value,
                "model": "model-2",
            },
        ),
        _event(
            event_id=3,
            action="approve",
            change_id=1,
            payload_extra={"batch_id": 10, "batch_title": "batch"},
        ),
        _event(
            event_id=4,
            action="approve",
            change_id=2,
            payload_extra={"batch_id": 10, "batch_title": "batch"},
        ),
        _event(
            event_id=5,
            action="apply",
            change_id=1,
            payload_extra={"batch_id": 10, "job_id": "job"},
        ),
        _event(
            event_id=6,
            action="batch_split",
            change_id=None,
            payload_extra={
                "original_batch_id": 10,
                "new_batch_id": 11,
                "reassigned_change_ids": [2],
                "reason": "partial_apply",
            },
        ),
    ]

    dataset = PendingQueueAuditDataset(events=events)

    original_batch = dataset.batch_detail(10)
    assert original_batch is not None
    assert {change.change_id for change in original_batch.changes} == {1}
    assert original_batch.changes[0].status is PendingChangeStatus.APPLIED

    new_batch = dataset.batch_detail(11)
    assert new_batch is not None
    assert {change.change_id for change in new_batch.changes} == {2}
    assert new_batch.changes[0].status is PendingChangeStatus.APPROVED
    assert new_batch.changes[0].batch_id == 11
