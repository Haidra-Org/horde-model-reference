"""Business logic for the pending change queue: proposal, approval, and application workflows."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from loguru import logger

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.audit import AuditTrailWriter
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.audit_events import (
    ApplyEvent,
    ApproveEvent,
    BatchSplitEvent,
    EnqueueEvent,
    PurgeEvent,
    RejectEvent,
    _PendingQueueEventBase,
)
from horde_model_reference.pending_queue.models import (
    BatchSplitInfo,
    MarkAppliedResult,
    PendingBatchResult,
    PendingChangeRecord,
    PendingChangeStatus,
    PendingQueueFilter,
    PendingQueuePage,
    ensure_seq,
    now_ts,
)
from horde_model_reference.pending_queue.store import PendingQueueStore, assert_pending

_QUEUE_CATEGORY = "pending_queue"


class PendingQueueService:
    """High-level orchestration around the pending queue store."""

    def __init__(self, *, store: PendingQueueStore, audit_writer: AuditTrailWriter | None) -> None:
        """Initialize the service with its storage backend and audit writer."""
        self._store = store
        self._audit_writer = audit_writer

    def enqueue_change(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        operation: AuditOperation,
        payload: dict[str, Any] | None,
        requestor_id: str,
        requestor_username: str,
        notes: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        related_models: list[str] | None = None,
    ) -> PendingChangeRecord:
        """Create a new pending change entry."""
        record = PendingChangeRecord(
            change_id=0,
            category=category,
            model_name=model_name,
            operation=operation,
            payload=payload,
            requested_by=requestor_id,
            requested_username=requestor_username,
            notes=notes,
            request_metadata=request_metadata,
            related_models=related_models,
        )
        persisted = self._store.enqueue_change(record)
        self._write_audit_event(
            logical_user_id=requestor_id,
            event=EnqueueEvent(
                change_id=persisted.change_id,
                operation=operation,
                category=category,
                model_name=model_name,
            ),
        )
        return persisted

    def get_change(self, change_id: int) -> PendingChangeRecord | None:
        """Return a single change if it exists."""
        return self._store.get_change(change_id)

    def list_changes(
        self,
        *,
        queue_filter: PendingQueueFilter | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> PendingQueuePage:
        """Return filtered queue entries plus pagination metadata."""
        items, total = self._store.list_changes(queue_filter=queue_filter, offset=offset, limit=limit)
        return PendingQueuePage(items=items, total=total, offset=offset, limit=limit)

    def purge_changes(
        self,
        *,
        queue_filter: PendingQueueFilter | None,
        purged_by: str,
        purged_username: str,
    ) -> list[PendingChangeRecord]:
        """Remove queued changes matching a filter and emit audit entries."""
        removed = self._store.purge_changes(queue_filter=queue_filter)
        if not removed:
            return []

        for record in removed:
            self._write_audit_event(
                logical_user_id=purged_by,
                event=PurgeEvent(
                    change_id=record.change_id,
                    category=record.category,
                    model_name=record.model_name,
                    requested_by=record.requested_by,
                    purged_by_username=purged_username,
                ),
            )

        return removed

    def process_batch(
        self,
        *,
        approver_id: str,
        approver_username: str,
        batch_title: str,
        approved_ids: Collection[int] | None,
        rejected_ids: Collection[int] | None,
        reject_reason: str | None = None,
    ) -> PendingBatchResult:
        """Approve and/or reject subsets of the current pending queue.

        Batch ID Semantics:
        - All approved-but-unapplied changes share the same batch ID.
        - When approving new changes, they join the existing open batch if one exists.
        - A new batch ID is only created when no APPROVED changes exist (i.e., all
          previous batches have been fully applied or this is the first approval).
        - After partial batch application, remaining APPROVED changes are reassigned
          to a new batch ID (see mark_applied and _handle_partial_batch_apply).
        """
        approved_list = ensure_seq(approved_ids)
        rejected_list = ensure_seq(rejected_ids)
        if not approved_list and not rejected_list:
            raise ValueError("Must approve or reject at least one change.")
        if rejected_list and not reject_reason:
            raise ValueError("reject_reason is required when rejecting changes.")

        # Reuse existing unapplied batch ID if available, otherwise create new one when approving
        batch_id = self._store.get_or_create_pending_batch_id() if approved_list else None
        now = now_ts()

        updated_records: list[PendingChangeRecord] = []
        approved_records: list[PendingChangeRecord] = []
        rejected_records: list[PendingChangeRecord] = []

        for change_id in approved_list:
            record = self._require_pending(change_id)
            updated = record.model_copy(
                update={
                    "status": PendingChangeStatus.APPROVED,
                    "approved_by": approver_id,
                    "approved_username": approver_username,
                    "approved_at": now,
                    "batch_id": batch_id,
                    "batch_title": batch_title,
                    "updated_at": now,
                }
            )
            updated_records.append(updated)
            approved_records.append(updated)

        for change_id in rejected_list:
            record = self._require_pending(change_id)
            updated = record.model_copy(
                update={
                    "status": PendingChangeStatus.REJECTED,
                    "rejected_by": approver_id,
                    "rejected_username": approver_username,
                    "rejected_at": now,
                    "reject_reason": reject_reason,
                    "batch_id": batch_id,
                    "batch_title": batch_title,
                    "updated_at": now,
                }
            )
            updated_records.append(updated)
            rejected_records.append(updated)

        persisted = self._store.save_many(updated_records)
        persisted_lookup = {record.change_id: record for record in persisted}
        approved_records = [persisted_lookup[record.change_id] for record in approved_records]
        rejected_records = [persisted_lookup[record.change_id] for record in rejected_records]

        for record in approved_records:
            self._write_audit_event(
                logical_user_id=approver_id,
                event=ApproveEvent(
                    change_id=record.change_id,
                    batch_id=batch_id,
                    batch_title=batch_title,
                ),
            )
        for record in rejected_records:
            self._write_audit_event(
                logical_user_id=approver_id,
                event=RejectEvent(
                    change_id=record.change_id,
                    batch_id=batch_id,
                    batch_title=batch_title,
                    reason=reject_reason,
                ),
            )

        return PendingBatchResult(
            batch_id=batch_id,
            batch_title=batch_title,
            approved=approved_records,
            rejected=rejected_records,
        )

    def mark_applied(
        self,
        *,
        change_id: int,
        applied_by: str,
        applied_username: str,
        job_id: str | None = None,
    ) -> MarkAppliedResult:
        """Mark an APPLYING change as APPLIED by a downstream job.

        Batch Split Semantics:
        - After applying a change, if other APPROVED changes remain in the same batch,
          this constitutes a "partial apply" and those changes are reassigned to a new
          batch ID.
        - This ensures that the next approval operation creates a fresh batch rather
          than mixing with partially-applied batches.
        - A 'batch_split' audit event is emitted when reassignment occurs.

        Returns:
            MarkAppliedResult containing the updated record and any batch split info.

        """
        record = self._store.get_change(change_id)
        if record is None:
            raise ValueError(f"Change {change_id} not found.")
        if record.status not in {PendingChangeStatus.APPROVED, PendingChangeStatus.APPLYING}:
            raise ValueError("Only approved or applying changes can transition to applied.")

        original_batch_id = record.batch_id
        now = now_ts()
        updated = record.model_copy(
            update={
                "status": PendingChangeStatus.APPLIED,
                "applied_at": now,
                "applied_by": applied_by,
                "applied_username": applied_username,
                "applied_job_id": job_id,
                "updated_at": now,
            }
        )
        persisted = self._store.save_many([updated])[0]
        self._write_audit_event(
            logical_user_id=applied_by,
            event=ApplyEvent(
                change_id=persisted.change_id,
                batch_id=persisted.batch_id,
                job_id=job_id,
            ),
        )

        # Handle partial batch application: reassign remaining APPROVED changes to new batch
        batch_split: BatchSplitInfo | None = None
        if original_batch_id is not None:
            batch_split = self._handle_partial_batch_apply(
                original_batch_id=original_batch_id,
                applied_by=applied_by,
            )

        return MarkAppliedResult(record=persisted, batch_split=batch_split)

    def _handle_partial_batch_apply(
        self,
        *,
        original_batch_id: int,
        applied_by: str,
    ) -> BatchSplitInfo | None:
        """Reassign remaining APPROVED changes to a new batch after partial application.

        When a batch is partially applied (some changes applied, others still APPROVED),
        the remaining APPROVED changes must be moved to a new batch ID. This ensures:
        1. The partially-applied batch is "closed" and won't receive new approvals.
        2. Future approvals will create or join a new batch.
        3. The audit trail clearly shows the batch split event.

        Args:
            original_batch_id: The batch ID that was partially applied.
            applied_by: The user ID who triggered the partial application.

        Returns:
            BatchSplitInfo if a split occurred, None if the batch was fully applied.

        """
        remaining_approved = self._store.get_approved_changes_in_batch(original_batch_id)
        if not remaining_approved:
            # Batch fully applied, no split needed
            return None

        # Allocate a new batch ID for the remaining changes
        new_batch_id = self._store.next_batch_id()
        now = now_ts()

        updated_records: list[PendingChangeRecord] = []
        reassigned_change_ids: list[int] = []
        for record in remaining_approved:
            updated = record.model_copy(
                update={
                    "batch_id": new_batch_id,
                    "updated_at": now,
                }
            )
            updated_records.append(updated)
            reassigned_change_ids.append(record.change_id)

        self._store.save_many(updated_records)

        # Emit audit event for the batch split
        self._write_audit_event(
            logical_user_id=applied_by,
            event=BatchSplitEvent(
                original_batch_id=original_batch_id,
                new_batch_id=new_batch_id,
                reassigned_change_ids=reassigned_change_ids,
            ),
        )

        return BatchSplitInfo(
            original_batch_id=original_batch_id,
            new_batch_id=new_batch_id,
            reassigned_change_ids=reassigned_change_ids,
        )

    def _require_pending(self, change_id: int) -> PendingChangeRecord:
        record = self._store.get_change(change_id)
        if record is None:
            raise ValueError(f"Change {change_id} does not exist.")
        return assert_pending(record)

    def reserve_for_apply(self, *, change_id: int, reservation_id: str) -> PendingChangeRecord:
        """Reserve an approved change for application using a reservation id."""
        return self._store.reserve_for_apply(change_id=change_id, reservation_id=reservation_id)

    def clear_apply_reservation(self, *, change_id: int, reservation_id: str) -> None:
        """Release a reservation when an apply attempt fails."""
        self._store.clear_reservation_if_matches(change_id=change_id, reservation_id=reservation_id)

    def scan_stuck_applying(self) -> list[PendingChangeRecord]:
        """Detect records stuck in APPLYING state after a crash and revert them.

        Should be called once on startup.  Each stuck record is reverted to
        APPROVED so it can be retried, and a warning is logged.

        Returns:
            The records that were reverted.

        """
        stuck = self._store.get_applying_records()
        if not stuck:
            return []

        reverted: list[PendingChangeRecord] = []
        for record in stuck:
            logger.warning(
                "Change %d (%s/%s) was stuck in APPLYING state — reverting to APPROVED",
                record.change_id,
                record.category,
                record.model_name,
            )
            try:
                updated = self._store.revert_applying_to_approved(record.change_id)
                reverted.append(updated)
            except ValueError as exc:
                logger.error("Failed to revert stuck change %d: %s", record.change_id, exc)
        return reverted

    def _write_audit_event(self, *, logical_user_id: str, event: _PendingQueueEventBase) -> None:
        if not self._audit_writer:
            return
        audit_payload = event.to_audit_payload()
        payload_dict = event.to_audit_dict()
        try:
            self._audit_writer.append_event(
                domain=horde_model_reference_settings.canonical_format,
                category=_QUEUE_CATEGORY,
                model_name=str(payload_dict.get("change_id", "queue")),
                operation=AuditOperation.UPDATE,
                logical_user_id=logical_user_id,
                payload=audit_payload,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to emit pending queue audit event: {}", exc)
