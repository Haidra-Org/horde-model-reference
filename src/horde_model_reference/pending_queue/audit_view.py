from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from horde_model_reference.audit import AuditDomain, AuditTrailReader
from horde_model_reference.audit.events import AuditEvent, AuditOperation
from horde_model_reference.audit.replay import AuditReplayer
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.diff_utils import (
    FieldDiff,
    NetChangeType,
    compute_field_diffs,
    has_critical_changes,
)
from horde_model_reference.pending_queue.models import PendingChangeStatus


class PendingQueueAuditEvent(BaseModel):
    """Single audit log entry tied to a pending change."""

    event_id: int
    timestamp: int
    action: str
    logical_user_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class PendingQueueAuditChange(BaseModel):
    """Lifecycle view of a pending change reconstructed from audit events."""

    change_id: int
    status: PendingChangeStatus
    operation: AuditOperation | None = None
    category: MODEL_REFERENCE_CATEGORY | None = None
    model_name: str | None = None
    requested_by: str | None = None
    requested_at: int | None = None
    approved_by: str | None = None
    approved_at: int | None = None
    rejected_by: str | None = None
    rejected_at: int | None = None
    reject_reason: str | None = None
    applied_by: str | None = None
    applied_at: int | None = None
    applied_job_id: str | None = None
    batch_id: int | None = None
    batch_title: str | None = None
    events: list[PendingQueueAuditEvent] = Field(default_factory=list)


class PendingQueueAuditBatchSummary(BaseModel):
    """High-level aggregate for a processed pending queue batch."""

    batch_id: int
    batch_title: str | None = None
    approved_by: str | None = None
    approved_at: int | None = None
    applied_at: int | None = None
    approved_change_count: int = 0
    rejected_change_count: int = 0
    applied_change_count: int = 0
    total_change_count: int = 0
    last_event_id: int | None = None


class PendingQueueAuditBatchDetail(PendingQueueAuditBatchSummary):
    """Detailed view combining batch summary with per-change timelines."""

    changes: list[PendingQueueAuditChange] = Field(default_factory=list)


class PendingQueueAuditBatchPage(BaseModel):
    """Cursor-based page of batch summaries."""

    domain: AuditDomain
    batches: list[PendingQueueAuditBatchSummary]
    next_cursor: int | None = None


class PendingQueueAuditCurrentResponse(BaseModel):
    """Snapshot of currently pending (unapproved) changes."""

    domain: AuditDomain
    pending_changes: list[PendingQueueAuditChange]
    total_pending: int
    generated_at: int


@dataclass
class _BatchState:
    batch_id: int
    batch_title: str | None = None
    approved_by: str | None = None
    approved_at: int | None = None
    applied_at: int | None = None
    last_event_id: int | None = None
    approved_change_ids: set[int] = field(default_factory=set)
    rejected_change_ids: set[int] = field(default_factory=set)
    applied_change_ids: set[int] = field(default_factory=set)

    def to_summary(self) -> PendingQueueAuditBatchSummary:
        total = len(self.approved_change_ids | self.rejected_change_ids)
        return PendingQueueAuditBatchSummary(
            batch_id=self.batch_id,
            batch_title=self.batch_title,
            approved_by=self.approved_by,
            approved_at=self.approved_at,
            applied_at=self.applied_at,
            approved_change_count=len(self.approved_change_ids),
            rejected_change_count=len(self.rejected_change_ids),
            applied_change_count=len(self.applied_change_ids),
            total_change_count=total,
            last_event_id=self.last_event_id,
        )


@dataclass
class _ChangeState:
    change_id: int
    status: PendingChangeStatus = PendingChangeStatus.PENDING
    operation: AuditOperation | None = None
    category: MODEL_REFERENCE_CATEGORY | None = None
    model_name: str | None = None
    requested_by: str | None = None
    requested_at: int | None = None
    approved_by: str | None = None
    approved_at: int | None = None
    rejected_by: str | None = None
    rejected_at: int | None = None
    reject_reason: str | None = None
    applied_by: str | None = None
    applied_at: int | None = None
    applied_job_id: str | None = None
    batch_id: int | None = None
    batch_title: str | None = None
    events: list[PendingQueueAuditEvent] = field(default_factory=list)

    def to_public(self) -> PendingQueueAuditChange:
        return PendingQueueAuditChange(
            change_id=self.change_id,
            status=self.status,
            operation=self.operation,
            category=self.category,
            model_name=self.model_name,
            requested_by=self.requested_by,
            requested_at=self.requested_at,
            approved_by=self.approved_by,
            approved_at=self.approved_at,
            rejected_by=self.rejected_by,
            rejected_at=self.rejected_at,
            reject_reason=self.reject_reason,
            applied_by=self.applied_by,
            applied_at=self.applied_at,
            applied_job_id=self.applied_job_id,
            batch_id=self.batch_id,
            batch_title=self.batch_title,
            events=self.events,
        )


class PendingQueueAuditDataset:
    """Reconstruct pending queue lifecycle details from audit events."""

    def __init__(self, *, events: Iterable[AuditEvent]) -> None:
        """Initialize the dataset by replaying the provided audit events."""
        self._events = sorted(events, key=lambda event: event.event_id)
        self._changes: dict[int, _ChangeState] = {}
        self._batches: dict[int, _BatchState] = {}
        self._build_state()

    def _build_state(self) -> None:
        for event in self._events:
            payload = _payload_dict(event)
            if not payload:
                continue
            action = payload.get("action")
            change_id = _parse_change_id(event, payload)
            if action is None or change_id is None:
                if action == "batch_split":
                    self._process_batch_split(payload, event)
                continue

            change = self._changes.setdefault(change_id, _ChangeState(change_id=change_id))
            change.events.append(
                PendingQueueAuditEvent(
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    action=action,
                    logical_user_id=event.logical_user_id,
                    payload=payload,
                )
            )

            if action == "enqueue":
                self._process_enqueue(change, payload, event)
                continue
            if action == "approve":
                self._process_approve(change, payload, event)
                continue
            if action == "reject":
                self._process_reject(change, payload, event)
                continue
            if action == "apply":
                self._process_apply(change, payload, event)
                continue
            if action == "batch_split":
                self._process_batch_split(payload, event)

    def _process_enqueue(self, change: _ChangeState, payload: dict[str, Any], event: AuditEvent) -> None:
        change.status = PendingChangeStatus.PENDING
        change.operation = _coerce_operation(payload.get("operation"))
        change.category = _coerce_category(payload.get("category"))
        change.model_name = payload.get("model")
        change.requested_by = event.logical_user_id
        change.requested_at = event.timestamp

    def _process_approve(self, change: _ChangeState, payload: dict[str, Any], event: AuditEvent) -> None:
        change.status = PendingChangeStatus.APPROVED
        change.approved_by = event.logical_user_id
        change.approved_at = event.timestamp
        change.batch_id = _coerce_int(payload.get("batch_id"))
        change.batch_title = payload.get("batch_title", change.batch_title)
        batch = self._ensure_batch(change.batch_id, change.batch_title)
        if batch is None:
            return
        batch.approved_by = batch.approved_by or event.logical_user_id
        batch.approved_at = batch.approved_at or event.timestamp
        batch.batch_title = change.batch_title or batch.batch_title
        batch.last_event_id = event.event_id
        batch.approved_change_ids.add(change.change_id)

    def _process_reject(self, change: _ChangeState, payload: dict[str, Any], event: AuditEvent) -> None:
        change.status = PendingChangeStatus.REJECTED
        change.rejected_by = event.logical_user_id
        change.rejected_at = event.timestamp
        change.reject_reason = payload.get("reason")
        change.batch_id = change.batch_id or _coerce_int(payload.get("batch_id"))
        change.batch_title = payload.get("batch_title", change.batch_title)
        batch = self._ensure_batch(change.batch_id, change.batch_title)
        if batch is None:
            return
        if batch.approved_by is None:
            batch.approved_by = event.logical_user_id
        if batch.approved_at is None:
            batch.approved_at = event.timestamp
        batch.last_event_id = event.event_id
        batch.rejected_change_ids.add(change.change_id)

    def _process_apply(self, change: _ChangeState, payload: dict[str, Any], event: AuditEvent) -> None:
        change.status = PendingChangeStatus.APPLIED
        change.applied_by = event.logical_user_id
        change.applied_at = event.timestamp
        change.applied_job_id = payload.get("job_id")
        change.batch_id = change.batch_id or _coerce_int(payload.get("batch_id"))
        batch = self._ensure_batch(change.batch_id, change.batch_title)
        if batch is None:
            return
        batch.applied_at = max(batch.applied_at or 0, event.timestamp)
        batch.last_event_id = event.event_id
        batch.applied_change_ids.add(change.change_id)

    def _process_batch_split(self, payload: dict[str, Any], event: AuditEvent) -> None:
        """Handle partial-apply batch split audit events by reassigning change ids."""
        original_batch_id = _coerce_int(payload.get("original_batch_id"))
        new_batch_id = _coerce_int(payload.get("new_batch_id"))
        raw_reassigned = payload.get("reassigned_change_ids", [])
        reassigned_ids = [coerced for value in raw_reassigned if (coerced := _coerce_int(value)) is not None]

        if original_batch_id is None or new_batch_id is None:
            return

        original_batch = self._ensure_batch(original_batch_id, None)
        new_batch = self._ensure_batch(new_batch_id, None)

        if original_batch:
            original_batch.last_event_id = event.event_id
        if new_batch:
            new_batch.last_event_id = event.event_id

        for change_id in reassigned_ids:
            change = self._changes.setdefault(change_id, _ChangeState(change_id=change_id))
            previous_batch = change.batch_id
            change.batch_id = new_batch_id

            if previous_batch is not None:
                batch_state = self._batches.get(previous_batch)
                if batch_state:
                    batch_state.approved_change_ids.discard(change_id)
                    batch_state.rejected_change_ids.discard(change_id)
                    batch_state.applied_change_ids.discard(change_id)

            if new_batch and change.status is PendingChangeStatus.APPROVED:
                new_batch.approved_change_ids.add(change_id)

    def _ensure_batch(self, batch_id: int | None, batch_title: str | None) -> _BatchState | None:
        if batch_id is None:
            return None
        batch = self._batches.get(batch_id)
        if batch is None:
            batch = _BatchState(batch_id=batch_id, batch_title=batch_title)
            self._batches[batch_id] = batch
        elif batch_title:
            batch.batch_title = batch_title
        return batch

    def pending_changes(self) -> list[PendingQueueAuditChange]:
        """Return pending changes (no approvals yet) newest-first."""
        return [
            change.to_public()
            for change in sorted(
                self._changes.values(),
                key=lambda change: (change.requested_at or 0, change.change_id),
                reverse=True,
            )
            if change.status is PendingChangeStatus.PENDING
        ]

    def batches_page(
        self,
        *,
        cursor: int | None,
        limit: int,
    ) -> tuple[list[PendingQueueAuditBatchSummary], int | None]:
        """Return a cursor slice of batch summaries sorted from newest to oldest."""
        batch_ids = sorted(self._batches)
        batch_ids.reverse()
        if cursor is not None:
            batch_ids = [batch_id for batch_id in batch_ids if batch_id < cursor]
        selected = batch_ids[:limit]
        summaries = [self._batches[batch_id].to_summary() for batch_id in selected]
        next_cursor = selected[-1] if len(batch_ids) > limit and selected else None
        return summaries, next_cursor

    def batch_detail(self, batch_id: int) -> PendingQueueAuditBatchDetail | None:
        """Return full change information for the requested batch id."""
        batch = self._batches.get(batch_id)
        if batch is None:
            return None
        changes = [change.to_public() for change in self._changes.values() if change.batch_id == batch_id]
        return PendingQueueAuditBatchDetail(**batch.to_summary().model_dump(), changes=changes)


class ModelNetChange(BaseModel):
    """Net change for a single model across a batch."""

    model_name: str
    category: MODEL_REFERENCE_CATEGORY
    net_operation: NetChangeType
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    field_diffs: list[FieldDiff] = Field(default_factory=list)
    is_critical: bool = False


class BatchNetChangeResponse(BaseModel):
    """Response containing net changes for all models in a batch."""

    batch_id: int
    batch_title: str | None = None
    domain: AuditDomain
    model_changes: list[ModelNetChange] = Field(default_factory=list)
    models_added: int = 0
    models_modified: int = 0
    models_deleted: int = 0
    models_unchanged: int = 0
    total_field_changes: int = 0
    has_critical_changes: bool = False
    generated_at: int


def _payload_dict(event: AuditEvent) -> dict[str, Any]:
    payload = event.payload
    if payload is None:
        return {}
    if payload.after:
        return dict(payload.after)
    if payload.before:
        return dict(payload.before)
    return {}


def _parse_change_id(event: AuditEvent, payload: dict[str, Any]) -> int | None:
    raw = payload.get("change_id")
    if raw is None:
        raw = event.model_name
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _coerce_operation(value: object) -> AuditOperation | None:
    if not isinstance(value, str):
        return None
    try:
        return AuditOperation(value)
    except ValueError:
        return None


def _coerce_category(value: object) -> MODEL_REFERENCE_CATEGORY | None:
    if not isinstance(value, str):
        return None
    try:
        return MODEL_REFERENCE_CATEGORY(value)
    except ValueError:
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def load_pending_queue_audit_dataset(*, root_path: Path, domain: AuditDomain) -> PendingQueueAuditDataset:
    """Create a dataset by scanning audit segments for the pending queue category."""
    reader = AuditTrailReader(root_path=root_path)
    events = list(
        reader.iter_events(
            domains={domain},
            categories={"pending_queue"},
        )
    )
    return PendingQueueAuditDataset(events=events)


def compute_batch_net_changes(
    *,
    root_path: Path,
    domain: AuditDomain,
    batch_id: int,
) -> BatchNetChangeResponse | None:
    """Compute net changes for all models affected by a batch.

    Replays audit events before and after the batch to detect the net effect
    of all operations (add, update, delete) on each model. Models that are
    deleted and re-added with identical content show net_operation=UNCHANGED.

    Args:
        root_path: Path to audit trail root directory.
        domain: Audit domain (legacy or v2).
        batch_id: The batch ID to analyze.

    Returns:
        BatchNetChangeResponse with per-model diffs, or None if batch not found.
    """
    import time

    # Load batch details to get the list of changes and metadata
    dataset = load_pending_queue_audit_dataset(root_path=root_path, domain=domain)
    batch_detail = dataset.batch_detail(batch_id)
    if batch_detail is None:
        return None

    # Get all model categories affected by this batch
    affected_models: dict[tuple[str, str], list[PendingQueueAuditChange]] = {}
    for change in batch_detail.changes:
        if change.category and change.model_name and change.status == PendingChangeStatus.APPLIED:
            key = (change.category.value, change.model_name)
            if key not in affected_models:
                affected_models[key] = []
            affected_models[key].append(change)

    # Initialize reader and replayer for reconstructing state
    reader = AuditTrailReader(root_path=root_path)
    replayer = AuditReplayer(reader=reader)

    model_changes: list[ModelNetChange] = []
    counts = {"added": 0, "modified": 0, "deleted": 0, "unchanged": 0}

    # For each affected model, compute before/after state
    for (category_str, model_name), changes in affected_models.items():
        category = MODEL_REFERENCE_CATEGORY(category_str)

        # Find the event ID range for this batch's changes on this model
        min_event = min((e.event_id for c in changes for e in c.events), default=None)
        max_event = max((e.event_id for c in changes for e in c.events), default=None)

        if min_event is None or max_event is None:
            continue

        # Replay state just before the batch
        before_result = replayer.reconstruct_state(
            domain=domain,
            category=category_str,
            model_names={model_name},
            max_event_id=min_event - 1 if min_event > 1 else None,
        )
        before_state = before_result.state.get(model_name)

        # Replay state just after the batch
        after_result = replayer.reconstruct_state(
            domain=domain,
            category=category_str,
            model_names={model_name},
            max_event_id=max_event,
        )
        after_state = after_result.state.get(model_name)

        # Determine net operation type
        if before_state is None and after_state is not None:
            net_op = NetChangeType.ADDED
            counts["added"] += 1
        elif before_state is not None and after_state is None:
            net_op = NetChangeType.DELETED
            counts["deleted"] += 1
        elif before_state == after_state:
            net_op = NetChangeType.UNCHANGED
            counts["unchanged"] += 1
        else:
            net_op = NetChangeType.MODIFIED
            counts["modified"] += 1

        # Compute field-level diffs
        field_diffs = compute_field_diffs(before_state, after_state)

        # Check if any critical fields changed
        is_critical = has_critical_changes(category, field_diffs)

        model_changes.append(
            ModelNetChange(
                model_name=model_name,
                category=category,
                net_operation=net_op,
                before_state=before_state,
                after_state=after_state,
                field_diffs=field_diffs,
                is_critical=is_critical,
            )
        )

    return BatchNetChangeResponse(
        batch_id=batch_id,
        batch_title=batch_detail.batch_title,
        domain=domain,
        model_changes=model_changes,
        models_added=counts["added"],
        models_modified=counts["modified"],
        models_deleted=counts["deleted"],
        models_unchanged=counts["unchanged"],
        total_field_changes=sum(len(mc.field_diffs) for mc in model_changes),
        has_critical_changes=any(mc.is_critical for mc in model_changes),
        generated_at=int(time.time()),
    )
