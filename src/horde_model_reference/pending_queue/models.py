"""Pydantic models for the pending change queue: records, status, and payload schemas."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from strenum import StrEnum

from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class PendingChangeStatus(StrEnum):
    """Lifecycle states for queued changes.

    State transitions::

        PENDING → APPROVED → APPLYING → APPLIED
        PENDING → REJECTED

    The ``APPLYING`` state is a transient lock held while the backend write is
    in progress.  If the process crashes during this window, records stuck in
    ``APPLYING`` are detected on restart and logged as warnings.
    """

    PENDING = "pending"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    REJECTED = "rejected"


class PendingChangeRecord(BaseModel):
    """Single pending change tracked by the queue."""

    change_id: int = Field(
        description="Unique monotonic identifier for this change, allocated by PendingQueueStore. "
        "Callers should pass 0 as a sentinel when constructing new records; the store replaces it "
        "with the next available ID in enqueue_change(). After persistence, this is the canonical "
        "identifier used to approve, reject, apply, and audit-trail this change.",
    )
    category: MODEL_REFERENCE_CATEGORY
    model_name: str
    operation: AuditOperation
    payload: dict[str, Any] | None = Field(default=None, description="Serialized model payload for apply job")
    requested_by: str
    requested_username: str
    requested_at: int = Field(default_factory=lambda: int(datetime.now(tz=UTC).timestamp()))
    status: PendingChangeStatus = PendingChangeStatus.PENDING
    notes: str | None = None

    batch_id: int | None = Field(
        default=None,
        description="Groups approved changes for atomic application. Allocated by the store's "
        "separate batch-ID counter when changes are approved. Multiple changes can share a "
        "batch_id. After partial application, remaining approved changes are reassigned to a "
        "new batch_id (see PendingQueueService._handle_partial_batch_apply).",
    )
    batch_title: str | None = None

    approved_by: str | None = None
    approved_username: str | None = None
    approved_at: int | None = None

    rejected_by: str | None = None
    rejected_username: str | None = None
    rejected_at: int | None = None
    reject_reason: str | None = None

    applied_at: int | None = None
    applied_by: str | None = None
    applied_username: str | None = None
    applied_job_id: str | None = Field(
        default=None,
        description="Reservation token set during the APPLYING phase to prevent concurrent "
        "apply attempts on the same change. This is a caller-supplied string (typically a UUID), "
        "not a store-allocated integer like change_id or batch_id.",
    )

    updated_at: int = Field(default_factory=lambda: int(datetime.now(tz=UTC).timestamp()))

    request_metadata: dict[str, Any] | None = None
    """Additional metadata for downstream jobs (e.g., original request body)."""

    related_models: list[str] | None = None
    """Backend-prefixed variant names affected by this change (text_generation only).

    When a text model is created/updated/deleted, the server auto-generates
    backend duplicates (aphrodite/, koboldcpp/). This field lists those variants
    so UI can display them and the apply job writes them atomically.
    """


class PendingQueueFilter(BaseModel):
    """Filter options when listing pending queue entries."""

    statuses: set[PendingChangeStatus] | None = None
    categories: set[MODEL_REFERENCE_CATEGORY] | None = None
    batch_id: int | None = None
    model_name: str | None = None
    requested_by: set[str] | None = None


class PendingBatchResult(BaseModel):
    """Result of processing a batch of pending changes."""

    batch_id: int | None
    batch_title: str
    approved: list[PendingChangeRecord]
    rejected: list[PendingChangeRecord]


class BatchSplitInfo(BaseModel):
    """Information about a batch split that occurred during partial application.

    When a batch is partially applied (some changes applied, others remain APPROVED),
    the remaining changes are reassigned to a new batch ID. This model captures
    the details of that reassignment for client notification.
    """

    original_batch_id: int = Field(description="The batch ID that was partially applied")
    new_batch_id: int = Field(description="The new batch ID assigned to remaining changes")
    reassigned_change_ids: list[int] = Field(
        default_factory=list,
        description="List of change IDs that were reassigned to the new batch",
    )


class MarkAppliedResult(BaseModel):
    """Result of marking a change as applied, including any batch split info."""

    record: PendingChangeRecord = Field(description="The updated change record")
    batch_split: BatchSplitInfo | None = Field(
        default=None,
        description="Populated if partial application triggered a batch split",
    )


class PendingQueuePage(BaseModel):
    """Paginated list of pending change records."""

    items: list[PendingChangeRecord]
    total: int
    offset: int
    limit: int | None


class PendingChangeDiff(BaseModel):
    """Diff between current model state and proposed pending change.

    This model provides a detailed view of what would change if the pending
    change were applied, including field-level diffs for update operations.
    """

    change_id: int
    category: MODEL_REFERENCE_CATEGORY
    model_name: str
    operation: AuditOperation

    current_state: dict[str, Any] | None = Field(
        default=None,
        description="The current model state in the backend (None if model doesn't exist for CREATE)",
    )
    proposed_state: dict[str, Any] | None = Field(
        default=None,
        description="The proposed new state from the pending change payload (None for DELETE)",
    )

    net_operation: str = Field(
        description="Computed net change type: 'added', 'modified', 'deleted', or 'unchanged'",
    )
    field_diffs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of field-level differences between current and proposed state",
    )
    is_critical: bool = Field(
        default=False,
        description="True if any critical fields (baseline, nsfw, etc.) are affected",
    )

    fields_added: list[str] = Field(
        default_factory=list,
        description="List of field paths that will be added",
    )
    fields_removed: list[str] = Field(
        default_factory=list,
        description="List of field paths that will be removed",
    )
    fields_modified: list[str] = Field(
        default_factory=list,
        description="List of field paths that will be modified",
    )


class PendingChangeDiffPage(BaseModel):
    """Bulk diff response for multiple pending changes."""

    diffs: list[PendingChangeDiff] = Field(default_factory=list)
    total: int = 0
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Changes that could not be diffed, with error details",
    )


def now_ts() -> int:
    """Return the current UTC timestamp as an integer."""
    return int(datetime.now(tz=UTC).timestamp())


def ensure_seq(items: Collection[int] | None) -> list[int]:
    """Normalize an optional sequence into a list."""
    if not items:
        return []
    return list(items)
