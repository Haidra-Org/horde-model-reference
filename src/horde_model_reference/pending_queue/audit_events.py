"""Typed audit event models for the pending change queue.

Each queue lifecycle action has a dedicated event class that carries only the
fields relevant to that action.  All classes expose a ``to_audit_dict()``
method that flattens the event to the ``dict[str, Any]`` shape expected by
``AuditPayload.from_create()``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from strenum import StrEnum

from horde_model_reference.audit.events import AuditOperation, AuditPayload
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class PendingQueueAction(StrEnum):
    """Lifecycle actions emitted by the pending queue."""

    ENQUEUE = "enqueue"
    APPROVE = "approve"
    REJECT = "reject"
    APPLY = "apply"
    PURGE = "purge"
    BATCH_SPLIT = "batch_split"


class _PendingQueueEventBase(BaseModel):
    """Shared serialisation helper for all queue events."""

    def to_audit_dict(self) -> dict[str, Any]:
        """Flatten the event to a plain dict suitable for ``AuditPayload.from_create()``."""
        data = self.model_dump(mode="json", exclude_none=True)
        # ``action`` must always be present
        data["action"] = self._action().value
        return data

    def to_audit_payload(self) -> AuditPayload:
        """Convert this event directly to an ``AuditPayload``."""
        return AuditPayload.from_create(self.to_audit_dict())

    def _action(self) -> PendingQueueAction:
        raise NotImplementedError


class EnqueueEvent(_PendingQueueEventBase):
    """A new change was submitted to the pending queue."""

    change_id: int
    operation: AuditOperation
    category: MODEL_REFERENCE_CATEGORY
    model_name: str = Field(serialization_alias="model")

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.ENQUEUE

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialize enqueue-specific fields using enum values and the expected key names."""
        data = super().to_audit_dict()
        # The operation and category fields store enum *values* by convention
        data["operation"] = self.operation.value
        data["category"] = self.category.value
        # Use "model" key as expected by audit_view._process_enqueue
        data["model"] = data.pop("model_name", self.model_name)
        return data


class ApproveEvent(_PendingQueueEventBase):
    """A pending change was approved and assigned to a batch."""

    change_id: int
    batch_id: int | None
    batch_title: str

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.APPROVE


class RejectEvent(_PendingQueueEventBase):
    """A pending change was rejected."""

    change_id: int
    batch_id: int | None
    batch_title: str
    reason: str | None = None

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.REJECT


class ApplyEvent(_PendingQueueEventBase):
    """An approved change was applied to the live dataset."""

    change_id: int
    batch_id: int | None
    job_id: str | None = None

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.APPLY


class PurgeEvent(_PendingQueueEventBase):
    """A queued change was removed without being applied."""

    change_id: int
    category: MODEL_REFERENCE_CATEGORY
    model_name: str = Field(serialization_alias="model")
    requested_by: str
    purged_by_username: str

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.PURGE

    def to_audit_dict(self) -> dict[str, Any]:
        """Serialize purge-specific fields using enum values and the expected key names."""
        data = super().to_audit_dict()
        data["category"] = self.category.value
        data["model"] = data.pop("model_name", self.model_name)
        return data


class BatchSplitEvent(_PendingQueueEventBase):
    """Remaining approved changes were reassigned to a new batch after partial application."""

    original_batch_id: int
    new_batch_id: int
    reassigned_change_ids: list[int]
    reason: str = "partial_apply"

    def _action(self) -> PendingQueueAction:
        return PendingQueueAction.BATCH_SPLIT
