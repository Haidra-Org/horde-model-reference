"""Audit event models and type definitions for the append-only audit trail."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from strenum import StrEnum

from horde_model_reference import CanonicalFormat


class AuditDomain(StrEnum):
    """Supported domains for audit events."""

    LEGACY = CanonicalFormat.LEGACY
    V2 = CanonicalFormat.v2


class AuditOperation(StrEnum):
    """CRUD operations captured by the audit log."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class AuditFieldChange(BaseModel):
    """Represents a field-level delta for update operations."""

    old: Any = Field(description="Previous value")
    new: Any = Field(description="New value")


class AuditPayload(BaseModel):
    """Payload recorded with an audit event (full snapshots or deltas)."""

    before: dict[str, Any] | None = Field(default=None, description="Full record state prior to the change")
    after: dict[str, Any] | None = Field(default=None, description="Full record state after the change")
    delta: dict[str, AuditFieldChange] | None = Field(
        default=None,
        description="Sparse representation of changed fields for updates",
    )

    @staticmethod
    def from_create(record: Mapping[str, Any]) -> AuditPayload:
        """Build payload for create operations using the new record snapshot."""
        return AuditPayload(after=_coerce_mapping(record))

    @staticmethod
    def from_delete(record: Mapping[str, Any]) -> AuditPayload:
        """Build payload for delete operations using the removed record snapshot."""
        return AuditPayload(before=_coerce_mapping(record))

    @staticmethod
    def from_update(before: Mapping[str, Any], after: Mapping[str, Any]) -> AuditPayload:
        """Build payload for update operations using a sparse delta representation."""
        return AuditPayload(delta=_compute_delta(before, after))


class AuditEvent(BaseModel):
    """Single append-only audit event."""

    event_id: int
    timestamp: int = Field(description="Unix timestamp (UTC) when the event was recorded")
    domain: AuditDomain
    category: str
    model_name: str
    operation: AuditOperation
    logical_user_id: str = Field(description="Immutable Horde user identifier")
    request_id: str | None = Field(default=None, description="Optional idempotency or tracing identifier")
    payload: AuditPayload | None = Field(default=None, description="Snapshot or delta payload")

    @staticmethod
    def new(
        *,
        event_id: int,
        domain: AuditDomain,
        category: str,
        model_name: str,
        operation: AuditOperation,
        logical_user_id: str,
        timestamp: int | None = None,
        request_id: str | None = None,
        payload: AuditPayload | None = None,
    ) -> AuditEvent:
        """Create an audit event while filling defaults such as timestamp."""
        return AuditEvent(
            event_id=event_id,
            timestamp=timestamp or int(datetime.now(tz=UTC).timestamp()),
            domain=domain,
            category=category,
            model_name=model_name,
            operation=operation,
            logical_user_id=logical_user_id,
            request_id=request_id,
            payload=payload,
        )


def _coerce_mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in record}


def _compute_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, AuditFieldChange]:
    delta: dict[str, AuditFieldChange] = {}
    before_dict = _coerce_mapping(before)
    after_dict = _coerce_mapping(after)
    keys = set(before_dict) | set(after_dict)
    for key in keys:
        old_value = before_dict.get(key)
        new_value = after_dict.get(key)
        if old_value != new_value:
            delta[key] = AuditFieldChange(old=old_value, new=new_value)
    return delta
