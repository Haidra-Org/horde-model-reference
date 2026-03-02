from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from loguru import logger

from horde_model_reference.audit.events import AuditDomain, AuditEvent, AuditOperation, AuditPayload
from horde_model_reference.audit.reader import AuditTrailReader


@dataclass(slots=True)
class ReplayResult:
    """Summary of an audit replay pass."""

    state: dict[str, dict[str, Any]]
    last_event_id: int | None
    applied_events: int


class AuditReplayer:
    """Reconstructs state by applying audit events sequentially."""

    def __init__(self, *, reader: AuditTrailReader) -> None:
        """Initialize the replayer with a reader instance."""
        self._reader = reader

    def reconstruct_state(
        self,
        *,
        domain: AuditDomain,
        category: str,
        model_names: Collection[str] | None = None,
        min_event_id: int | None = None,
        max_event_id: int | None = None,
    ) -> ReplayResult:
        """Replay events and return the resulting record state."""
        state: dict[str, dict[str, Any]] = {}
        last_event_id: int | None = None
        applied_events = 0

        for event in self._reader.iter_events(
            domains={domain},
            categories={category},
            model_names=model_names,
            min_event_id=min_event_id,
            max_event_id=max_event_id,
        ):
            self._apply_event(state, event)
            last_event_id = event.event_id
            applied_events += 1

        return ReplayResult(state=state, last_event_id=last_event_id, applied_events=applied_events)

    def _apply_event(self, state: dict[str, dict[str, Any]], event: AuditEvent) -> None:
        payload = event.payload
        model_name = event.model_name

        if event.operation == AuditOperation.CREATE:
            state[model_name] = _snapshot_from_payload(payload) or {}
            return

        if event.operation == AuditOperation.UPDATE:
            if payload and payload.after:
                state[model_name] = dict(payload.after)
                return

            delta_applied = self._apply_delta(state, model_name, payload)
            if not delta_applied:
                logger.warning(
                    "Unable to apply delta for model '%s' (event %s); snapshot missing",
                    model_name,
                    event.event_id,
                )
            return

        if event.operation == AuditOperation.DELETE:
            state.pop(model_name, None)

    def _apply_delta(self, state: dict[str, dict[str, Any]], model_name: str, payload: AuditPayload | None) -> bool:
        if not payload or not payload.delta:
            return False

        snapshot = state.get(model_name)
        if snapshot is None:
            return False

        snapshot = dict(snapshot)
        for field, change in payload.delta.items():
            snapshot[field] = change.new
        state[model_name] = snapshot
        return True


def _snapshot_from_payload(payload: AuditPayload | None) -> dict[str, Any] | None:
    if not payload:
        return None

    if payload.after is not None:
        return dict(payload.after)

    if payload.before is not None:
        return dict(payload.before)

    return None
