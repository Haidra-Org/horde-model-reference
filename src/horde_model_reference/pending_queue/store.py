from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from loguru import logger

from horde_model_reference.pending_queue.models import (
    PendingChangeRecord,
    PendingChangeStatus,
    PendingQueueFilter,
    now_ts,
)


class PendingQueueStore:
    """File-backed storage for pending queue records."""

    def __init__(self, *, root_path: Path) -> None:
        """Create a store rooted at the provided filesystem path."""
        self._root_path = root_path
        self._root_path.mkdir(parents=True, exist_ok=True)
        self._changes_path = self._root_path / "changes.json"
        self._state_path = self._root_path / "index.json"
        self._lock = RLock()
        self._changes: dict[int, PendingChangeRecord] = {}
        self._last_change_id = 0
        self._last_batch_id = 0
        self._load_state()
        self._load_changes()

    # ---------------------------- public API ----------------------------
    def enqueue_change(self, record: PendingChangeRecord) -> PendingChangeRecord:
        """Persist a new pending change and allocate an id if needed."""
        with self._lock:
            if record.change_id == 0:
                record.change_id = self._next_change_id_locked()
            stored = record.model_copy(deep=True)
            self._changes[stored.change_id] = stored
            self._persist_locked()
            return stored.model_copy(deep=True)

    def get_change(self, change_id: int) -> PendingChangeRecord | None:
        """Return a copy of the requested change, if available."""
        with self._lock:
            entry = self._changes.get(change_id)
            if entry is None:
                return None
            return entry.model_copy(deep=True)

    def list_changes(
        self,
        *,
        queue_filter: PendingQueueFilter | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[list[PendingChangeRecord], int]:
        """Return filtered records and total count before pagination."""
        with self._lock:
            records = sorted(self._changes.values(), key=lambda record: record.change_id)
            if queue_filter:
                records = [record for record in records if self._matches_filter(record, queue_filter)]
            total = len(records)
            if offset:
                records = records[offset:]
            if limit is not None:
                records = records[:limit]
            return [record.model_copy(deep=True) for record in records], total

    def purge_changes(self, *, queue_filter: PendingQueueFilter | None = None) -> list[PendingChangeRecord]:
        """Delete queue entries matching the provided filter and return removed copies."""
        with self._lock:
            records = sorted(self._changes.values(), key=lambda record: record.change_id)
            if queue_filter:
                records = [record for record in records if self._matches_filter(record, queue_filter)]

            removed: list[PendingChangeRecord] = []
            for record in records:
                removed.append(record.model_copy(deep=True))
                self._changes.pop(record.change_id, None)

            if removed:
                self._persist_locked()

            return removed

    def save_many(self, records: Iterable[PendingChangeRecord]) -> list[PendingChangeRecord]:
        """Persist multiple records atomically."""
        with self._lock:
            stored: list[PendingChangeRecord] = []
            for record in records:
                stored_record = record.model_copy(deep=True)
                self._changes[stored_record.change_id] = stored_record
                stored.append(stored_record)
            self._persist_locked()
            return [record.model_copy(deep=True) for record in stored]

    def get_current_pending_batch_id(self) -> int | None:
        """Return the batch ID of the current open batch (APPROVED but not yet applied).

        Batch ID Semantics:
        - All approved-but-unapplied changes share the same batch ID.
        - A new batch ID is only created when:
          1. No unapplied batch exists (first approval after all batches are applied).
          2. A batch was partially applied (remaining changes get a new batch ID).
        - This ensures approvals are grouped together until application.

        Returns:
            The batch ID of existing APPROVED changes, or None if no open batch exists.
        """
        with self._lock:
            for record in self._changes.values():
                if record.status == PendingChangeStatus.APPROVED and record.batch_id is not None:
                    return record.batch_id
            return None

    def get_or_create_pending_batch_id(self) -> int:
        """Get the current pending batch ID, or create a new one if none exists.

        This method ensures all approved-but-unapplied changes share the same batch ID.
        A new batch ID is only allocated when no APPROVED changes exist.

        Returns:
            The batch ID to use for new approvals.
        """
        with self._lock:
            existing_batch_id = self._get_current_pending_batch_id_locked()
            if existing_batch_id is not None:
                return existing_batch_id
            # No existing unapplied batch, allocate a new one
            self._last_batch_id += 1
            self._persist_state_locked()
            return self._last_batch_id

    def _get_current_pending_batch_id_locked(self) -> int | None:
        """Internal: find existing APPROVED batch ID without acquiring lock."""
        for record in self._changes.values():
            if record.status == PendingChangeStatus.APPROVED and record.batch_id is not None:
                return record.batch_id
        return None

    def has_approved_changes_in_batch(self, batch_id: int) -> bool:
        """Check if any APPROVED changes remain in the specified batch.

        Used after applying changes to determine if the batch was partially applied.

        Args:
            batch_id: The batch ID to check.

        Returns:
            True if APPROVED changes exist in the batch, False otherwise.
        """
        with self._lock:
            for record in self._changes.values():
                if record.batch_id == batch_id and record.status == PendingChangeStatus.APPROVED:
                    return True
            return False

    def get_approved_changes_in_batch(self, batch_id: int) -> list[PendingChangeRecord]:
        """Return all APPROVED changes in the specified batch.

        Args:
            batch_id: The batch ID to filter by.

        Returns:
            List of APPROVED change records in the batch.
        """
        with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._changes.values()
                if record.batch_id == batch_id and record.status == PendingChangeStatus.APPROVED
            ]

    def next_batch_id(self) -> int:
        """Allocate the next batch id unconditionally.

        Note: For normal approval operations, use get_or_create_pending_batch_id()
        to reuse existing unapplied batch IDs. This method is used when a new
        batch ID must be created (e.g., after partial batch application).
        """
        with self._lock:
            self._last_batch_id += 1
            self._persist_state_locked()
            return self._last_batch_id

    # ----------------------------- internals ----------------------------
    def _matches_filter(self, record: PendingChangeRecord, queue_filter: PendingQueueFilter) -> bool:
        if queue_filter.statuses and record.status not in queue_filter.statuses:
            return False
        if queue_filter.categories and record.category not in queue_filter.categories:
            return False
        if queue_filter.batch_id is not None and record.batch_id != queue_filter.batch_id:
            return False
        if queue_filter.requested_by and record.requested_by not in queue_filter.requested_by:
            return False
        if queue_filter.model_name:
            lowered = queue_filter.model_name.lower()
            return lowered in record.model_name.lower()
        return True

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Unable to parse pending queue index: %s", exc)
            return
        self._last_change_id = int(payload.get("last_change_id", 0))
        self._last_batch_id = int(payload.get("last_batch_id", 0))

    def _load_changes(self) -> None:
        if not self._changes_path.exists():
            return
        try:
            payload = json.loads(self._changes_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Unable to parse pending queue state: %s", exc)
            return
        entries = payload if isinstance(payload, list) else []
        for raw_entry in entries:
            try:
                record = PendingChangeRecord.model_validate(raw_entry)
            except ValueError as exc:  # pragma: no cover - defensive
                logger.warning("Skipping malformed pending queue entry: %s", exc)
                continue
            self._changes[record.change_id] = record
        if self._changes:
            self._last_change_id = max(self._last_change_id, max(self._changes))

    def _persist_locked(self) -> None:
        self._persist_state_locked()
        serialized = [record.model_dump(mode="json", exclude_none=True) for record in self._changes.values()]
        _atomic_write_json(self._changes_path, serialized, ensure_ascii=False)

    def _persist_state_locked(self) -> None:
        state_payload = {
            "last_change_id": self._last_change_id,
            "last_batch_id": self._last_batch_id,
        }
        _atomic_write_json(self._state_path, state_payload, ensure_ascii=True)

    def _next_change_id_locked(self) -> int:
        self._last_change_id += 1
        self._persist_state_locked()
        return self._last_change_id

    # ----------------------------- apply locks -----------------------------
    def reserve_for_apply(self, *, change_id: int, reservation_id: str) -> PendingChangeRecord:
        """Mark a change as reserved for application if it is still approved.

        The reservation is recorded on the change via ``applied_job_id`` to prevent
        concurrent apply attempts from issuing duplicate backend mutations.
        """
        with self._lock:
            record = self._changes.get(change_id)
            if record is None:
                raise ValueError(f"Change {change_id} does not exist.")
            if record.status is not PendingChangeStatus.APPROVED:
                raise ValueError(f"Change {change_id} is not approved (status={record.status}).")

            existing_reservation = record.applied_job_id
            if existing_reservation is not None and existing_reservation != reservation_id:
                raise ValueError(
                    f"Change {change_id} is already reserved for apply (job_id={existing_reservation}).",
                )

            if existing_reservation == reservation_id:
                # Idempotent re-entry for the same job id
                return record.model_copy(deep=True)

            updated = record.model_copy(update={"applied_job_id": reservation_id, "updated_at": now_ts()})
            self._changes[change_id] = updated
            self._persist_locked()
            return updated.model_copy(deep=True)

    def clear_reservation_if_matches(self, *, change_id: int, reservation_id: str) -> None:
        """Release a reservation if it still matches and the change is not applied yet."""
        with self._lock:
            record = self._changes.get(change_id)
            if record is None:
                return
            if record.status is not PendingChangeStatus.APPROVED:
                return
            if record.applied_job_id != reservation_id:
                return
            updated = record.model_copy(update={"applied_job_id": None, "updated_at": now_ts()})
            self._changes[change_id] = updated
            self._persist_locked()


def assert_pending(record: PendingChangeRecord) -> PendingChangeRecord:
    """Validate that a record is still pending before mutation."""
    if record.status is not PendingChangeStatus.PENDING:
        raise ValueError(f"Change {record.change_id} is not pending (status={record.status}).")
    return record


def _atomic_write_json(path: Path, payload: object, *, ensure_ascii: bool) -> None:
    """Atomically write JSON content to ``path`` using a temporary file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=ensure_ascii)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
