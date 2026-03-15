"""Service for computing diffs between pending changes and current model state.

This module provides the PendingChangeDiffService which computes preview diffs
for pending changes by comparing the proposed payload against the current
model state in the backend.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from horde_model_reference import ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue import PendingQueueService
from horde_model_reference.pending_queue.diff_utils import (
    NetChangeType,
    categorize_field_diffs,
    compute_field_diffs,
    has_critical_changes,
)
from horde_model_reference.pending_queue.models import (
    PendingChangeDiff,
    PendingChangeDiffPage,
    PendingChangeRecord,
)


class PendingChangeDiffService:
    """Service for computing preview diffs for pending changes.

    This service compares pending change payloads against the current model
    state in the backend to produce detailed field-level diffs.
    """

    def __init__(
        self,
        *,
        manager: ModelReferenceManager,
        queue_service: PendingQueueService,
    ) -> None:
        """Initialize the diff service.

        Args:
            manager: The model reference manager for fetching current state.
            queue_service: The pending queue service for fetching change records.
        """
        self._manager = manager
        self._queue_service = queue_service

    def compute_change_diff(self, change_id: int) -> PendingChangeDiff | None:
        """Compute the diff for a single pending change.

        Args:
            change_id: The ID of the pending change to diff.

        Returns:
            PendingChangeDiff with computed field diffs, or None if change not found.
        """
        record = self._queue_service.get_change(change_id)
        if record is None:
            return None

        return self._compute_diff_for_record(record)

    def compute_bulk_diffs(
        self,
        change_ids: list[int],
    ) -> PendingChangeDiffPage:
        """Compute diffs for multiple pending changes.

        Args:
            change_ids: List of change IDs to compute diffs for.

        Returns:
            PendingChangeDiffPage containing all computed diffs and any errors.
        """
        diffs: list[PendingChangeDiff] = []
        errors: list[dict[str, Any]] = []

        for change_id in change_ids:
            try:
                record = self._queue_service.get_change(change_id)
                if record is None:
                    errors.append(
                        {
                            "change_id": change_id,
                            "error": "Change not found",
                            "error_type": "NotFound",
                        }
                    )
                    continue

                diff = self._compute_diff_for_record(record)
                diffs.append(diff)

            except Exception as exc:
                logger.warning(f"Failed to compute diff for change {change_id}: {exc}")
                errors.append(
                    {
                        "change_id": change_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )

        return PendingChangeDiffPage(
            diffs=diffs,
            total=len(change_ids),
            errors=errors,
        )

    def _compute_diff_for_record(self, record: PendingChangeRecord) -> PendingChangeDiff:
        """Compute the diff for a single pending change record.

        Args:
            record: The pending change record to compute diff for.

        Returns:
            PendingChangeDiff with computed field diffs.
        """
        current_state = self._fetch_current_state(record)

        proposed_state = record.payload

        # Determine net operation type based on operation and current state
        net_operation = self._determine_net_operation(
            operation=record.operation,
            current_state=current_state,
            proposed_state=proposed_state,
        )

        # Compute field-level diffs
        field_diffs = compute_field_diffs(current_state, proposed_state)

        # Check for critical changes
        is_critical = has_critical_changes(record.category, field_diffs)

        # Categorize diffs by change type
        fields_added, fields_removed, fields_modified = categorize_field_diffs(field_diffs)

        # Convert FieldDiff objects to dicts for JSON serialization
        field_diffs_serialized = [
            {
                "field_path": diff.field_path,
                "old_value": diff.old_value,
                "new_value": diff.new_value,
                "change_type": diff.change_type.value,
            }
            for diff in field_diffs
        ]

        return PendingChangeDiff(
            change_id=record.change_id,
            category=record.category,
            model_name=record.model_name,
            operation=record.operation,
            current_state=current_state,
            proposed_state=proposed_state,
            net_operation=net_operation.value,
            field_diffs=field_diffs_serialized,
            is_critical=is_critical,
            fields_added=fields_added,
            fields_removed=fields_removed,
            fields_modified=fields_modified,
        )

    def _fetch_current_state(
        self,
        record: PendingChangeRecord,
    ) -> dict[str, Any] | None:
        """Fetch the current model state in the same format as the pending change payload.

        The pending change payload is stored in whatever format the client submitted
        (legacy via v1 API, or v2 via v2 API). The diff must compare like-for-like,
        so we fetch the current state in the matching format based on canonical_format.

        Args:
            record: The pending change record whose format determines the retrieval method.

        Returns:
            The current model state dict, or None if the model doesn't exist.
        """
        if horde_model_reference_settings.canonical_format == "legacy":
            legacy_json = self._manager.backend.get_legacy_json(record.category)
            if legacy_json is None:
                return None
            return legacy_json.get(record.model_name)

        return self._manager.get_raw_model_json(
            category=record.category,
            model_name=record.model_name,
        )

    def _determine_net_operation(
        self,
        *,
        operation: AuditOperation,
        current_state: dict[str, Any] | None,
        proposed_state: dict[str, Any] | None,
    ) -> NetChangeType:
        """Determine the net operation type based on operation and states.

        Args:
            operation: The declared operation type from the pending change.
            current_state: The current model state (None if doesn't exist).
            proposed_state: The proposed new state (None for deletes).

        Returns:
            NetChangeType indicating the effective operation.
        """
        if operation == AuditOperation.CREATE:
            # CREATE on existing model is effectively an update
            if current_state is not None:
                return NetChangeType.MODIFIED
            return NetChangeType.ADDED

        if operation == AuditOperation.DELETE:
            # DELETE on non-existent model is a no-op
            if current_state is None:
                return NetChangeType.UNCHANGED
            return NetChangeType.DELETED

        # UPDATE operation
        if current_state is None:
            # UPDATE on non-existent model is effectively a create
            return NetChangeType.ADDED

        if current_state == proposed_state:
            return NetChangeType.UNCHANGED

        return NetChangeType.MODIFIED
