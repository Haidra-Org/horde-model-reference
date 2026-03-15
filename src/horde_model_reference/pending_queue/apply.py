from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from loguru import logger

from horde_model_reference import ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

from .models import BatchSplitInfo, PendingChangeRecord
from .service import PendingQueueService


class BackendUpdateCallable(Protocol):
    """Protocol for backend update operations, supporting both legacy and canonical formats."""

    def __call__(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Protocol for backend update operations, supporting both legacy and canonical formats.

        The callable should perform the necessary update or create operation for the given category and model name,
        using the provided record dictionary as the source of truth for the model's fields. The callable must also
        accept optional parameters for logical user ID and request ID to support auditing and traceability of changes
        through the pending change application process.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category of the model being updated (e.g., image
                generation, text generation, etc.).
            model_name (str): The unique name of the model to update.
            record_dict (dict[str, Any]): A dictionary representing the model's fields and their new values.
            logical_user_id (str | None): An optional immutable user ID for auditing purposes, representing
                the user on whose behalf the change is being applied.
            request_id (str | None): An optional identifier for the request or job performing the update
                (e.g., a batch apply job ID or CLI invocation ID) to support traceability in logs and audits.
        """


class BackendDeleteCallable(Protocol):
    """Protocol for backend delete operations, supporting both legacy and canonical formats."""

    def __call__(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        *,
        logical_user_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Protocol for backend delete operations, supporting both legacy and canonical formats.

        The callable should perform the necessary delete operation for the given category and model name. The callable
        must also accept optional parameters for logical user ID and request ID to support auditing and traceability
        of changes through the pending change application process.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category of the model being deleted (e.g., image
                generation, text generation, etc.).
            model_name (str): The unique name of the model to delete.
            logical_user_id (str | None): An optional immutable user ID for auditing purposes, representing
                the user on whose behalf the change is being applied.
            request_id (str | None): An optional identifier for the request or job performing the delete
                (e.g., a batch apply job ID or CLI invocation ID) to support traceability in logs and audits.
        """


class PendingChangeApplyError(RuntimeError):
    """Base class for pending change apply failures."""


@dataclass(slots=True)
class PendingChangeApplyResult:
    """Return value when applying a pending change via helper APIs."""

    record: PendingChangeRecord
    batch_split: BatchSplitInfo | None = None


@dataclass(slots=True)
class PendingChangeApplyManyResult:
    """Return value when applying multiple pending changes sequentially."""

    applied_records: list[PendingChangeRecord]
    failed_change_id: int | None = None
    failed_error: PendingChangeApplyError | None = None

    # Batch split information (populated if any apply triggered a batch split)
    batch_split_occurred: bool = False
    batch_split_original_batch_id: int | None = None
    batch_split_new_batch_id: int | None = None
    batch_split_reassigned_count: int | None = None


class PendingChangeNotFoundError(PendingChangeApplyError):
    """Raised when a requested pending change cannot be found."""


class PendingChangeStateError(PendingChangeApplyError):
    """Raised when a pending change is in an invalid state for apply."""


class PendingChangePayloadError(PendingChangeApplyError):
    """Raised when a change lacks the payload required for application."""


class PendingChangeBackendError(PendingChangeApplyError):
    """Raised when the backend fails to persist the applied change."""


def apply_pending_change(
    *,
    manager: ModelReferenceManager,
    queue_service: PendingQueueService,
    change_id: int,
    applied_by: str,
    applied_username: str,
    job_id: str | None = None,
) -> PendingChangeApplyResult:
    """Apply an approved pending change through the write-capable backend.

    Args:
        manager: Singleton manager exposing the write-capable backend.
        queue_service: Pending queue service used for persistence updates.
        change_id: Identifier of the pending change to apply.
        applied_by: Immutable Horde user id for auditing purposes.
        applied_username: Username corresponding to ``applied_by``.
        job_id: Optional identifier for the job or CLI invocation performing the apply.

    Returns:
        PendingChangeApplyResult containing the updated record (now marked as applied).

    Raises:
        PendingChangeNotFoundError: If the change cannot be located.
        PendingChangeStateError: If the change is not approved yet.
        PendingChangePayloadError: When the change operation requires a payload but none exists.
        PendingChangeBackendError: If the backend rejects or fails to persist the write.
    """
    reservation_id = job_id or f"apply-{change_id}-{uuid4().hex}"

    record = queue_service.get_change(change_id)
    if record is None:
        raise PendingChangeNotFoundError(f"Change {change_id} not found.")

    try:
        record = queue_service.reserve_for_apply(change_id=change_id, reservation_id=reservation_id)
    except ValueError as exc:
        if "does not exist" in str(exc):
            raise PendingChangeNotFoundError(f"Change {change_id} not found.") from exc
        raise PendingChangeStateError(str(exc)) from exc

    backend = manager.backend
    canonical_format = horde_model_reference_settings.canonical_format
    if canonical_format == "legacy":
        if not backend.supports_legacy_writes():
            raise PendingChangeBackendError(
                "Backend does not support legacy write operations in this deployment.",
            )
        backend_update = backend.update_model_legacy
        backend_delete = backend.delete_model_legacy
    else:
        if not backend.supports_writes():
            raise PendingChangeBackendError(
                "Backend does not support write operations in this deployment.",
            )
        backend_update = backend.update_model
        backend_delete = backend.delete_model

    try:
        _apply_change_to_backend(
            record,
            backend_update=backend_update,
            backend_delete=backend_delete,
            logical_user_id=applied_by,
            request_id=reservation_id,
        )
    except PendingChangePayloadError:
        queue_service.clear_apply_reservation(change_id=change_id, reservation_id=reservation_id)
        raise
    except Exception as exc:  # pragma: no cover - defensive log for backend errors
        logger.error("Failed to apply pending change %s: %s", change_id, exc)
        queue_service.clear_apply_reservation(change_id=change_id, reservation_id=reservation_id)
        raise PendingChangeBackendError(str(exc)) from exc

    try:
        mark_result = queue_service.mark_applied(
            change_id=change_id,
            applied_by=applied_by,
            applied_username=applied_username,
            job_id=reservation_id,
        )
    except Exception as exc:  # pragma: no cover - defensive log for store errors
        logger.error("Failed to mark pending change %s applied: %s", change_id, exc)
        queue_service.clear_apply_reservation(change_id=change_id, reservation_id=reservation_id)
        raise PendingChangeBackendError(str(exc)) from exc
    return PendingChangeApplyResult(record=mark_result.record, batch_split=mark_result.batch_split)


def validate_batch_cohesion(
    *,
    change_ids: Sequence[int],
    queue_service: PendingQueueService,
) -> None:
    """Validate that all change_ids belong to the same batch.

    Args:
        change_ids: List of change IDs to validate
        queue_service: The active pending queue service

    Raises:
        ValueError: If changes belong to different batches or have no batch_id
        PendingChangeNotFoundError: If any change_id is not found
    """
    if not change_ids:
        return

    batch_ids: set[int | None] = set()
    for change_id in change_ids:
        change = queue_service.get_change(change_id)
        if change is None:
            raise PendingChangeNotFoundError(f"Change {change_id} not found")
        batch_ids.add(change.batch_id)

    if None in batch_ids:
        raise ValueError("Cannot apply changes that have not been approved in a batch")

    if len(batch_ids) > 1:
        # All batch_ids are non-None at this point
        sorted_ids = sorted(bid for bid in batch_ids if bid is not None)
        raise ValueError(f"All changes must belong to the same batch. Found batch IDs: {sorted_ids}")


def apply_pending_changes(
    *,
    manager: ModelReferenceManager,
    queue_service: PendingQueueService,
    change_ids: Sequence[int],
    applied_by: str,
    applied_username: str,
    job_id: str | None = None,
    enforce_batch_cohesion: bool = True,
) -> PendingChangeApplyManyResult:
    """Apply multiple approved changes sequentially, stopping on first failure.

    Args:
        manager: The active model reference manager
        queue_service: The active pending queue service
        change_ids: List of change IDs to apply
        applied_by: The user ID applying the changes
        applied_username: The username applying the changes
        job_id: An optional job identifier for tracking the apply job
        enforce_batch_cohesion: If True, all changes must belong to the same batch

    Returns:
        PendingChangeApplyManyResult: Summary of the apply operation, including
        batch split information if a partial apply triggered reassignment.

    Raises:
        ValueError: If enforce_batch_cohesion=True and changes belong to different batches
        PendingChangeNotFoundError: If any change_id is not found
    """
    if not change_ids:
        raise ValueError("change_ids must contain at least one id")

    if enforce_batch_cohesion:
        validate_batch_cohesion(change_ids=change_ids, queue_service=queue_service)

    applied_records: list[PendingChangeRecord] = []
    last_batch_split: BatchSplitInfo | None = None

    for change_id in change_ids:
        try:
            result = apply_pending_change(
                manager=manager,
                queue_service=queue_service,
                change_id=change_id,
                applied_by=applied_by,
                applied_username=applied_username,
                job_id=job_id,
            )
        except PendingChangeApplyError as exc:
            # On failure, include any batch split info from previous applies
            return PendingChangeApplyManyResult(
                applied_records=applied_records,
                failed_change_id=change_id,
                failed_error=exc,
                batch_split_occurred=last_batch_split is not None,
                batch_split_original_batch_id=last_batch_split.original_batch_id if last_batch_split else None,
                batch_split_new_batch_id=last_batch_split.new_batch_id if last_batch_split else None,
                batch_split_reassigned_count=len(last_batch_split.reassigned_change_ids) if last_batch_split else None,
            )

        applied_records.append(result.record)
        # Track the last batch split (typically only the last apply in a batch triggers it)
        if result.batch_split is not None:
            last_batch_split = result.batch_split

    return PendingChangeApplyManyResult(
        applied_records=applied_records,
        batch_split_occurred=last_batch_split is not None,
        batch_split_original_batch_id=last_batch_split.original_batch_id if last_batch_split else None,
        batch_split_new_batch_id=last_batch_split.new_batch_id if last_batch_split else None,
        batch_split_reassigned_count=len(last_batch_split.reassigned_change_ids) if last_batch_split else None,
    )


def _apply_change_to_backend(
    record: PendingChangeRecord,
    *,
    backend_update: BackendUpdateCallable,
    backend_delete: BackendDeleteCallable,
    logical_user_id: str,
    request_id: str | None,
) -> None:
    """Execute the backend mutation for the given pending change."""
    category = record.category
    model_name = record.model_name

    if record.operation in {AuditOperation.CREATE, AuditOperation.UPDATE}:
        payload = record.payload
        if payload is None:
            raise PendingChangePayloadError(
                f"Change {record.change_id} ({record.operation}) is missing payload data.",
            )
        backend_update(
            category,
            model_name,
            payload,
            logical_user_id=logical_user_id,
            request_id=request_id,
        )
        return

    if record.operation is AuditOperation.DELETE:
        backend_delete(
            category,
            model_name,
            logical_user_id=logical_user_id,
            request_id=request_id,
        )
        return

    raise PendingChangeBackendError(f"Unsupported operation {record.operation} for change {record.change_id}.")
