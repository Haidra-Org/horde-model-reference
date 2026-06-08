"""Shared pending queue router builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from horde_model_reference import ModelReferenceManager
from horde_model_reference.diff_service import PendingChangeDiffService
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue import (
    PendingBatchResult,
    PendingChangeApplyError,
    PendingChangeBackendError,
    PendingChangeDiff,
    PendingChangeDiffPage,
    PendingChangeNotFoundError,
    PendingChangePayloadError,
    PendingChangeRecord,
    PendingChangeStateError,
    PendingQueueFilter,
    PendingQueuePage,
    apply_pending_change,
    apply_pending_changes,
)
from horde_model_reference.pending_queue.models import PendingChangeStatus
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    ErrorResponse,
    authenticate_queue_approver,
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
)

WriteGuard = Callable[[ModelReferenceManager], None]

StatusesQuery = Annotated[list[PendingChangeStatus] | None, Query()]
CategoriesQuery = Annotated[list[MODEL_REFERENCE_CATEGORY] | None, Query()]
BatchIdQuery = Annotated[int | None, Query(ge=1)]
ModelNameQuery = Annotated[str | None, Query(min_length=1, max_length=200)]
OffsetQuery = Annotated[int, Query(ge=0)]
LimitQuery = Annotated[int, Query(ge=1, le=500)]
RequestedByQuery = Annotated[list[str] | None, Query()]


class PendingBatchRequest(BaseModel):
    """Request payload used to approve/reject queued changes."""

    batch_title: str = Field(min_length=1, max_length=120)
    approved_ids: list[int] | None = None
    rejected_ids: list[int] | None = None
    reject_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _validate_payload(self) -> PendingBatchRequest:
        approved = {change_id for change_id in (self.approved_ids or []) if change_id > 0}
        rejected = {change_id for change_id in (self.rejected_ids or []) if change_id > 0}

        if not approved and not rejected:
            raise ValueError("Provide at least one approved or rejected change id.")

        overlap = approved & rejected
        if overlap:
            raise ValueError(f"Change ids cannot be both approved and rejected: {sorted(overlap)}")

        if rejected and (self.reject_reason is None or not self.reject_reason.strip()):
            raise ValueError("reject_reason is required when rejecting changes.")

        self.approved_ids = sorted(approved)
        self.rejected_ids = sorted(rejected)
        self.batch_title = self.batch_title.strip()
        if not self.batch_title:
            raise ValueError("batch_title must not be blank.")
        if self.reject_reason:
            self.reject_reason = self.reject_reason.strip()
        return self


class ApplyPendingChangeRequest(BaseModel):
    """Request payload for applying an approved change."""

    job_id: str | None = Field(default=None, max_length=120)

    @field_validator("job_id")
    @classmethod
    def _normalize_job_id(cls, job_id: str | None) -> str | None:
        if job_id is None:
            return None
        normalized = job_id.strip()
        return normalized or None


class ApplyPendingChangesRequest(BaseModel):
    """Request payload for applying multiple approved changes."""

    change_ids: list[int] = Field(min_length=1)
    job_id: str | None = Field(default=None, max_length=120)
    allow_mixed_batch: bool = Field(
        default=False,
        description="If False, all changes must belong to the same batch",
    )

    @field_validator("job_id")
    @classmethod
    def _normalize_job_id(cls, job_id: str | None) -> str | None:
        if job_id is None:
            return None
        normalized = job_id.strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_change_ids(self) -> ApplyPendingChangesRequest:
        if not self.change_ids:
            raise ValueError("change_ids must include at least one id")

        deduped: list[int] = []
        seen: set[int] = set()
        for change_id in self.change_ids:
            if change_id <= 0:
                raise ValueError("change_ids must be positive integers")
            if change_id not in seen:
                deduped.append(change_id)
                seen.add(change_id)

        self.change_ids = deduped
        return self


class ApplyPendingChangesResponse(BaseModel):
    """Response payload summarizing a bulk apply attempt.

    Batch Split Semantics:
    When a partial apply occurs (some changes in a batch are applied while others remain),
    the remaining APPROVED changes are automatically reassigned to a new batch ID. This
    information is provided in the batch_split_* fields to help clients update their UI.
    """

    applied: list[PendingChangeRecord] = Field(default_factory=list)
    failed_change_id: int | None = None
    failed_error: str | None = None
    failed_error_type: str | None = None

    # Batch split information (populated when partial apply triggers reassignment)
    batch_split_occurred: bool = Field(
        default=False,
        description="True if applying changes triggered a batch split (remaining changes reassigned)",
    )
    batch_split_original_batch_id: int | None = Field(
        default=None,
        description="The original batch ID that was partially applied",
    )
    batch_split_new_batch_id: int | None = Field(
        default=None,
        description="The new batch ID assigned to remaining unapplied changes",
    )
    batch_split_reassigned_count: int | None = Field(
        default=None,
        description="Number of changes that were reassigned to the new batch",
    )


class ApplySingleChangeResponse(BaseModel):
    """Response payload for applying a single pending change.

    Includes the updated record and batch split information when applicable.
    """

    record: PendingChangeRecord = Field(
        description="The applied pending change record with updated status",
    )
    batch_split_occurred: bool = Field(
        default=False,
        description="True if applying this change triggered a batch split",
    )
    batch_split_original_batch_id: int | None = Field(
        default=None,
        description="The original batch ID that was partially applied",
    )
    batch_split_new_batch_id: int | None = Field(
        default=None,
        description="The new batch ID assigned to remaining unapplied changes",
    )
    batch_split_reassigned_count: int | None = Field(
        default=None,
        description="Number of changes that were reassigned to the new batch",
    )


class PurgePendingChangesRequest(BaseModel):
    """Request payload to purge pending changes matching a filter."""

    statuses: list[PendingChangeStatus] | None = None
    categories: list[MODEL_REFERENCE_CATEGORY] | None = None
    model_name: str | None = Field(default=None, max_length=200)
    requested_by: list[str] | None = None
    purge_all: bool = False
    dry_run: bool = False

    @model_validator(mode="after")
    def _validate_payload(self) -> PurgePendingChangesRequest:
        statuses = set(self.statuses or [])
        categories = set(self.categories or [])
        requested_by = {user_id.strip() for user_id in self.requested_by or [] if user_id and user_id.strip()}
        model_name = self.model_name.strip() if self.model_name else None

        has_filters = bool(statuses or categories or requested_by or model_name)
        if not self.purge_all and not has_filters:
            raise ValueError("Provide at least one filter or set purge_all=true to clear the entire queue.")

        self.statuses = sorted(statuses)
        self.categories = sorted(categories)
        self.requested_by = sorted(requested_by) if requested_by else None
        self.model_name = model_name
        return self


class PurgePendingChangesResponse(BaseModel):
    """Response payload for a purge operation."""

    removed_count: int = Field(ge=0)
    removed_change_ids: list[int] = Field(default_factory=list)
    dry_run: bool = False


def _status_for_apply_error(error: PendingChangeApplyError) -> int:
    if isinstance(error, PendingChangeNotFoundError):
        return status.HTTP_404_NOT_FOUND
    if isinstance(error, (PendingChangeStateError, PendingChangePayloadError)):
        return status.HTTP_400_BAD_REQUEST
    if isinstance(error, PendingChangeBackendError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_400_BAD_REQUEST


async def _assert_approver(apikey: str) -> tuple[str, str]:
    approver = await authenticate_queue_approver(apikey)
    return approver.user_id, approver.username


def build_pending_queue_router(*, tags: Sequence[str], assert_write_enabled: WriteGuard) -> APIRouter:
    """Create a pending queue router whose guards can differ per API version."""
    router = APIRouter(prefix="/pending_queue", tags=list(tags))

    @router.get(
        "/changes",
        response_model=PendingQueuePage,
        summary="List pending queue entries",
        responses={
            200: {"description": "Filtered queue entries"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            503: {"description": "Pending queue disabled", "model": ErrorResponse},
        },
    )
    async def list_pending_changes(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        statuses: StatusesQuery = None,
        categories: CategoriesQuery = None,
        batch_id: BatchIdQuery = None,
        model_name: ModelNameQuery = None,
        requested_by: RequestedByQuery = None,
        offset: OffsetQuery = 0,
        limit: LimitQuery = 50,
    ) -> PendingQueuePage:
        """Return a filtered, paginated list of pending queue entries."""
        await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        normalized_name = model_name.strip() if model_name else None
        normalized_requestors = {value.strip() for value in requested_by or [] if value and value.strip()}
        queue_filter = PendingQueueFilter(
            statuses=set(statuses) if statuses else None,
            categories=set(categories) if categories else None,
            batch_id=batch_id,
            model_name=normalized_name,
            requested_by=normalized_requestors or None,
        )

        return queue_service.list_changes(queue_filter=queue_filter, offset=offset, limit=limit)

    @router.get(
        "/my_changes",
        response_model=PendingQueuePage,
        summary="List your own submitted pending changes",
        responses={
            200: {"description": "Your submitted queue entries"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            503: {"description": "Pending queue disabled", "model": ErrorResponse},
        },
    )
    async def list_my_pending_changes(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        statuses: StatusesQuery = None,
        categories: CategoriesQuery = None,
        offset: OffsetQuery = 0,
        limit: LimitQuery = 50,
    ) -> PendingQueuePage:
        """Return the caller's own queued changes so a requestor can track a proposal's fate.

        Unlike ``/changes`` this requires only the *requestor* role; visibility is hard-scoped
        to the calling key's user id and cannot be widened to other users' changes.
        """
        requestor = await authenticate_queue_requestor(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        queue_filter = PendingQueueFilter(
            statuses=set(statuses) if statuses else None,
            categories=set(categories) if categories else None,
            requested_by={requestor.user_id},
        )

        return queue_service.list_changes(queue_filter=queue_filter, offset=offset, limit=limit)

    @router.post(
        "/purge",
        response_model=PurgePendingChangesResponse,
        summary="Purge pending changes matching filters",
        responses={
            200: {"description": "Filtered changes removed"},
            400: {"description": "Invalid purge request", "model": ErrorResponse},
            401: {"description": "Invalid API key", "model": ErrorResponse},
        },
    )
    async def purge_pending_changes(
        request: PurgePendingChangesRequest,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> PurgePendingChangesResponse:
        """Delete queued changes in bulk, optionally as a dry run."""
        approver_id, approver_username = await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        queue_filter = PendingQueueFilter(
            statuses=set(request.statuses) if request.statuses else None,
            categories=set(request.categories) if request.categories else None,
            model_name=request.model_name,
            requested_by=set(request.requested_by) if request.requested_by else None,
        )
        has_filter = bool(
            queue_filter.statuses or queue_filter.categories or queue_filter.model_name or queue_filter.requested_by
        )
        active_filter = queue_filter if has_filter else None

        if request.dry_run:
            page = queue_service.list_changes(queue_filter=active_filter, offset=0, limit=None)
            return PurgePendingChangesResponse(
                removed_count=page.total,
                removed_change_ids=[record.change_id for record in page.items],
                dry_run=True,
            )

        removed = queue_service.purge_changes(
            queue_filter=None if request.purge_all and not has_filter else active_filter,
            purged_by=approver_id,
            purged_username=approver_username,
        )

        return PurgePendingChangesResponse(
            removed_count=len(removed),
            removed_change_ids=[record.change_id for record in removed],
            dry_run=False,
        )

    @router.get(
        "/changes/{change_id}",
        response_model=PendingChangeRecord,
        summary="Get a single pending change",
        responses={
            200: {"description": "Pending change details"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Change not found", "model": ErrorResponse},
        },
    )
    async def read_pending_change(
        change_id: int,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> PendingChangeRecord:
        """Return details for a single pending change."""
        await _assert_approver(apikey)
        queue_service = require_pending_queue_service(manager)
        record = queue_service.get_change(change_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")
        return record

    @router.get(
        "/changes/{change_id}/diff",
        response_model=PendingChangeDiff,
        summary="Get diff for a pending change",
        responses={
            200: {"description": "Diff between current and proposed state"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Change not found", "model": ErrorResponse},
        },
    )
    async def get_pending_change_diff(
        change_id: int,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> PendingChangeDiff:
        """Return a detailed diff for a pending change.

        Compares the pending change payload against the current model state
        in the backend to show exactly what would change if applied.

        For UPDATE operations, returns field-level diffs showing added,
        removed, and modified fields. For CREATE/DELETE operations, shows
        the full proposed/current state respectively.
        """
        await _assert_approver(apikey)
        queue_service = require_pending_queue_service(manager)
        diff_service = PendingChangeDiffService(manager=manager, queue_service=queue_service)

        diff = diff_service.compute_change_diff(change_id)
        if diff is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change not found")
        return diff

    @router.get(
        "/changes/diff",
        response_model=PendingChangeDiffPage,
        summary="Get diffs for multiple pending changes",
        responses={
            200: {"description": "Diffs for requested changes"},
            400: {"description": "Invalid request", "model": ErrorResponse},
            401: {"description": "Invalid API key", "model": ErrorResponse},
        },
    )
    async def get_pending_changes_diffs(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        change_ids: Annotated[list[int], Query(min_length=1, max_length=100)],
    ) -> PendingChangeDiffPage:
        """Return diffs for multiple pending changes in bulk.

        Accepts a list of change IDs and returns diffs for each. Changes
        that cannot be found or diffed are reported in the errors array.
        """
        await _assert_approver(apikey)
        queue_service = require_pending_queue_service(manager)
        diff_service = PendingChangeDiffService(manager=manager, queue_service=queue_service)

        return diff_service.compute_bulk_diffs(change_ids)

    @router.post(
        "/batches",
        response_model=PendingBatchResult,
        summary="Approve or reject queued changes",
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "Batch processed"},
            400: {"description": "Invalid batch request", "model": ErrorResponse},
            401: {"description": "Invalid API key", "model": ErrorResponse},
        },
    )
    async def process_pending_batch(
        request: PendingBatchRequest,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> PendingBatchResult:
        """Approve and/or reject a set of pending changes in one batch."""
        approver_id, approver_username = await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        try:
            return queue_service.process_batch(
                approver_id=approver_id,
                approver_username=approver_username,
                batch_title=request.batch_title,
                approved_ids=request.approved_ids,
                rejected_ids=request.rejected_ids,
                reject_reason=request.reject_reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @router.post(
        "/changes/{change_id}/apply",
        response_model=ApplySingleChangeResponse,
        summary="Apply an approved change to the backend",
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "Change applied"},
            400: {"description": "Change not ready for apply", "model": ErrorResponse},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Change not found", "model": ErrorResponse},
            503: {"description": "Writes not supported", "model": ErrorResponse},
        },
    )
    async def apply_pending_change_endpoint(
        change_id: int,
        request: ApplyPendingChangeRequest,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> JSONResponse:
        """Apply an approved pending change and mark it as applied."""
        approver_id, approver_username = await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        try:
            result = apply_pending_change(
                manager=manager,
                queue_service=queue_service,
                change_id=change_id,
                applied_by=approver_id,
                applied_username=approver_username,
                job_id=request.job_id,
            )
        except PendingChangeNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except PendingChangeStateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except PendingChangePayloadError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except PendingChangeBackendError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except PendingChangeApplyError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        # Build response with batch split information
        response_data: dict[str, object] = {
            "record": result.record.model_dump(mode="json", exclude_none=True),
            "batch_split_occurred": result.batch_split is not None,
        }
        if result.batch_split is not None:
            response_data["batch_split_original_batch_id"] = result.batch_split.original_batch_id
            response_data["batch_split_new_batch_id"] = result.batch_split.new_batch_id
            response_data["batch_split_reassigned_count"] = len(result.batch_split.reassigned_change_ids)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data,
        )

    @router.post(
        "/apply",
        response_model=ApplyPendingChangesResponse,
        summary="Apply multiple approved changes",
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "All requested changes applied"},
            400: {"description": "Invalid request or change state"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "One of the change ids was not found"},
            503: {"description": "Writes not supported or backend failure"},
        },
    )
    async def apply_pending_changes_endpoint(
        request: ApplyPendingChangesRequest,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> JSONResponse:
        """Apply a batch of approved pending changes sequentially."""
        approver_id, approver_username = await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        try:
            result = apply_pending_changes(
                manager=manager,
                queue_service=queue_service,
                change_ids=request.change_ids,
                applied_by=approver_id,
                applied_username=approver_username,
                job_id=request.job_id,
                enforce_batch_cohesion=not request.allow_mixed_batch,
            )
        except PendingChangeNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        response_payload = ApplyPendingChangesResponse(
            applied=result.applied_records,
            failed_change_id=result.failed_change_id,
            failed_error=str(result.failed_error) if result.failed_error else None,
            failed_error_type=type(result.failed_error).__name__ if result.failed_error else None,
            batch_split_occurred=result.batch_split_occurred,
            batch_split_original_batch_id=result.batch_split_original_batch_id,
            batch_split_new_batch_id=result.batch_split_new_batch_id,
            batch_split_reassigned_count=result.batch_split_reassigned_count,
        )

        status_code = status.HTTP_200_OK
        if result.failed_error is not None:
            status_code = _status_for_apply_error(result.failed_error)

        return JSONResponse(
            status_code=status_code,
            content=response_payload.model_dump(mode="json", exclude_none=True),
        )

    @router.post(
        "/apply_batch/{batch_id}",
        response_model=ApplyPendingChangesResponse,
        summary="Apply all approved changes in a specific batch",
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "All approved changes in batch applied"},
            400: {"description": "Invalid batch or change state"},
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Batch not found or has no approved changes"},
            503: {"description": "Writes not supported or backend failure"},
        },
    )
    async def apply_batch_endpoint(
        batch_id: int,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        job_id: Annotated[str | None, Query(max_length=120)] = None,
    ) -> JSONResponse:
        """Apply all APPROVED changes in a batch, skipping already-applied changes."""
        approver_id, approver_username = await _assert_approver(apikey)
        assert_write_enabled(manager)
        queue_service = require_pending_queue_service(manager)

        # Get all changes in the batch
        batch_filter = PendingQueueFilter(batch_id=batch_id)
        all_changes = queue_service.list_changes(queue_filter=batch_filter, offset=0, limit=None)

        if all_changes.total == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No changes found for batch {batch_id}",
            )

        # Filter to only APPROVED changes (skip APPLIED, REJECTED, PENDING)
        approved_changes = [change for change in all_changes.items if change.status == PendingChangeStatus.APPROVED]

        if not approved_changes:
            # Check if batch exists but all changes are already applied
            applied_count = sum(1 for c in all_changes.items if c.status == PendingChangeStatus.APPLIED)
            if applied_count == all_changes.total:
                # All changes already applied - return success with empty list
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content=ApplyPendingChangesResponse(applied=[]).model_dump(mode="json", exclude_none=True),
                )
            # Batch exists but has no approved changes (all pending/rejected)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No approved changes found in batch {batch_id}",
            )

        change_ids = [change.change_id for change in approved_changes]

        try:
            result = apply_pending_changes(
                manager=manager,
                queue_service=queue_service,
                change_ids=change_ids,
                applied_by=approver_id,
                applied_username=approver_username,
                job_id=job_id,
                enforce_batch_cohesion=True,  # Always enforce for batch endpoint
            )
        except PendingChangeNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        response_payload = ApplyPendingChangesResponse(
            applied=result.applied_records,
            failed_change_id=result.failed_change_id,
            failed_error=str(result.failed_error) if result.failed_error else None,
            failed_error_type=type(result.failed_error).__name__ if result.failed_error else None,
            batch_split_occurred=result.batch_split_occurred,
            batch_split_original_batch_id=result.batch_split_original_batch_id,
            batch_split_new_batch_id=result.batch_split_new_batch_id,
            batch_split_reassigned_count=result.batch_split_reassigned_count,
        )

        status_code = status.HTTP_200_OK
        if result.failed_error is not None:
            status_code = _status_for_apply_error(result.failed_error)

        return JSONResponse(
            status_code=status_code,
            content=response_payload.model_dump(mode="json", exclude_none=True),
        )

    return router
