"""Read-only endpoints exposing pending queue audit data."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from horde_model_reference import (
    CanonicalFormat,
    ModelReferenceManager,
    horde_model_reference_paths,
    horde_model_reference_settings,
)
from horde_model_reference.audit import AuditDomain
from horde_model_reference.pending_queue.audit_view import (
    BatchNetChangeResponse,
    PendingQueueAuditBatchDetail,
    PendingQueueAuditBatchPage,
    PendingQueueAuditCurrentResponse,
    compute_batch_net_changes,
    load_pending_queue_audit_dataset,
)
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    ErrorResponse,
    authenticate_queue_approver,
    get_model_reference_manager,
    header_auth_scheme,
)

DomainOverride = Annotated[AuditDomain | None, Query(description="Optional audit domain override")]
CursorQuery = Annotated[int | None, Query(ge=1, description="Return items older than this batch id")]
LimitQuery = Annotated[int, Query(ge=1, le=50, description="Maximum number of batches to return")]


async def _assert_audit_access(apikey: str) -> None:
    await authenticate_queue_approver(apikey)


def build_pending_queue_audit_router(*, tags: Sequence[str]) -> APIRouter:
    """Construct a router serving pending queue audit data."""
    router = APIRouter(prefix="/pending_queue/audit", tags=list(tags))

    @router.get(
        "/current",
        response_model=PendingQueueAuditCurrentResponse,
        summary="List currently pending (unapproved) changes",
        responses={401: {"description": "Invalid API key", "model": ErrorResponse}},
    )
    async def get_current_pending_changes(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        domain_override: DomainOverride = None,
    ) -> PendingQueueAuditCurrentResponse:
        _ensure_audit_enabled()
        await _assert_audit_access(apikey)
        require_pending_queue_service(manager)
        domain = _resolve_domain(domain_override)
        dataset = load_pending_queue_audit_dataset(
            root_path=horde_model_reference_paths.audit_path,
            domain=domain,
        )
        pending = dataset.pending_changes()
        return PendingQueueAuditCurrentResponse(
            domain=domain,
            pending_changes=pending,
            total_pending=len(pending),
            generated_at=int(time.time()),
        )

    @router.get(
        "/batches",
        response_model=PendingQueueAuditBatchPage,
        summary="List historical pending queue batches",
        responses={401: {"description": "Invalid API key", "model": ErrorResponse}},
    )
    async def list_pending_queue_batches(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
        cursor: CursorQuery = None,
        limit: LimitQuery = 10,
        domain_override: DomainOverride = None,
    ) -> PendingQueueAuditBatchPage:
        _ensure_audit_enabled()
        await _assert_audit_access(apikey)
        require_pending_queue_service(manager)
        domain = _resolve_domain(domain_override)
        dataset = load_pending_queue_audit_dataset(
            root_path=horde_model_reference_paths.audit_path,
            domain=domain,
        )
        summaries, next_cursor = dataset.batches_page(cursor=cursor, limit=limit)
        return PendingQueueAuditBatchPage(domain=domain, batches=summaries, next_cursor=next_cursor)

    @router.get(
        "/batches/{batch_id}",
        response_model=PendingQueueAuditBatchDetail,
        summary="Get details for a specific batch",
        responses={
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Batch not found", "model": ErrorResponse},
        },
    )
    async def get_pending_queue_batch_detail(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        batch_id: int,
        apikey: Annotated[str, Depends(header_auth_scheme)],
        domain_override: DomainOverride = None,
    ) -> PendingQueueAuditBatchDetail:
        _ensure_audit_enabled()
        await _assert_audit_access(apikey)
        require_pending_queue_service(manager)
        domain = _resolve_domain(domain_override)
        dataset = load_pending_queue_audit_dataset(
            root_path=horde_model_reference_paths.audit_path,
            domain=domain,
        )
        detail = dataset.batch_detail(batch_id)
        if detail is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
        return detail

    @router.get(
        "/batches/{batch_id}/net_changes",
        response_model=BatchNetChangeResponse,
        summary="Get net changes for a specific batch",
        responses={
            401: {"description": "Invalid API key", "model": ErrorResponse},
            404: {"description": "Batch not found", "model": ErrorResponse},
        },
    )
    async def get_batch_net_changes(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        batch_id: int,
        apikey: Annotated[str, Depends(header_auth_scheme)],
        domain_override: DomainOverride = None,
    ) -> BatchNetChangeResponse:
        """Compute the net effect of all changes in a batch.

        Analyzes all operations (add, update, delete) applied in the batch and
        computes the net change for each affected model. Models that are deleted
        and re-added with identical content show net_operation=UNCHANGED.

        Results are cached for 5 minutes to match existing audit caching patterns.
        """
        _ensure_audit_enabled()
        await _assert_audit_access(apikey)
        require_pending_queue_service(manager)
        domain = _resolve_domain(domain_override)

        # Use cached computation with 5-minute TTL
        result = _get_batch_net_changes_cached(
            root_path_str=str(horde_model_reference_paths.audit_path),
            domain=domain,
            batch_id=batch_id,
        )
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found")
        return result

    return router


_NET_CHANGES_CACHE: dict[tuple[str, AuditDomain, int], tuple[float, BatchNetChangeResponse | None]] = {}
_NET_CHANGES_CACHE_LOCK = threading.Lock()
_NET_CHANGES_TTL_SECONDS = 300


def _get_batch_net_changes_cached(
    root_path_str: str,
    domain: AuditDomain,
    batch_id: int,
) -> BatchNetChangeResponse | None:
    """Batch net change computation with 5-minute TTL cache.

    Args:
        root_path_str (str): The root path for the audit dataset.
        domain (AuditDomain): The audit domain.
        batch_id (int): The batch ID.

    Returns:
        BatchNetChangeResponse | None: The net changes for the batch, or None if not found.

    """
    from pathlib import Path

    key = (root_path_str, domain, batch_id)
    now = time.monotonic()

    with _NET_CHANGES_CACHE_LOCK:
        entry = _NET_CHANGES_CACHE.get(key)
        if entry is not None and (now - entry[0]) < _NET_CHANGES_TTL_SECONDS:
            return entry[1]

    result = compute_batch_net_changes(
        root_path=Path(root_path_str),
        domain=domain,
        batch_id=batch_id,
    )

    with _NET_CHANGES_CACHE_LOCK:
        _NET_CHANGES_CACHE[key] = (now, result)

    return result


def _resolve_domain(domain_override: AuditDomain | None) -> AuditDomain:
    if domain_override is not None:
        return domain_override
    canonical = horde_model_reference_settings.canonical_format
    return AuditDomain.LEGACY if canonical == CanonicalFormat.LEGACY else AuditDomain.V2


def _ensure_audit_enabled() -> None:
    if not horde_model_reference_settings.audit.enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audit trail is disabled on this deployment.",
        )
