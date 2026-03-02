"""Dependencies shared across pending queue routes."""

from fastapi import HTTPException, status

from horde_model_reference import ModelReferenceManager
from horde_model_reference.pending_queue import PendingQueueService


def require_pending_queue_service(manager: ModelReferenceManager) -> PendingQueueService:
    """Return the configured pending queue service or raise when disabled."""
    queue_service = manager.pending_queue_service
    if queue_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pending queue is disabled or unsupported on this deployment.",
        )
    return queue_service
