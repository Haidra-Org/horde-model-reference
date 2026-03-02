"""V2 pending queue audit router."""

from horde_model_reference.service.pending_queue.audit_router import build_pending_queue_audit_router

router = build_pending_queue_audit_router(tags=("v2", "pending_queue", "audit"))
