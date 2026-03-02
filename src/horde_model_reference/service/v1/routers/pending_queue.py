"""v1 pending queue router wiring."""

from horde_model_reference.service.pending_queue.router import build_pending_queue_router
from horde_model_reference.service.shared import assert_pending_queue_write_enabled

router = build_pending_queue_router(
    tags=("v1", "pending_queue"), assert_write_enabled=assert_pending_queue_write_enabled
)
