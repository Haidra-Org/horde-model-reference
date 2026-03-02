"""Pending change queue coordination for model reference edits."""

from .apply import (
    PendingChangeApplyError,
    PendingChangeApplyManyResult,
    PendingChangeApplyResult,
    PendingChangeBackendError,
    PendingChangeNotFoundError,
    PendingChangePayloadError,
    PendingChangeStateError,
    apply_pending_change,
    apply_pending_changes,
)
from .diff_utils import (
    CRITICAL_FIELDS_BY_CATEGORY,
    DOWNLOAD_URL_FIELDS,
    FieldChangeType,
    FieldDiff,
    NetChangeType,
    categorize_field_diffs,
    compute_field_diffs,
    has_critical_changes,
)
from .models import (
    PendingBatchResult,
    PendingChangeDiff,
    PendingChangeDiffPage,
    PendingChangeRecord,
    PendingChangeStatus,
    PendingQueueFilter,
    PendingQueuePage,
)
from .service import PendingQueueService
from .store import PendingQueueStore

__all__ = [
    "CRITICAL_FIELDS_BY_CATEGORY",
    "DOWNLOAD_URL_FIELDS",
    "FieldChangeType",
    "FieldDiff",
    "NetChangeType",
    "PendingBatchResult",
    "PendingChangeApplyError",
    "PendingChangeApplyManyResult",
    "PendingChangeApplyResult",
    "PendingChangeBackendError",
    "PendingChangeDiff",
    "PendingChangeDiffPage",
    "PendingChangeNotFoundError",
    "PendingChangePayloadError",
    "PendingChangeRecord",
    "PendingChangeStateError",
    "PendingChangeStatus",
    "PendingQueueFilter",
    "PendingQueuePage",
    "PendingQueueService",
    "PendingQueueStore",
    "apply_pending_change",
    "apply_pending_changes",
    "categorize_field_diffs",
    "compute_field_diffs",
    "has_critical_changes",
]
