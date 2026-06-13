"""Materialize pending queue changes into ready-to-serve model records.

"Beta" models are surfaced to REPLICA clients as the ``CREATE``/``UPDATE`` payloads of
``PENDING``/``APPROVED`` queue entries. This module turns those raw payloads into the
same v2 record shape the canonical category endpoint returns, so a client can overlay
them without caring that they came from the queue.

When the deployment's canonical format is ``legacy`` the stored payloads are legacy
records (that is what the v1 write path enqueues), so they are round-tripped through the
canonical legacy->v2 converter — the same code path :class:`GitHubBackend` uses — rather
than a bespoke per-record conversion, so beta records never diverge from canonical ones.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from horde_model_reference import CanonicalFormat, horde_model_reference_paths
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_legacy_database_by_category
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue.models import PendingChangeRecord

_BETA_OPERATIONS = frozenset({AuditOperation.CREATE, AuditOperation.UPDATE})
"""Queue operations that contribute a usable beta model. ``DELETE`` is intentionally
excluded: a pending deletion must never remove a model from a worker's reference."""


def _change_sort_key(change: PendingChangeRecord) -> tuple[int, int]:
    """Order changes so the most recent one for a model name wins."""
    return (change.updated_at, change.change_id)


def select_beta_changes(
    changes: Iterable[PendingChangeRecord],
) -> dict[str, PendingChangeRecord]:
    """Return the winning ``CREATE``/``UPDATE`` change per model name.

    ``DELETE`` operations and payload-less entries are skipped. When several queued
    changes target the same model name, the most recently updated one wins.
    """
    latest: dict[str, PendingChangeRecord] = {}
    for change in changes:
        if change.operation not in _BETA_OPERATIONS or change.payload is None:
            continue
        current = latest.get(change.model_name)
        if current is None or _change_sort_key(change) >= _change_sort_key(current):
            latest[change.model_name] = change
    return latest


def materialize_pending_records(
    category: MODEL_REFERENCE_CATEGORY,
    changes: Iterable[PendingChangeRecord],
    *,
    domain: CanonicalFormat,
) -> dict[str, dict[str, Any]]:
    """Return ``{model_name: v2_record_dict}`` for the category's beta changes.

    Args:
        category: The category the changes belong to.
        changes: Pending queue changes (any statuses; callers pre-filter by status).
        domain: The deployment's canonical format. ``v2`` payloads pass through; ``legacy``
            payloads are converted to v2.

    Returns:
        A mapping of model name to v2 record dict, ready to serve like the canonical
        category payload. Empty when nothing is materializable.

    """
    selected = select_beta_changes(changes)
    if not selected:
        return {}

    payloads: dict[str, dict[str, Any]] = {
        name: change.payload for name, change in selected.items() if change.payload is not None
    }

    if domain == CanonicalFormat.v2:
        return payloads

    return _convert_legacy_payloads_to_v2(category, payloads)


def _convert_legacy_payloads_to_v2(
    category: MODEL_REFERENCE_CATEGORY,
    legacy_payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert legacy-domain payloads to v2 via the canonical file-based converter.

    The converter reads a legacy reference file and writes the converted v2 file, so the
    payloads are written to a throwaway temp tree laid out exactly like a real cache
    (``{base}/legacy/...`` in, ``{base}/...`` out) and the converted file is read back.
    """
    with tempfile.TemporaryDirectory(prefix="hmr-pending-") as tmp:
        tmp_base = Path(tmp)
        legacy_file = horde_model_reference_paths.get_legacy_model_reference_file_path(category, base_path=tmp_base)
        legacy_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.write_text(json.dumps(legacy_payloads), encoding="utf-8")

        if not convert_legacy_database_by_category(category, tmp_base, tmp_base):
            logger.warning(f"Legacy->v2 conversion of pending {category.value} models failed; serving none.")
            return {}

        converted_file = horde_model_reference_paths.get_model_reference_file_path(category, base_path=tmp_base)
        if not converted_file.exists():
            return {}

        converted: dict[str, dict[str, Any]] = json.loads(converted_file.read_text(encoding="utf-8"))
        return converted
