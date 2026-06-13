"""Tests for materializing pending-queue changes into beta model records."""

from __future__ import annotations

from typing import Any

from horde_model_reference import MODEL_REFERENCE_CATEGORY, CanonicalFormat
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue.materialize import (
    materialize_pending_records,
    select_beta_changes,
)
from horde_model_reference.pending_queue.models import PendingChangeRecord

_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation


def _change(
    model_name: str,
    operation: AuditOperation,
    payload: dict[str, Any] | None,
    *,
    change_id: int = 1,
    updated_at: int = 1000,
) -> PendingChangeRecord:
    return PendingChangeRecord(
        change_id=change_id,
        category=_CATEGORY,
        model_name=model_name,
        operation=operation,
        payload=payload,
        requested_by="user-1",
        requested_username="tester#user-1",
        updated_at=updated_at,
    )


def test_select_beta_skips_delete_and_payloadless() -> None:
    """DELETE operations and payload-less entries never become beta models."""
    changes = [
        _change("keep", AuditOperation.CREATE, {"name": "keep"}, change_id=1),
        _change("gone", AuditOperation.DELETE, {"name": "gone"}, change_id=2),
        _change("empty", AuditOperation.UPDATE, None, change_id=3),
    ]
    selected = select_beta_changes(changes)
    assert set(selected) == {"keep"}


def test_select_beta_latest_change_wins() -> None:
    """When multiple changes target one model name, the most recently updated wins."""
    changes = [
        _change("m", AuditOperation.CREATE, {"name": "m", "v": "old"}, change_id=1, updated_at=100),
        _change("m", AuditOperation.UPDATE, {"name": "m", "v": "new"}, change_id=2, updated_at=200),
    ]
    selected = select_beta_changes(changes)
    assert selected["m"].payload == {"name": "m", "v": "new"}


def test_materialize_v2_passes_payload_through() -> None:
    """With a v2 canonical domain, payloads are returned as-is keyed by model name."""
    payload = {"name": "beta", "baseline": "stable_diffusion_1", "nsfw": False}
    records = materialize_pending_records(
        _CATEGORY,
        [_change("beta", AuditOperation.CREATE, payload)],
        domain=CanonicalFormat.v2,
    )
    assert records == {"beta": payload}


def test_materialize_empty_when_nothing_selected() -> None:
    """A queue with only DELETE changes materializes nothing."""
    records = materialize_pending_records(
        _CATEGORY,
        [_change("gone", AuditOperation.DELETE, {"name": "gone"})],
        domain=CanonicalFormat.v2,
    )
    assert records == {}


def test_materialize_legacy_converts_to_v2(
    minimal_legacy_stable_diffusion_data: dict[str, Any],
) -> None:
    """A legacy-domain payload is converted to v2 shape via the canonical converter."""
    name, legacy_payload = next(iter(minimal_legacy_stable_diffusion_data.items()))

    records = materialize_pending_records(
        _CATEGORY,
        [_change(name, AuditOperation.CREATE, legacy_payload)],
        domain=CanonicalFormat.legacy,
    )

    assert name in records
    converted = records[name]
    # v2 records carry a normalized baseline and a config/download block, unlike the
    # legacy ``config.files`` shape that went in.
    assert converted["baseline"] == "stable_diffusion_1"
    assert "download" in converted["config"]
