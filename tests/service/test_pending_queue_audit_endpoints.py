from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    horde_model_reference_settings,
)
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue import PendingQueueService
from horde_model_reference.service.shared import get_model_reference_manager

pytestmark = pytest.mark.usefixtures("mock_auth_success")

_REQUESTOR_ID = "requestor-1"
_REQUESTOR_USERNAME = "tester#requestor"
_APPROVER_ID = "approver-1"
_APPROVER_USERNAME = "tester#approver"
_API_HEADERS = {"apikey": "test_key"}


@pytest.fixture
def isolated_audit_root(tmp_path: Path) -> Iterator[None]:
    """Provide an empty audit root so tests never read historical events."""
    audit_root = tmp_path / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    previous_override = horde_model_reference_settings.audit.root_path_override
    horde_model_reference_settings.audit.root_path_override = str(audit_root)
    try:
        yield
    finally:
        horde_model_reference_settings.audit.root_path_override = previous_override


@pytest.fixture
def v2_primary_manager(
    isolated_audit_root: None,
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """PRIMARY manager with canonical v2 for pending queue audit tests."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "v2")
    manager = primary_manager_override_factory(get_model_reference_manager)
    yield manager


def _enqueue(
    service: PendingQueueService,
    *,
    model_name: str,
    operation: AuditOperation = AuditOperation.CREATE,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous,
) -> int:
    record = service.enqueue_change(
        category=category,
        model_name=model_name,
        operation=operation,
        payload={"name": model_name, "record_type": category.value},
        requestor_id=_REQUESTOR_ID,
        requestor_username=_REQUESTOR_USERNAME,
        notes=None,
        request_metadata={"source": "tests"},
    )
    return record.change_id


def test_pending_queue_audit_endpoints_surfaces_batches(
    api_client: TestClient,
    v2_primary_manager: ModelReferenceManager,
) -> None:
    """End-to-end check that new audit endpoints expose pending and historical data."""
    service = v2_primary_manager.pending_queue_service
    assert service is not None

    pending_change = _enqueue(service, model_name="pending-model")
    approved_change = _enqueue(service, model_name="to-approve")

    batch = service.process_batch(
        approver_id=_APPROVER_ID,
        approver_username=_APPROVER_USERNAME,
        batch_title="batch-1",
        approved_ids=[approved_change],
        rejected_ids=None,
    )
    service.mark_applied(
        change_id=approved_change,
        applied_by=_APPROVER_ID,
        applied_username=_APPROVER_USERNAME,
        job_id="job-123",
    )

    current_resp = api_client.get(
        "/model_references/v2/pending_queue/audit/current",
        headers=_API_HEADERS,
    )
    assert current_resp.status_code == 200
    current_payload = current_resp.json()
    assert current_payload["domain"] == "v2"
    assert current_payload["total_pending"] == 1
    change_ids = {item["change_id"] for item in current_payload["pending_changes"]}
    assert pending_change in change_ids

    batches_resp = api_client.get(
        "/model_references/v2/pending_queue/audit/batches",
        params={"limit": 5},
        headers=_API_HEADERS,
    )
    assert batches_resp.status_code == 200
    batches_payload = batches_resp.json()
    assert batches_payload["domain"] == "v2"
    assert batches_payload["batches"]
    first_batch = batches_payload["batches"][0]
    assert first_batch["batch_id"] == batch.batch_id
    assert first_batch["approved_change_count"] == 1
    assert first_batch["applied_change_count"] == 1

    detail_resp = api_client.get(
        f"/model_references/v2/pending_queue/audit/batches/{batch.batch_id}",
        headers=_API_HEADERS,
    )
    assert detail_resp.status_code == 200
    detail_payload = detail_resp.json()
    assert detail_payload["batch_id"] == batch.batch_id
    detail_change_ids = {item["change_id"] for item in detail_payload["changes"]}
    assert approved_change in detail_change_ids


def test_pending_queue_audit_defaults_to_legacy_domain(
    api_client: TestClient,
    isolated_audit_root: None,
    v1_canonical_manager: ModelReferenceManager,
) -> None:
    """Ensure canonical legacy deployments default to the legacy audit domain."""
    service = v1_canonical_manager.pending_queue_service
    assert service is not None

    approved_change = _enqueue(service, model_name="legacy-model")
    batch = service.process_batch(
        approver_id=_APPROVER_ID,
        approver_username=_APPROVER_USERNAME,
        batch_title="legacy-batch",
        approved_ids=[approved_change],
        rejected_ids=None,
    )

    response = api_client.get(
        "/model_references/v1/pending_queue/audit/batches",
        headers=_API_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "legacy"
    assert payload["batches"][0]["batch_id"] == batch.batch_id

    override_resp = api_client.get(
        "/model_references/v1/pending_queue/audit/batches",
        params={"domain_override": "legacy", "limit": 1},
        headers=_API_HEADERS,
    )
    assert override_resp.status_code == 200
    assert override_resp.json()["domain"] == "legacy"


def test_pending_queue_audit_requires_authentication(
    api_client: TestClient,
    v2_primary_manager: ModelReferenceManager,
) -> None:
    """Ensure audit endpoints reject requests without the required API key."""
    service = v2_primary_manager.pending_queue_service
    assert service is not None

    _enqueue(service, model_name="secured-model")

    response = api_client.get("/model_references/v2/pending_queue/audit/current")
    assert response.status_code in {401, 403}
