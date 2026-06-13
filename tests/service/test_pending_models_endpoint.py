"""End-to-end tests for the v2 ``/{category}/pending`` (beta models) endpoint."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    CanonicalFormat,
    ModelReferenceManager,
    horde_model_reference_settings,
)
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue import PendingQueueService
from horde_model_reference.service.shared import get_model_reference_manager

pytestmark = pytest.mark.usefixtures("mock_auth_success")

_API_HEADERS = {"apikey": "test_key"}
_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation
_PENDING_URL = f"/model_references/v2/{_CATEGORY.value}/pending"


@pytest.fixture
def v2_primary_manager(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """Return a PRIMARY manager with canonical v2 so pending payloads need no conversion."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    yield primary_manager_override_factory(get_model_reference_manager)


def _enqueue(
    service: PendingQueueService,
    *,
    model_name: str,
    operation: AuditOperation = AuditOperation.CREATE,
    nsfw: bool = False,
) -> int:
    record = service.enqueue_change(
        category=_CATEGORY,
        model_name=model_name,
        operation=operation,
        payload={"name": model_name, "baseline": "stable_diffusion_1", "nsfw": nsfw},
        requestor_id="test-user-id",
        requestor_username="tester#test-user-id",
        notes=None,
        request_metadata={"source": "tests"},
    )
    return record.change_id


def test_pending_endpoint_requires_auth(api_client: TestClient, v2_primary_manager: ModelReferenceManager) -> None:
    """Without an API key the endpoint is unauthorized."""
    response = api_client.get(_PENDING_URL)
    assert response.status_code in (401, 403)


def test_pending_endpoint_surfaces_pending_and_approved_create_update(
    api_client: TestClient,
    v2_primary_manager: ModelReferenceManager,
) -> None:
    """PENDING + APPROVED create/update changes are materialized; DELETE is skipped."""
    service = v2_primary_manager.pending_queue_service
    assert service is not None

    _enqueue(service, model_name="beta_pending")
    approved = _enqueue(service, model_name="beta_approved")
    _enqueue(service, model_name="beta_deleted", operation=AuditOperation.DELETE)

    service.process_batch(
        approver_id="test-user-id",
        approver_username="tester#test-user-id",
        batch_title="approve-one",
        approved_ids=[approved],
        rejected_ids=None,
    )

    response = api_client.get(_PENDING_URL, headers=_API_HEADERS)
    assert response.status_code == 200

    payload = response.json()
    assert set(payload) == {"beta_pending", "beta_approved"}
    assert payload["beta_pending"]["baseline"] == "stable_diffusion_1"


def test_pending_endpoint_empty_when_no_changes(
    api_client: TestClient,
    v2_primary_manager: ModelReferenceManager,
) -> None:
    """An empty queue yields an empty mapping (not a 404)."""
    response = api_client.get(_PENDING_URL, headers=_API_HEADERS)
    assert response.status_code == 200
    assert response.json() == {}
