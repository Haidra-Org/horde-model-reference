"""Verify category statistics behavior at the HTTP consumer boundary."""

from collections.abc import Callable

from fastapi.testclient import TestClient

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager
from horde_model_reference.service.shared import get_model_reference_manager


def test_known_empty_category_returns_zero_statistics(
    api_client: TestClient,
    primary_manager_override_factory: Callable[
        [Callable[[], ModelReferenceManager]],
        ModelReferenceManager,
    ],
) -> None:
    """Return an analyzable zero state for a valid category that has no records."""
    primary_manager_override_factory(get_model_reference_manager)

    response = api_client.get(
        f"/model_references/statistics/{MODEL_REFERENCE_CATEGORY.audio_generation.value}",
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["category"] == MODEL_REFERENCE_CATEGORY.audio_generation.value
    assert response_body["total_models"] == 0
    assert response_body["returned_models"] == 0
    assert response_body["baseline_distribution"] == {}


def test_runtime_statistics_reject_unsupported_category_as_client_error(
    api_client: TestClient,
    primary_manager_override_factory: Callable[
        [Callable[[], ModelReferenceManager]],
        ModelReferenceManager,
    ],
) -> None:
    """Tell consumers that utility records have no Horde runtime statistics."""
    primary_manager_override_factory(get_model_reference_manager)

    response = api_client.get(
        f"/model_references/statistics/{MODEL_REFERENCE_CATEGORY.blip.value}/with-stats",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Category 'blip' does not support Horde statistics. Only image_generation and text_generation are supported."
    )
