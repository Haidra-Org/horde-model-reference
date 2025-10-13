"""Tests for v2 API CRUD operations."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    ReplicateMode,
)
from horde_model_reference.service.shared import (
    PathVariables,
    RouteNames,
    route_registry,
    v2_prefix,
)
from horde_model_reference.service.v2.routers.references import (
    get_model_reference_manager,
)
from tests.helpers import ALL_MODEL_CATEGORIES


@pytest.fixture
def primary_manager_for_api(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
) -> Iterator[ModelReferenceManager]:
    """Create a PRIMARY mode manager for API tests."""
    manager = primary_manager_override_factory(get_model_reference_manager)
    yield manager


def _create_minimal_model_dict(
    name: str,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous,
    *,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a minimal valid model record for testing.

    Args:
        name: Model name
        category: Model category (determines schema and required fields)
        description: Optional description

    Returns:
        Category-appropriate model payload
    """
    model_dict: dict[str, Any] = {
        "name": name,
        "model_classification": {
            "domain": "image",
            "purpose": "miscellaneous",
        },
    }
    if description is not None:
        model_dict["description"] = description

    # Add category-specific required fields based on model definitions
    if category == MODEL_REFERENCE_CATEGORY.image_generation:
        model_dict["model_classification"]["purpose"] = "generation"
        model_dict["baseline"] = "stable_diffusion_1"
        model_dict["nsfw"] = False
    elif category == MODEL_REFERENCE_CATEGORY.text_generation:
        model_dict["model_classification"]["domain"] = "text"
        model_dict["model_classification"]["purpose"] = "generation"
        model_dict["parameters"] = 7000000000
    elif category == MODEL_REFERENCE_CATEGORY.clip:
        model_dict["model_classification"]["purpose"] = "feature_extractor"
    elif category == MODEL_REFERENCE_CATEGORY.controlnet:
        model_dict["model_classification"]["purpose"] = "auxiliary_or_patch"
        model_dict["style"] = "control_seg"
    elif category == MODEL_REFERENCE_CATEGORY.video_generation:
        model_dict["model_classification"]["domain"] = "video"
        model_dict["model_classification"]["purpose"] = "generation"
    elif category == MODEL_REFERENCE_CATEGORY.audio_generation:
        model_dict["model_classification"]["domain"] = "audio"
        model_dict["model_classification"]["purpose"] = "generation"

    return model_dict


def _model_url(
    route_name: RouteNames,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str | None = None,
) -> str:
    """Compose a v2 API URL for model endpoints.

    Args:
        route_name: Route name from RouteNames enum
        category: Model category
        model_name: Optional model name (required for single model operations)

    Returns:
        Formatted URL for the endpoint
    """
    path_vars: dict[str, str] = {PathVariables.model_category_name: category.value}
    if model_name is not None:
        path_vars[PathVariables.model_name] = model_name
    return route_registry.url_for(route_name, path_vars, v2_prefix)


def _assert_success_response(response: Response, expected_status: int = 200) -> dict[str, Any]:
    """Assert response has expected success status and return parsed JSON.

    Args:
        response: Response to check
        expected_status: Expected HTTP status code

    Returns:
        Parsed JSON data from response
    """
    assert response.status_code == expected_status
    json_data: dict[str, Any] = response.json()
    return json_data


def _assert_error_response(
    response: Response,
    expected_status: int,
    expected_substring: str,
) -> None:
    """Assert response has expected error status with message containing substring.

    Args:
        response: Response to check
        expected_status: Expected HTTP status code
        expected_substring: Expected substring in error detail (case-insensitive)
    """
    assert response.status_code == expected_status
    data = response.json()
    assert expected_substring.lower() in data["detail"].lower()


class TestGetSingleModel:
    """Tests for GET model by category and name endpoint."""

    def test_get_single_model_success(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """GET should return a single model when it exists."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "test_model"
        model_data = _create_minimal_model_dict(model_name, category, description="Test model")

        primary_manager_for_api.backend.update_model(category, model_name, model_data)

        response = api_client.get(_model_url(RouteNames.get_single_model, category, model_name))

        data = _assert_success_response(response)
        assert data["name"] == model_name
        assert "model_classification" in data

    def test_get_single_model_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """GET should return 404 when model doesn't exist."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "nonexistent_model"

        response = api_client.get(_model_url(RouteNames.get_single_model, category, model_name))

        _assert_error_response(response, 404, "not found")

    def test_get_single_model_category_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """GET should return 404 when category file doesn't exist."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "test_model"

        response = api_client.get(_model_url(RouteNames.get_single_model, category, model_name))

        _assert_error_response(response, 404, "category")


class TestCreateModel:
    """Tests for POST /{category}/{model_name} endpoint."""

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_create_model_success(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """POST should create a new model successfully."""
        model_name = "new_model"
        model_data = _create_minimal_model_dict(model_name, category, description="New test model")

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        created_data = _assert_success_response(response, 201)
        assert created_data["name"] == model_name

        raw_json = primary_manager_for_api.get_raw_model_reference_json(category)
        assert raw_json is not None
        assert model_name in raw_json

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_create_model_already_exists(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """POST should return 409 when model already exists."""
        model_name = "existing_model"
        model_data = _create_minimal_model_dict(model_name, category)

        primary_manager_for_api.backend.update_model(category, model_name, model_data)

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        _assert_error_response(response, 409, "already exists")

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_create_model_validation_error(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """POST should return 422 when model data is invalid."""
        model_name = "invalid_model"
        model_data = {
            "name": model_name,
            "random_field": "invalid_value",
        }

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        assert response.status_code == 422


class TestUpdateModel:
    """Tests for PUT /{category}/{model_name} endpoint."""

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_update_existing_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """PUT should update an existing model."""
        model_name = "update_test"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated")

        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=updated_data,
        )

        result = _assert_success_response(response)
        assert result["description"] == "Updated"
        assert "metadata" in result
        assert "updated_at" in result["metadata"]

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_update_creates_new_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """PUT should create a new model if it doesn't exist (upsert)."""
        model_name = "new_via_put"
        model_data = _create_minimal_model_dict(model_name, category, description="Created via PUT")

        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=model_data,
        )

        result = _assert_success_response(response, 201)
        assert result["name"] == model_name
        assert "created_at" in result["metadata"]

    def test_update_preserves_created_metadata(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT should preserve created_at and created_by on update."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "metadata_test"
        original_data = _create_minimal_model_dict(model_name)
        original_data["metadata"] = {
            "created_at": 1000000,
            "created_by": "original_user",
        }

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated")

        response = api_client.put(
            route_registry.url_for(
                RouteNames.update_model,
                {
                    PathVariables.model_category_name: category.value,
                    PathVariables.model_name: model_name,
                },
                v2_prefix,
            ),
            json=updated_data,
        )

        assert response.status_code == 200
        result = response.json()
        assert result["metadata"]["created_at"] == 1000000
        assert result["metadata"]["created_by"] == "original_user"
        assert "updated_at" in result["metadata"]

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_update_model_name_mismatch(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """PUT should return 400 when URL name doesn't match body name."""
        url_name = "url_name"
        body_name = "body_name"
        model_data = _create_minimal_model_dict(body_name, category)

        response = api_client.put(
            _model_url(RouteNames.update_model, category, url_name),
            json=model_data,
        )

        _assert_error_response(response, 400, "must match")

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_update_model_validation_error(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """PUT should return 422 when model data is invalid."""
        model_name = "invalid_update"
        model_data = {
            "name": model_name,
        }

        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=model_data,
        )

        assert response.status_code == 422


class TestDeleteModel:
    """Tests for DELETE /{category}/{model_name} endpoint."""

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_success(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should remove a model successfully."""
        model_name = "delete_test"
        model_data = _create_minimal_model_dict(model_name, category)

        primary_manager_for_api.backend.update_model(category, model_name, model_data)

        response = api_client.delete(_model_url(RouteNames.delete_model, category, model_name))

        assert response.status_code == 204

        raw_json = primary_manager_for_api.get_raw_model_reference_json(category)
        assert raw_json is None or model_name not in raw_json

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should return 404 when model doesn't exist."""
        model_name = "nonexistent"

        response = api_client.delete(_model_url(RouteNames.delete_model, category, model_name))

        _assert_error_response(response, 404, "not found")


class TestReplicaModeRestriction:
    """Tests to ensure REPLICA mode instances reject write operations."""

    @pytest.fixture
    def replica_manager(
        self,
        tmp_path: Path,
        restore_manager_singleton: None,
        dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    ) -> Iterator[ModelReferenceManager]:
        """Create a REPLICA mode manager for testing."""
        from horde_model_reference.backends.github_backend import GitHubBackend

        backend = GitHubBackend(
            base_path=tmp_path,
            replicate_mode=ReplicateMode.REPLICA,
        )
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.REPLICA,
        )

        dependency_override(get_model_reference_manager, lambda: manager)
        yield manager

    def test_create_fails_in_replica_mode(
        self,
        api_client: TestClient,
        replica_manager: ModelReferenceManager,
    ) -> None:
        """POST should return 503 in REPLICA mode."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "test_model"
        model_data = _create_minimal_model_dict(model_name, category)

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        _assert_error_response(response, 503, "REPLICA mode")

    def test_update_fails_in_replica_mode(
        self,
        api_client: TestClient,
        replica_manager: ModelReferenceManager,
    ) -> None:
        """PUT should return 503 in REPLICA mode."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "test_model"
        model_data = _create_minimal_model_dict(model_name)

        response = api_client.put(
            route_registry.url_for(
                RouteNames.update_model,
                {
                    PathVariables.model_category_name: category.value,
                    PathVariables.model_name: model_name,
                },
                v2_prefix,
            ),
            json=model_data,
        )

        assert response.status_code == 503
        assert "REPLICA mode" in response.json()["detail"]

    def test_delete_fails_in_replica_mode(
        self,
        api_client: TestClient,
        replica_manager: ModelReferenceManager,
    ) -> None:
        """DELETE should return 503 in REPLICA mode."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "test_model"

        response = api_client.delete(_model_url(RouteNames.delete_model, category, model_name))

        _assert_error_response(response, 503, "REPLICA mode")


class TestCacheInvalidationOnWrite:
    """Tests to ensure cache is properly invalidated after write operations."""

    def test_create_invalidates_cache(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Creating a model should invalidate the category cache."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_create"

        primary_manager_for_api.get_all_model_references()

        model_data = _create_minimal_model_dict(model_name)
        response = api_client.post(
            route_registry.url_for(
                RouteNames.create_model, {PathVariables.model_category_name: category.value}, v2_prefix
            ),
            json=model_data,
        )

        assert response.status_code == 201

        references = primary_manager_for_api.get_all_model_references()
        assert category in references
        if references[category]:
            assert model_name in references[category]

    def test_update_invalidates_cache(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Updating a model should invalidate the category cache."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_update"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)
        primary_manager_for_api.get_all_model_references()

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated")
        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=updated_data,
        )

        _assert_success_response(response)

        references = primary_manager_for_api.get_all_model_references()
        assert category in references
        if references[category] and model_name in references[category]:
            assert references[category][model_name].description == "Updated"

    def test_delete_invalidates_cache(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Deleting a model should invalidate the category cache."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_delete"
        model_data = _create_minimal_model_dict(model_name)

        primary_manager_for_api.backend.update_model(category, model_name, model_data)
        primary_manager_for_api.get_all_model_references()

        response = api_client.delete(
            route_registry.url_for(
                RouteNames.delete_model,
                {
                    PathVariables.model_category_name: category.value,
                    PathVariables.model_name: model_name,
                },
                v2_prefix,
            )
        )

        assert response.status_code == 204

        references = primary_manager_for_api.get_all_model_references()
        assert category in references
        assert not references[category] or model_name not in references[category]


class TestImageGenerationModelValidation:
    """Tests for category-specific validation (ImageGenerationModelRecord)."""

    def test_create_image_generation_model_with_required_fields(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Creating an image_generation model should validate required fields."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "test_sd_model"
        model_data = _create_minimal_model_dict(model_name, category)

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        result = _assert_success_response(response, 201)
        assert result["baseline"] == "stable_diffusion_1"
        assert result["nsfw"] is False

    def test_create_image_generation_model_missing_required_field(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Creating an image_generation model without required fields should fail."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "invalid_sd_model"
        model_data = {
            "name": model_name,
            "model_classification": {
                "domain": "image",
                "purpose": "generation",
            },
        }

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
        )

        assert response.status_code == 422
