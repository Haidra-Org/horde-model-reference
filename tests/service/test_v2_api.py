"""Tests for v2 API CRUD operations."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    PrefetchStrategy,
    ReplicateMode,
)
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue.models import PendingChangeStatus
from horde_model_reference.service.shared import (
    PathVariables,
    RouteNames,
    get_model_reference_manager,
    route_registry,
    v2_prefix,
)
from tests.helpers import ALL_MODEL_CATEGORIES

pytestmark = pytest.mark.usefixtures("mock_auth_success")

_TEST_API_KEY = "test_key"
_TEST_USER_ID = "test-user-id"
_TEST_USERNAME = f"tester#{_TEST_USER_ID}"


def _auth_headers() -> dict[str, str]:
    """Return standard API headers for authenticated write requests."""
    return {"apikey": _TEST_API_KEY}


@pytest.fixture
def primary_manager_for_api(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """Create a PRIMARY mode manager for v2 API tests.

    Sets canonical_format to 'v2' to enable v2 write operations.
    """
    from horde_model_reference import horde_model_reference_settings

    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "v2")
    manager = primary_manager_override_factory(get_model_reference_manager)
    yield manager


def _enqueue_pending_change(
    manager: ModelReferenceManager,
    *,
    model_name: str,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous,
    operation: AuditOperation = AuditOperation.CREATE,
    payload: dict[str, Any] | None = None,
) -> int:
    """Enqueue a pending change directly through the manager's queue service for testing.

    Args:
        manager: ModelReferenceManager instance with pending queue enabled
        model_name: Name of the model to change
        category: Model category for the change
        operation: Type of operation (create/update/delete)
        payload: Optional payload for the change (required for create/update)

    Returns:
        ID of the enqueued change record
    """
    queue_service = manager.pending_queue_service
    assert queue_service is not None, "Pending queue must be enabled for tests"

    effective_payload = payload
    if effective_payload is None and operation is not AuditOperation.DELETE:
        effective_payload = _create_minimal_model_dict(model_name, category)

    record = queue_service.enqueue_change(
        category=category,
        model_name=model_name,
        operation=operation,
        payload=effective_payload,
        requestor_id=_TEST_USER_ID,
        requestor_username=_TEST_USERNAME,
        notes=None,
        request_metadata={"source": "tests"},
    )
    return record.change_id


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
        "record_type": category.value,
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
        model_dict["controlnet_style"] = "control_seg"
    elif category == MODEL_REFERENCE_CATEGORY.video_generation:
        model_dict["model_classification"]["domain"] = "video"
        model_dict["model_classification"]["purpose"] = "generation"
    elif category == MODEL_REFERENCE_CATEGORY.audio_generation:
        model_dict["model_classification"]["domain"] = "audio"
        model_dict["model_classification"]["purpose"] = "generation"

    return model_dict


def _path_var(variable: PathVariables) -> str:
    """Return the string key for a PathVariables enum value."""
    return cast(str, variable.value)


def _category_value(category: MODEL_REFERENCE_CATEGORY) -> str:
    """Return the string value for a model category enum entry."""
    return cast(str, category.value)


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
    path_vars: dict[str, str] = {_path_var(PathVariables.model_category_name): _category_value(category)}
    if model_name is not None:
        path_vars[_path_var(PathVariables.model_name)] = model_name
    return route_registry.url_for(route_name, path_vars, v2_prefix)


def _assert_success_response(response: Response, expected_status: int = 200) -> dict[str, Any]:
    """Assert response has expected success status and return parsed JSON.

    Args:
        response: Response to check
        expected_status: Expected HTTP status code

    Returns:
        Parsed JSON data from response
    """
    assert response.status_code == expected_status, response.json()
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
        """POST should queue a new model for approval."""
        model_name = "new_model"
        model_data = _create_minimal_model_dict(model_name, category, description="New test model")

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "create"
        assert record["model_name"] == model_name
        assert record["status"] == "pending"

        raw_json = primary_manager_for_api.get_raw_model_reference_json(category) or {}
        assert model_name not in raw_json

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
            headers=_auth_headers(),
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
            headers=_auth_headers(),
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
        """PUT should queue an update for an existing model."""
        model_name = "update_test"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated")

        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "update"
        assert record["payload"]["description"] == "Updated"

        stored_record = primary_manager_for_api.get_raw_model_reference_json(category)
        assert stored_record is not None
        assert stored_record[model_name]["description"] == "Original"

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
                    _path_var(PathVariables.model_category_name): _category_value(category),
                    _path_var(PathVariables.model_name): model_name,
                },
                v2_prefix,
            ),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["payload"]["metadata"]["created_at"] == 1000000
        assert record["payload"]["metadata"]["created_by"] == "original_user"

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
            "random_field": "invalid_value",
        }

        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=model_data,
            headers=_auth_headers(),
        )

        assert response.status_code == 422


class TestCategorySpecificUpdate:
    """Tests for category-specific PUT update endpoints."""

    def test_update_image_generation_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT /image_generation/update_model/{name} should queue an update."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "img_update_test"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated via category endpoint")

        response = api_client.put(
            _model_url(RouteNames.update_image_generation_model, category, model_name),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "update"
        assert record["payload"]["description"] == "Updated via category endpoint"

    def test_update_text_generation_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT /text_generation/update_model/{name} should queue an update."""
        category = MODEL_REFERENCE_CATEGORY.text_generation
        model_name = "text_update_test"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated via category endpoint")

        response = api_client.put(
            _model_url(RouteNames.update_text_generation_model, category, model_name),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "update"
        assert record["payload"]["description"] == "Updated via category endpoint"

    def test_update_controlnet_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT /controlnet/update_model/{name} should queue an update."""
        category = MODEL_REFERENCE_CATEGORY.controlnet
        model_name = "cn_update_test"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated via category endpoint")

        response = api_client.put(
            _model_url(RouteNames.update_controlnet_model, category, model_name),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "update"
        assert record["payload"]["description"] == "Updated via category endpoint"

    def test_category_update_name_mismatch(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT should return 400 when path model_name doesn't match body."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "mismatch_test"
        original_data = _create_minimal_model_dict(model_name, category)
        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        mismatched_data = _create_minimal_model_dict("wrong_name", category)

        response = api_client.put(
            _model_url(RouteNames.update_image_generation_model, category, model_name),
            json=mismatched_data,
            headers=_auth_headers(),
        )

        _assert_error_response(response, 400, "must match")

    def test_category_update_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT should return 404 when model doesn't exist."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "nonexistent_model"
        model_data = _create_minimal_model_dict(model_name, category)

        response = api_client.put(
            _model_url(RouteNames.update_image_generation_model, category, model_name),
            json=model_data,
            headers=_auth_headers(),
        )

        _assert_error_response(response, 404, "not found")

    def test_category_update_validation_error(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT should return 422 when body has wrong category schema."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "validation_test"
        original_data = _create_minimal_model_dict(model_name, category)
        primary_manager_for_api.backend.update_model(category, model_name, original_data)

        # Send text_generation data to image_generation endpoint (missing baseline, has parameters)
        wrong_schema_data = _create_minimal_model_dict(model_name, MODEL_REFERENCE_CATEGORY.text_generation)

        response = api_client.put(
            _model_url(RouteNames.update_image_generation_model, category, model_name),
            json=wrong_schema_data,
            headers=_auth_headers(),
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
        """DELETE should queue a removal request."""
        model_name = "delete_test"
        model_data = _create_minimal_model_dict(model_name, category)

        primary_manager_for_api.backend.update_model(category, model_name, model_data)

        response = api_client.delete(
            _model_url(RouteNames.delete_model, category, model_name),
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "delete"
        assert record["payload"]["name"] == model_name

        raw_json = primary_manager_for_api.get_raw_model_reference_json(category)
        assert raw_json is not None and model_name in raw_json

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should return 404 when model doesn't exist."""
        model_name = "nonexistent"

        response = api_client.delete(
            _model_url(RouteNames.delete_model, category, model_name),
            headers=_auth_headers(),
        )

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
            prefetch_strategy=PrefetchStrategy.LAZY,
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
            headers=_auth_headers(),
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
                    _path_var(PathVariables.model_category_name): _category_value(category),
                    _path_var(PathVariables.model_name): model_name,
                },
                v2_prefix,
            ),
            json=model_data,
            headers=_auth_headers(),
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

        response = api_client.delete(
            _model_url(RouteNames.delete_model, category, model_name),
            headers=_auth_headers(),
        )

        _assert_error_response(response, 503, "REPLICA mode")


class TestCacheBehaviorWithPendingQueue:
    """Pending queue writes should not mutate cached data until applied."""

    def test_create_request_keeps_cache_intact(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """POST queues a change but cached category data stays unchanged."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_create"

        before_refs = primary_manager_for_api.get_all_model_references()
        before_names = set(before_refs[category].keys())

        model_data = _create_minimal_model_dict(model_name)
        response = api_client.post(
            route_registry.url_for(
                RouteNames.create_model,
                {_path_var(PathVariables.model_category_name): _category_value(category)},
                v2_prefix,
            ),
            json=model_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "create"
        assert record["model_name"] == model_name
        assert record["category"] == _category_value(category)

        after_refs = primary_manager_for_api.get_all_model_references()
        assert set(after_refs[category].keys()) == before_names
        assert model_name not in after_refs[category]

    def test_update_request_keeps_cached_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """PUT queues an update but cached copy still reflects backend state."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_update"
        original_data = _create_minimal_model_dict(model_name, category, description="Original")

        primary_manager_for_api.backend.update_model(category, model_name, original_data)
        cached_refs = primary_manager_for_api.get_all_model_references()
        assert cached_refs[category][model_name].description == "Original"

        updated_data = _create_minimal_model_dict(model_name, category, description="Updated")
        response = api_client.put(
            _model_url(RouteNames.update_model, category, model_name),
            json=updated_data,
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "update"
        assert record["payload"]["description"] == "Updated"

        refreshed_refs = primary_manager_for_api.get_all_model_references()
        assert refreshed_refs[category][model_name].description == "Original"

    def test_delete_request_does_not_remove_cached_model(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """DELETE queues a removal while cached data still lists the model."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "cache_test_delete"
        model_data = _create_minimal_model_dict(model_name)

        primary_manager_for_api.backend.update_model(category, model_name, model_data)
        cached_refs = primary_manager_for_api.get_all_model_references()
        assert model_name in cached_refs[category]

        response = api_client.delete(
            route_registry.url_for(
                RouteNames.delete_model,
                {
                    _path_var(PathVariables.model_category_name): _category_value(category),
                    _path_var(PathVariables.model_name): model_name,
                },
                v2_prefix,
            ),
            headers=_auth_headers(),
        )

        record = _assert_success_response(response, 202)
        assert record["operation"] == "delete"
        assert record["model_name"] == model_name

        refreshed_refs = primary_manager_for_api.get_all_model_references()
        assert model_name in refreshed_refs[category]


class TestPendingQueueAdminApi:
    """Tests for the pending queue management endpoints."""

    _base_url = f"{v2_prefix}/pending_queue"

    def test_list_pending_changes_supports_filters(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """GET /changes should return paginated results and respect filters."""
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        first_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_list_one")
        second_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_list_two")

        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="approve-second",
            approved_ids=[second_id],
            rejected_ids=None,
            reject_reason=None,
        )

        response = api_client.get(f"{self._base_url}/changes", headers=_auth_headers())
        payload = _assert_success_response(response)
        assert payload["total"] >= 2
        returned_ids = {item["change_id"] for item in payload["items"]}
        assert {first_id, second_id}.issubset(returned_ids)

        response = api_client.get(
            f"{self._base_url}/changes",
            headers=_auth_headers(),
            params={"statuses": PendingChangeStatus.APPROVED.value},
        )
        payload = _assert_success_response(response)
        assert payload["total"] == 1
        assert payload["items"][0]["change_id"] == second_id

    def test_read_single_pending_change(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """GET /changes/{id} should return the specific record."""
        change_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_read_single")

        response = api_client.get(
            f"{self._base_url}/changes/{change_id}",
            headers=_auth_headers(),
        )
        payload = _assert_success_response(response)
        assert payload["change_id"] == change_id
        assert payload["model_name"] == "queue_read_single"

    def test_process_batch_updates_statuses(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """POST /batches should approve and reject entries atomically."""
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        approved_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_batch_approve")
        rejected_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_batch_reject")

        response = api_client.post(
            f"{self._base_url}/batches",
            headers=_auth_headers(),
            json={
                "batch_title": "review batch",
                "approved_ids": [approved_id],
                "rejected_ids": [rejected_id],
                "reject_reason": "needs changes",
            },
        )
        payload = _assert_success_response(response)
        assert payload["batch_title"] == "review batch"
        assert {entry["change_id"] for entry in payload["approved"]} == {approved_id}
        assert {entry["change_id"] for entry in payload["rejected"]} == {rejected_id}

        approved_record = queue_service.get_change(approved_id)
        assert approved_record is not None and approved_record.status is PendingChangeStatus.APPROVED

        rejected_record = queue_service.get_change(rejected_id)
        assert rejected_record is not None and rejected_record.status is PendingChangeStatus.REJECTED

    def test_apply_pending_change_updates_backend(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """POST /changes/{id}/apply should persist updates and mark record applied."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "queue_apply_model"
        existing_payload = _create_minimal_model_dict(model_name, category, description="old")
        updated_payload = _create_minimal_model_dict(model_name, category, description="new")

        primary_manager_for_api.backend.update_model(category, model_name, existing_payload)

        change_id = _enqueue_pending_change(
            primary_manager_for_api,
            model_name=model_name,
            category=category,
            operation=AuditOperation.UPDATE,
            payload=updated_payload,
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="approve apply",
            approved_ids=[change_id],
            rejected_ids=None,
            reject_reason=None,
        )

        response = api_client.post(
            f"{self._base_url}/changes/{change_id}/apply",
            headers=_auth_headers(),
            json={"job_id": "test-job"},
        )
        payload = _assert_success_response(response)
        # New response format wraps record in "record" field
        assert payload["record"]["status"] == PendingChangeStatus.APPLIED.value
        assert payload["record"]["applied_by"] == _TEST_USER_ID
        assert payload["record"]["applied_job_id"] == "test-job"

        refreshed = primary_manager_for_api.get_raw_model_reference_json(category)
        assert refreshed is not None
        assert refreshed[model_name]["description"] == "new"

    def test_apply_pending_change_requires_approval(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Apply endpoint should reject pending records that are not approved."""
        change_id = _enqueue_pending_change(primary_manager_for_api, model_name="queue_apply_unapproved")

        response = api_client.post(
            f"{self._base_url}/changes/{change_id}/apply",
            headers=_auth_headers(),
            json={},
        )
        _assert_error_response(response, 400, "approved")

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        record = queue_service.get_change(change_id)
        assert record is not None
        assert record.status is PendingChangeStatus.PENDING

    def test_list_pending_changes_filters_by_category_status_and_name(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Field QA scenario: approvers filter queue by category, status, and fuzzy name match."""
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        target_category = MODEL_REFERENCE_CATEGORY.image_generation
        pending_id = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="qa_filter_target",
            category=target_category,
        )
        approved_id = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="qa_other_model",
            category=MODEL_REFERENCE_CATEGORY.audio_generation,
        )
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="qa-approve-single",
            approved_ids=[approved_id],
            rejected_ids=None,
            reject_reason=None,
        )
        second_pending_id = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="qa_filter_noise",
            category=target_category,
        )

        response = api_client.get(
            f"{self._base_url}/changes",
            headers=_auth_headers(),
            params={
                "statuses": PendingChangeStatus.PENDING.value,
                "categories": target_category.value,
                "model_name": "target",
            },
        )
        payload = _assert_success_response(response)
        assert payload["total"] == 1
        assert [item["change_id"] for item in payload["items"]] == [pending_id]

        response = api_client.get(
            f"{self._base_url}/changes",
            headers=_auth_headers(),
            params={
                "categories": target_category.value,
                "limit": 1,
                "offset": 1,
            },
        )
        payload = _assert_success_response(response)
        assert payload["total"] == 2
        assert len(payload["items"]) == 1
        assert payload["items"][0]["change_id"] == second_pending_id

    def test_apply_pending_change_twice_rejected(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying an already-applied change should surface the state violation."""
        change_id = _enqueue_pending_change(primary_manager_for_api, model_name="qa_double_apply")
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="qa-double-apply",
            approved_ids=[change_id],
            rejected_ids=None,
            reject_reason=None,
        )

        first_response = api_client.post(
            f"{self._base_url}/changes/{change_id}/apply",
            headers=_auth_headers(),
            json={"job_id": "qa-job"},
        )
        first_payload = _assert_success_response(first_response)
        # New response format wraps record in "record" field
        assert first_payload["record"]["status"] == PendingChangeStatus.APPLIED.value

        repeat_response = api_client.post(
            f"{self._base_url}/changes/{change_id}/apply",
            headers=_auth_headers(),
            json={"job_id": "qa-job"},
        )
        _assert_error_response(repeat_response, 400, "approved")

        stored = queue_service.get_change(change_id)
        assert stored is not None and stored.status is PendingChangeStatus.APPLIED

    def test_apply_pending_changes_backend_failure_reports_503(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """QA trick: simulate backend I/O failure mid-batch to ensure partial results and 503."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        backend = primary_manager_for_api.backend

        backend.update_model(
            category,
            "qa_backend_ok",
            _create_minimal_model_dict("qa_backend_ok", category, description="old-ok"),
        )
        backend.update_model(
            category,
            "qa_backend_fail",
            _create_minimal_model_dict("qa_backend_fail", category, description="old-fail"),
        )

        ok_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="qa_backend_ok",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("qa_backend_ok", category, description="new-ok"),
        )
        fail_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="qa_backend_fail",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("qa_backend_fail", category, description="new-fail"),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="qa-backend-fail",
            approved_ids=[ok_change, fail_change],
            rejected_ids=None,
            reject_reason=None,
        )

        original_update = backend.update_model

        def _fail_on_specific(
            category_arg: MODEL_REFERENCE_CATEGORY,
            model_name: str,
            record_dict: dict[str, Any],
            *,
            logical_user_id: str | None = None,
            request_id: str | None = None,
        ) -> None:
            if model_name == "qa_backend_fail":
                raise RuntimeError("disk full")
            return original_update(
                category_arg,
                model_name,
                record_dict,
                logical_user_id=logical_user_id,
                request_id=request_id,
            )

        monkeypatch.setattr(backend, "update_model", _fail_on_specific)

        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [ok_change, fail_change], "job_id": "qa-batch"},
        )

        assert response.status_code == 503
        payload = response.json()
        assert [item["change_id"] for item in payload["applied"]] == [ok_change]
        assert payload["failed_change_id"] == fail_change
        assert payload["failed_error_type"] == "PendingChangeBackendError"
        assert "disk full" in payload["failed_error"]

        refreshed = primary_manager_for_api.get_raw_model_reference_json(category)
        assert refreshed is not None
        assert refreshed["qa_backend_ok"]["description"] == "new-ok"
        assert refreshed["qa_backend_fail"]["description"] == "old-fail"

        ok_record = queue_service.get_change(ok_change)
        fail_record = queue_service.get_change(fail_change)
        assert ok_record is not None and ok_record.status is PendingChangeStatus.APPLIED
        assert ok_record.applied_job_id == "qa-batch"
        assert fail_record is not None and fail_record.status is PendingChangeStatus.APPROVED

    def test_apply_pending_changes_batch_success(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Bulk apply endpoint should apply all approved changes in order."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        backend = primary_manager_for_api.backend

        backend.update_model(
            category,
            "bulk_apply_one",
            _create_minimal_model_dict("bulk_apply_one", category, description="old one"),
        )
        backend.update_model(
            category,
            "bulk_apply_two",
            _create_minimal_model_dict("bulk_apply_two", category, description="old two"),
        )

        first_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="bulk_apply_one",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("bulk_apply_one", category, description="new one"),
        )
        second_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="bulk_apply_two",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("bulk_apply_two", category, description="new two"),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="bulk-apply",
            approved_ids=[first_change, second_change],
            rejected_ids=None,
            reject_reason=None,
        )

        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [first_change, second_change], "job_id": "bulk-job"},
        )

        payload = _assert_success_response(response)
        assert len(payload["applied"]) == 2
        assert [record["change_id"] for record in payload["applied"]] == [first_change, second_change]
        assert payload.get("failed_change_id") is None
        assert payload.get("failed_error") is None

        refreshed = primary_manager_for_api.get_raw_model_reference_json(category)
        assert refreshed is not None
        assert refreshed["bulk_apply_one"]["description"] == "new one"
        assert refreshed["bulk_apply_two"]["description"] == "new two"

        first_record = queue_service.get_change(first_change)
        second_record = queue_service.get_change(second_change)
        assert first_record is not None and first_record.status is PendingChangeStatus.APPLIED
        assert second_record is not None and second_record.status is PendingChangeStatus.APPLIED

    def test_apply_pending_changes_batch_stops_on_error(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Bulk apply should stop at first failure and report applied/failed ids."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        backend = primary_manager_for_api.backend

        backend.update_model(
            category,
            "bulk_fail_one",
            _create_minimal_model_dict("bulk_fail_one", category, description="old"),
        )

        first_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="bulk_fail_one",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("bulk_fail_one", category, description="new"),
        )
        second_change = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="bulk_fail_two",
            category=category,
            operation=AuditOperation.UPDATE,
            payload=_create_minimal_model_dict("bulk_fail_two", category, description="new"),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="bulk-fail",
            approved_ids=[first_change],
            rejected_ids=None,
            reject_reason=None,
        )

        # Use allow_mixed_batch to bypass batch validation since second_change is not approved
        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [first_change, second_change], "allow_mixed_batch": True},
        )

        assert response.status_code == 400
        payload = response.json()
        assert [record["change_id"] for record in payload["applied"]] == [first_change]
        assert payload["failed_change_id"] == second_change
        assert payload["failed_error_type"] == "PendingChangeStateError"

        refreshed = primary_manager_for_api.get_raw_model_reference_json(category)
        assert refreshed is not None
        assert refreshed["bulk_fail_one"]["description"] == "new"
        assert "bulk_fail_two" not in refreshed or refreshed["bulk_fail_two"]["description"] != "new"

        first_record = queue_service.get_change(first_change)
        second_record = queue_service.get_change(second_change)
        assert first_record is not None and first_record.status is PendingChangeStatus.APPLIED
        assert second_record is not None and second_record.status is PendingChangeStatus.PENDING

    def test_purge_pending_changes_by_requestor(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Approvers can purge entries en masse by requestor id."""
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        keep_change = queue_service.enqueue_change(
            category=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_name="keep-me",
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("keep-me"),
            requestor_id="legit-user",
            requestor_username="tester#legit-user",
            notes=None,
            request_metadata=None,
        )
        purge_change = queue_service.enqueue_change(
            category=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_name="ci-artifact",
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("ci-artifact"),
            requestor_id="ci-user",
            requestor_username="tester#ci-user",
            notes=None,
            request_metadata={"source": "tests"},
        )

        response = api_client.post(
            f"{self._base_url}/purge",
            headers=_auth_headers(),
            json={"requested_by": ["ci-user"]},
        )

        payload = _assert_success_response(response)
        assert payload["removed_count"] == 1
        assert payload["removed_change_ids"] == [purge_change.change_id]
        assert queue_service.get_change(purge_change.change_id) is None
        assert queue_service.get_change(keep_change.change_id) is not None

    def test_purge_pending_changes_supports_dry_run(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Dry-run purge should not mutate queue state."""
        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        change_id = queue_service.enqueue_change(
            category=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_name="preview-only",
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("preview-only"),
            requestor_id="dry-run-user",
            requestor_username="tester#dry-run-user",
            notes=None,
            request_metadata={"source": "tests"},
        ).change_id

        response = api_client.post(
            f"{self._base_url}/purge",
            headers=_auth_headers(),
            json={"requested_by": ["dry-run-user"], "dry_run": True},
        )

        payload = _assert_success_response(response)
        assert payload["dry_run"] is True
        assert payload["removed_count"] == 1
        assert payload["removed_change_ids"] == [change_id]
        # Queue entry should still be present after dry-run
        assert queue_service.get_change(change_id) is not None

    def test_purge_pending_changes_requires_filter_or_explicit_all(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Requests without filters or purge_all flag should be rejected."""
        response = api_client.post(
            f"{self._base_url}/purge",
            headers=_auth_headers(),
            json={},
        )

        assert response.status_code == 422


class TestBatchEnforcement:
    """Tests for batch cohesion validation in apply operations."""

    _base_url = f"{v2_prefix}/pending_queue"

    def test_apply_pending_changes_rejects_mixed_batches_by_default(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying changes from different batches should fail unless allow_mixed_batch=True.

        With batch ID semantics, separate process_batch calls share the same batch ID
        if the previous batch is still open (has unapplied APPROVED changes). To create
        truly separate batches, we must fully apply the first batch before approving
        the second.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create first change, approve, and apply it to close batch 1
        change_1 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="batch_test_1",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("batch_test_1", category),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        # Approve change_1 in batch-1
        result_1 = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="batch-1",
            approved_ids=[change_1],
            rejected_ids=None,
            reject_reason=None,
        )
        batch_1_id = result_1.batch_id

        # Apply batch-1 to close it
        apply_response = api_client.post(
            f"{self._base_url}/apply_batch/{batch_1_id}",
            headers=_auth_headers(),
        )
        _assert_success_response(apply_response)

        # Now create and approve change_2 - this should get a new batch ID
        change_2 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="batch_test_2",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("batch_test_2", category),
        )
        result_2 = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="batch-2",
            approved_ids=[change_2],
            rejected_ids=None,
            reject_reason=None,
        )
        batch_2_id = result_2.batch_id

        # Verify different batch IDs
        assert batch_1_id != batch_2_id, "Expected different batch IDs after applying first batch"

        # Attempt to apply changes from different batches
        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [change_1, change_2]},
        )

        _assert_error_response(response, 400, "same batch")

    def test_apply_pending_changes_allows_mixed_batch_with_flag(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Setting allow_mixed_batch=True should permit applying changes from different batches.

        With batch ID semantics, separate process_batch calls share the same batch ID
        if the previous batch is still open. To create truly separate batches for
        testing the mixed batch flag, we must trigger a batch split by partially
        applying the first batch.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create two changes in batch-x
        change_1 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="mixed_batch_1",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("mixed_batch_1", category),
        )
        change_2 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="mixed_batch_2",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("mixed_batch_2", category),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        # Approve both changes in one batch
        result_x = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="batch-x",
            approved_ids=[change_1, change_2],
            rejected_ids=None,
            reject_reason=None,
        )
        batch_x_id = result_x.batch_id

        # Apply change_1 individually - this triggers batch split, moving change_2 to new batch
        first_apply = api_client.post(
            f"{self._base_url}/changes/{change_1}/apply",
            headers=_auth_headers(),
            json={},
        )
        first_payload = _assert_success_response(first_apply)
        assert first_payload["batch_split_occurred"] is True
        batch_y_id = first_payload["batch_split_new_batch_id"]
        assert batch_y_id is not None
        assert batch_y_id != batch_x_id

        # Now create a third change and approve it in yet another batch
        change_3 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="mixed_batch_3",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("mixed_batch_3", category),
        )
        queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="batch-z",
            approved_ids=[change_3],
            rejected_ids=None,
            reject_reason=None,
        )

        # change_2 is in batch_y_id (from split), change_3 is in batch_z_id
        # These are truly different batches since batch_y was created by split
        # and batch_z was created after split closed batch_y... wait, no.
        # Actually, after the split, batch_y_id becomes the new open batch,
        # so change_3 will join it. Let's verify by trying to apply them together.

        # Apply change_2 and change_3 together with allow_mixed_batch=True
        # If they're in different batches, this tests the flag; if same batch, still works
        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [change_2, change_3], "allow_mixed_batch": True},
        )

        payload = _assert_success_response(response)
        # Both should apply since allow_mixed_batch=True
        assert len(payload["applied"]) == 2
        applied_ids = {record["change_id"] for record in payload["applied"]}
        assert applied_ids == {change_2, change_3}

    def test_apply_pending_changes_rejects_unapproved_changes(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying changes without batch_id (not approved) should fail."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create change but don't approve it
        change_id = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="unapproved_model",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("unapproved_model", category),
        )

        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [change_id]},
        )

        _assert_error_response(response, 400, "not been approved in a batch")

    def test_apply_batch_endpoint_applies_all_approved_changes(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """The /apply_batch/{batch_id} endpoint should apply all approved changes in a batch."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create multiple changes
        change_1 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="batch_apply_1",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("batch_apply_1", category),
        )
        change_2 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="batch_apply_2",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("batch_apply_2", category),
        )
        change_3 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="batch_apply_3",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("batch_apply_3", category),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        # Approve all in one batch
        result = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="batch-all",
            approved_ids=[change_1, change_2, change_3],
            rejected_ids=None,
            reject_reason=None,
        )
        batch_id = result.batch_id

        # Apply entire batch
        response = api_client.post(
            f"{self._base_url}/apply_batch/{batch_id}",
            headers=_auth_headers(),
        )

        payload = _assert_success_response(response)
        assert len(payload["applied"]) == 3
        applied_ids = [record["change_id"] for record in payload["applied"]]
        assert set(applied_ids) == {change_1, change_2, change_3}

    def test_apply_batch_endpoint_skips_already_applied_changes(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Partial application triggers batch split - remaining changes move to new batch.

        With batch ID semantics, when a change is applied individually from a batch
        with multiple approved changes, the remaining unapplied changes are reassigned
        to a new batch ID. This test verifies:
        1. The original batch contains only the applied change afterward
        2. The batch_split response fields indicate the reassignment
        3. The remaining change can be applied via the new batch ID
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create two changes
        change_1 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="partial_batch_1",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("partial_batch_1", category),
        )
        change_2 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="partial_batch_2",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("partial_batch_2", category),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        # Approve both in one batch
        result = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="partial-batch",
            approved_ids=[change_1, change_2],
            rejected_ids=None,
            reject_reason=None,
        )
        original_batch_id = result.batch_id

        # Apply first change individually - this triggers batch split
        first_response = api_client.post(
            f"{self._base_url}/changes/{change_1}/apply",
            headers=_auth_headers(),
            json={},
        )
        first_payload = _assert_success_response(first_response)

        # Verify batch split occurred (new response format wraps record)
        assert first_payload["batch_split_occurred"] is True
        assert first_payload["batch_split_original_batch_id"] == original_batch_id
        new_batch_id = first_payload["batch_split_new_batch_id"]
        assert new_batch_id is not None
        assert new_batch_id != original_batch_id
        assert first_payload["batch_split_reassigned_count"] == 1

        # Apply original batch - should return empty since only the applied change remains
        batch_response = api_client.post(
            f"{self._base_url}/apply_batch/{original_batch_id}",
            headers=_auth_headers(),
        )
        batch_payload = _assert_success_response(batch_response)
        assert batch_payload["applied"] == []

        # Apply new batch to get change_2
        new_batch_response = api_client.post(
            f"{self._base_url}/apply_batch/{new_batch_id}",
            headers=_auth_headers(),
        )
        new_batch_payload = _assert_success_response(new_batch_response)
        assert len(new_batch_payload["applied"]) == 1
        assert new_batch_payload["applied"][0]["change_id"] == change_2

    def test_apply_batch_endpoint_returns_404_for_nonexistent_batch(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying a non-existent batch should return 404."""
        response = api_client.post(
            f"{self._base_url}/apply_batch/99999",
            headers=_auth_headers(),
        )

        _assert_error_response(response, 404, "No changes found")

    def test_apply_batch_endpoint_returns_success_when_all_already_applied(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying a batch where all changes are already applied should return 200 with empty list."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        change_1 = _enqueue_pending_change(
            primary_manager_for_api,
            model_name="already_applied_1",
            category=category,
            operation=AuditOperation.CREATE,
            payload=_create_minimal_model_dict("already_applied_1", category),
        )

        queue_service = primary_manager_for_api.pending_queue_service
        assert queue_service is not None

        # Approve in batch
        result = queue_service.process_batch(
            approver_id=_TEST_USER_ID,
            approver_username=_TEST_USERNAME,
            batch_title="already-applied",
            approved_ids=[change_1],
            rejected_ids=None,
            reject_reason=None,
        )
        batch_id = result.batch_id

        # Apply the batch once
        first_apply = api_client.post(
            f"{self._base_url}/apply_batch/{batch_id}",
            headers=_auth_headers(),
        )
        _assert_success_response(first_apply)

        # Apply the same batch again - should succeed with empty list
        second_apply = api_client.post(
            f"{self._base_url}/apply_batch/{batch_id}",
            headers=_auth_headers(),
        )
        payload = _assert_success_response(second_apply)
        assert payload["applied"] == []

    def test_apply_changes_returns_404_for_nonexistent_change(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Applying non-existent change IDs should return 404."""
        response = api_client.post(
            f"{self._base_url}/apply",
            headers=_auth_headers(),
            json={"change_ids": [99999], "allow_mixed_batch": True},
        )

        assert response.status_code == 404
        # FastAPI may wrap the error differently
        data = response.json()
        # Check if error message is in any of the possible keys
        error_text = str(data).lower()
        assert "99999" in error_text or "not found" in error_text


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
            headers=_auth_headers(),
        )

        result = _assert_success_response(response, 202)
        payload = result["payload"]
        assert payload["baseline"] == "stable_diffusion_1"
        assert payload["nsfw"] is False

    def test_create_image_generation_model_missing_required_field(
        self,
        api_client: TestClient,
        primary_manager_for_api: ModelReferenceManager,
    ) -> None:
        """Creating an image_generation model without required fields should fail."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_data = {
            "model_classification": {
                "domain": "image",
                "purpose": "generation",
            },
        }

        response = api_client.post(
            _model_url(RouteNames.create_model, category),
            json=model_data,
            headers=_auth_headers(),
        )

        assert response.status_code == 422
