"""Tests for v1 API (legacy format) read-only operations."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    CanonicalFormat,
    ModelReferenceManager,
    horde_model_reference_paths,
    horde_model_reference_settings,
)
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.pending_queue.models import PendingChangeStatus
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v1_prefix

from ..helpers import ALL_MODEL_CATEGORIES

# Note: The v1_canonical_manager fixture is now defined in conftest.py
# It provides a PRIMARY mode manager with canonical_format='LEGACY' for v1 API tests


def _create_legacy_json_file(base_path: Path, category: MODEL_REFERENCE_CATEGORY, data: dict[str, Any]) -> None:
    """Create a legacy format JSON file for testing.

    Args:
        base_path: Base path for the test
        category: Model category
        data: Legacy format data to write
    """
    # Use the canonical path helper to get the base legacy path
    legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
        category,
        base_path=base_path,
    )

    # For text_generation, the path returns models.csv but we're writing JSON,
    # so use text_generation.json instead (the backend's fallback for JSON format)
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        legacy_file_path = legacy_file_path.with_name("text_generation.json")

    legacy_file_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_file_path.write_text(json.dumps(data, indent=2))


def _create_legacy_model_payload(
    name: str,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    description: str | None = None,
) -> dict[str, Any]:
    """Create a minimal legacy model payload for CRUD tests.

    Args:
        name: Model name
        category: Model category (determines schema)
        description: Optional description

    Returns:
        Category-appropriate legacy model payload
    """
    payload: dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "config": {
            "files": [
                {
                    "path": f"{name}.ckpt",
                    "sha256sum": "a" * 64,  # Valid 64-char hash
                },
            ],
            "download": [
                {
                    "file_name": f"{name}.ckpt",
                    "file_url": f"https://example.com/{name}.ckpt",
                    "file_path": "",
                },
            ],
        },
    }

    if description is not None:
        payload["description"] = description

    if category == MODEL_REFERENCE_CATEGORY.image_generation:
        payload["type"] = "ckpt"
        payload["baseline"] = "stable diffusion 1"
        payload["inpainting"] = False
    elif category == MODEL_REFERENCE_CATEGORY.text_generation:
        payload["parameters"] = 7000000000
    elif category == MODEL_REFERENCE_CATEGORY.clip:
        payload["pretrained_name"] = f"{name}_pretrained"
    elif category == MODEL_REFERENCE_CATEGORY.controlnet:
        payload["type"] = "control_canny"
    elif category == MODEL_REFERENCE_CATEGORY.miscellaneous:
        payload["type"] = "layer_diffuse"

    return payload


_V1_PENDING_QUEUE_BASE = f"{v1_prefix}/pending_queue"
_V1_QUEUE_USER_ID = "test-user-id"
_V1_QUEUE_USERNAME = f"tester#{_V1_QUEUE_USER_ID}"


def _queue_auth_headers() -> dict[str, str]:
    """Return headers accepted by queue approver/requestor auth mocks."""
    return {"apikey": "test_key"}


def _enqueue_legacy_pending_change(
    manager: ModelReferenceManager,
    *,
    model_name: str,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous,
    operation: AuditOperation = AuditOperation.CREATE,
    payload: dict[str, Any] | None = None,
) -> int:
    """Enqueue a pending change using the manager's queue service."""
    queue_service = manager.pending_queue_service
    assert queue_service is not None, "Pending queue must be enabled for tests"

    effective_payload = payload
    if effective_payload is None and operation is not AuditOperation.DELETE:
        effective_payload = _create_legacy_model_payload(model_name, category)

    record = queue_service.enqueue_change(
        category=category,
        model_name=model_name,
        operation=operation,
        payload=effective_payload,
        requestor_id=_V1_QUEUE_USER_ID,
        requestor_username=_V1_QUEUE_USERNAME,
        notes=None,
        request_metadata={"source": "v1-tests"},
    )
    return record.change_id


def _get_create_route_for_category(category: MODEL_REFERENCE_CATEGORY) -> RouteNames | None:
    """Get the create route name for a category, or None if not supported."""
    category_route_map = {
        MODEL_REFERENCE_CATEGORY.image_generation: RouteNames.image_generation_model,
        MODEL_REFERENCE_CATEGORY.text_generation: RouteNames.text_generation_model,
        MODEL_REFERENCE_CATEGORY.blip: RouteNames.blip_model,
        MODEL_REFERENCE_CATEGORY.clip: RouteNames.clip_model,
        MODEL_REFERENCE_CATEGORY.codeformer: RouteNames.codeformer_model,
        MODEL_REFERENCE_CATEGORY.controlnet: RouteNames.controlnet_model,
        MODEL_REFERENCE_CATEGORY.esrgan: RouteNames.esrgan_model,
        MODEL_REFERENCE_CATEGORY.gfpgan: RouteNames.gfpgan_model,
        MODEL_REFERENCE_CATEGORY.safety_checker: RouteNames.safety_checker_model,
        MODEL_REFERENCE_CATEGORY.miscellaneous: RouteNames.miscellaneous_model,
    }
    return category_route_map.get(category)


def _legacy_model_url(route_name: RouteNames, category: MODEL_REFERENCE_CATEGORY, model_name: str) -> str:
    """Compose a v1 API URL for CRUD endpoints."""
    return route_registry.url_for(
        route_name,
        {
            PathVariables.model_category_name: category.value,
            PathVariables.model_name: model_name,
        },
        v1_prefix,
    )


def _read_legacy_model_file(base_path: Path, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any]:
    """Read and return the legacy JSON file for a category."""
    legacy_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
        category,
        base_path=base_path,
    )

    # For text_generation, the path returns models.csv but we're reading JSON,
    # so use text_generation.json instead (matches _create_legacy_json_file)
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        legacy_path = legacy_path.with_name("text_generation.json")

    with open(legacy_path, encoding="utf-8") as legacy_file:
        return cast(dict[str, Any], json.load(legacy_file))


def _make_v1_get_request(
    api_client: TestClient,
    route_name: RouteNames,
    path_variables: dict[str, str] | None = None,
) -> Response:
    """Make a GET request to a v1 API endpoint.

    Args:
        api_client: FastAPI test client
        route_name: Route name from RouteNames enum
        path_variables: Optional path variables dict

    Returns:
        Response from the API
    """
    url = route_registry.url_for(route_name, path_variables or {}, v1_prefix)
    return api_client.get(url)


def _assert_success_response(response: Response) -> dict[str, Any]:
    """Assert response is successful and return parsed JSON.

    Args:
        response: Response to check

    Returns:
        Parsed JSON data from response
    """
    assert response.status_code == 200
    json_data: dict[str, Any] = response.json()
    return json_data


def _assert_not_found_response(response: Response, expected_substring: str = "not found") -> None:
    """Assert response is 404 with expected error message.

    Args:
        response: Response to check
        expected_substring: Expected substring in error detail (case-insensitive)
    """
    assert response.status_code == 404
    data = response.json()
    assert expected_substring.lower() in data["detail"].lower()


def _assert_validation_error(response: Response) -> None:
    """Assert response is a 422 validation error.

    Args:
        response: Response to check
    """
    assert response.status_code == 422


def _get_category_url(
    api_client: TestClient,
    category_name: str,
) -> Response:
    """Make a GET request to retrieve a category by name.

    Args:
        api_client: FastAPI test client
        category_name: Category name (can be enum value or legacy name)

    Returns:
        Response from the API
    """
    return _make_v1_get_request(
        api_client,
        RouteNames.get_reference_by_category,
        {PathVariables.model_category_name: category_name},
    )


class TestGetInfo:
    """Tests for GET /info endpoint."""

    def test_get_info_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """GET /info should return API information."""
        response = _make_v1_get_request(api_client, RouteNames.get_reference_info)
        data = _assert_success_response(response)
        assert "message" in data
        assert "legacy" in data["message"].lower()

    def test_get_info_mentions_github_repos(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """GET /info should mention the GitHub repos."""
        response = _make_v1_get_request(api_client, RouteNames.get_reference_info)
        data = _assert_success_response(response)
        message = data["message"].lower()
        assert "github" in message or "haidra-org" in message


class TestGetModelCategories:
    """Tests for GET /model_categories endpoint."""

    def test_get_model_categories_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """GET /model_categories should return list of category names."""
        response = _make_v1_get_request(api_client, RouteNames.get_reference_names)
        data = _assert_success_response(response)
        assert isinstance(data, list)

    def test_get_model_categories_returns_valid_categories(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /model_categories should return valid category names."""
        _create_legacy_json_file(
            primary_base,
            MODEL_REFERENCE_CATEGORY.miscellaneous,
            {"test_model": {"name": "test_model"}},
        )

        response = _make_v1_get_request(api_client, RouteNames.get_reference_names)
        data = _assert_success_response(response)
        assert isinstance(data, list)
        for category in data:
            assert isinstance(category, str)


class TestGetLegacyReference:
    """Tests for GET /{model_category_name} endpoint."""

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_get_legacy_reference_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """GET /{category} should return legacy JSON for a category."""
        test_data = {
            "model_1": {"name": "model_1", "description": "Test model 1"},
            "model_2": {"name": "model_2", "description": "Test model 2"},
        }
        _create_legacy_json_file(primary_base, category, test_data)

        response = _get_category_url(api_client, category.value)
        assert response.headers["content-type"] == "application/json"

        data = _assert_success_response(response)
        assert "model_1" in data
        assert "model_2" in data
        assert data["model_1"]["description"] == "Test model 1"

    def test_get_legacy_reference_stable_diffusion_mapping(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /stable_diffusion should map to image_generation category."""
        test_data = {
            "test_sd_model": {"name": "test_sd_model", "baseline": "stable_diffusion_1"},
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.image_generation, test_data)

        response = _get_category_url(api_client, "stable_diffusion")
        data = _assert_success_response(response)
        assert "test_sd_model" in data

    def test_get_legacy_reference_db_json_mapping(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /db.json should map to text_generation category."""
        test_data = {
            "test_text_model": {"name": "test_text_model", "parameters": 7000000000},
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        response = _get_category_url(api_client, "db.json")
        data = _assert_success_response(response)
        assert "test_text_model" in data

    def test_get_legacy_reference_json_suffix_stripping(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /{category}.json should strip .json suffix."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        test_data = {
            "test_model": {"name": "test_model"},
        }
        _create_legacy_json_file(primary_base, category, test_data)

        response = _get_category_url(api_client, f"{category.value}.json")
        data = _assert_success_response(response)
        assert "test_model" in data

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_get_legacy_reference_not_found(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """GET /{category} should return 404 when category not found."""
        response = _get_category_url(api_client, category.value)
        _assert_not_found_response(response)

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_get_legacy_reference_empty_category(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """GET /{category} should return 404 when category is empty."""
        _create_legacy_json_file(primary_base, category, {})

        response = _get_category_url(api_client, category.value)
        _assert_not_found_response(response, "empty")

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_get_legacy_reference_returns_raw_json(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """GET /{category} should return raw JSON string (not parsed models)."""
        test_data = {
            "test_model": {
                "name": "test_model",
                "custom_field": "custom_value",
                "nested": {"key": "value"},
            },
        }
        _create_legacy_json_file(primary_base, category, test_data)

        response = _get_category_url(api_client, category.value)
        data = _assert_success_response(response)
        assert "test_model" in data
        assert data["test_model"]["custom_field"] == "custom_value"
        assert data["test_model"]["nested"]["key"] == "value"

    def test_get_legacy_reference_invalid_category(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """GET /{category} should return 422 for invalid category names."""
        response = _get_category_url(api_client, "invalid_category_name")
        _assert_validation_error(response)


class TestCreateLegacyModel:
    """Tests for POST /{category}/create_model endpoint.

    When pending queue is enabled, create operations return 202 Accepted with a
    PendingChangeRecord. The model is not written to disk until the change is
    approved and applied.
    """

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_create_model_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """POST should enqueue a new legacy model creation and return 202."""
        route_name = _get_create_route_for_category(category)
        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 create endpoint")

        model_name = "new_legacy_model"
        payload = _create_legacy_model_payload(model_name, category, description="Created via POST")

        url = route_registry.url_for(route_name, {}, v1_prefix)

        response = api_client.post(url, json=payload, headers={"apikey": "test_key"})

        assert response.status_code == 202

        response_json = response.json()
        # With pending queue enabled, response is a PendingChangeRecord
        assert "change_id" in response_json
        assert response_json["model_name"] == model_name
        assert response_json["category"] == category.value
        assert response_json["operation"] == AuditOperation.CREATE.value
        assert response_json["status"] == PendingChangeStatus.PENDING.value

        # Verify the change is in the queue, not written to disk yet
        queue_service = v1_canonical_manager.pending_queue_service
        assert queue_service is not None
        change = queue_service.get_change(response_json["change_id"])
        assert change is not None
        assert change.model_name == model_name
        assert change.payload is not None
        assert change.payload["description"] == "Created via POST"

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_create_model_conflict(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """POST should return 409 when model already exists."""
        route_name = _get_create_route_for_category(category)
        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 create endpoint")

        model_name = "existing_model"
        existing_payload = _create_legacy_model_payload(model_name, category)
        _create_legacy_json_file(primary_base, category, {model_name: existing_payload})

        url = route_registry.url_for(route_name, {}, v1_prefix)

        response = api_client.post(url, json=existing_payload, headers={"apikey": "test_key"})

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()


class TestLegacyFormatWriteRestriction:
    """Tests that write operations require legacy canonical format.

    Note: These tests verify the conditional import behavior. In the test environment,
    canonical_format is set to 'LEGACY' at import time so routes are registered.
    In production with canonical_format='v2', the v1 CRUD routes would not be registered at all.
    """

    def test_backend_supports_legacy_writes_in_legacy_mode(
        self,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """Backend should support legacy writes when canonical_format='LEGACY'."""
        assert horde_model_reference_settings.canonical_format == "LEGACY"
        assert v1_canonical_manager.backend.supports_legacy_writes() is True

    def test_backend_rejects_legacy_writes_in_v2_mode(
        self,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """Backend should reject legacy writes when canonical_format='v2'."""
        previous_format = horde_model_reference_settings.canonical_format
        try:
            horde_model_reference_settings.canonical_format = "v2"
            assert v1_canonical_manager.backend.supports_legacy_writes() is False
        finally:
            horde_model_reference_settings.canonical_format = previous_format


class TestUpdateLegacyModel:
    """Tests for PUT /{category} endpoint.

    When pending queue is enabled, update operations return 202 Accepted with a
    PendingChangeRecord. The model is not updated on disk until the change is
    approved and applied.
    """

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_update_existing_model(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """PUT should enqueue an update for an existing legacy model and return 202."""
        route_name = _get_create_route_for_category(category)
        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 update endpoint")

        model_name = "update_me"
        original_payload = _create_legacy_model_payload(model_name, category, description="Original")
        _create_legacy_json_file(primary_base, category, {model_name: original_payload})

        updated_payload = _create_legacy_model_payload(model_name, category, description="Updated")

        url = route_registry.url_for(route_name, {}, v1_prefix)

        response = api_client.put(url, json=updated_payload, headers={"apikey": "test_key"})

        assert response.status_code == 202
        response_json = response.json()

        # With pending queue enabled, response is a PendingChangeRecord
        assert "change_id" in response_json
        assert response_json["model_name"] == model_name
        assert response_json["category"] == category.value
        assert response_json["operation"] == AuditOperation.UPDATE.value
        assert response_json["status"] == PendingChangeStatus.PENDING.value

        # Verify the change is in the queue with the updated payload
        queue_service = v1_canonical_manager.pending_queue_service
        assert queue_service is not None
        change = queue_service.get_change(response_json["change_id"])
        assert change is not None
        assert change.payload is not None
        assert change.payload["description"] == "Updated"

        # Original file should still have the old value (not yet applied)
        legacy_data = _read_legacy_model_file(primary_base, category)
        assert legacy_data[model_name]["description"] == "Original"


class TestDeleteLegacyModel:
    """Tests for DELETE /{category}/{model_name} endpoint.

    When pending queue is enabled, delete operations return 202 Accepted with a
    PendingChangeRecord. The model is not deleted from disk until the change is
    approved and applied.
    """

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should enqueue a deletion and return 202."""
        model_name = "delete_me"
        payload = _create_legacy_model_payload(model_name, category)
        _create_legacy_json_file(primary_base, category, {model_name: payload})

        response = api_client.delete(
            _legacy_model_url(RouteNames.delete_model, category, model_name), headers={"apikey": "test_key"}
        )

        assert response.status_code == 202

        response_json = response.json()
        # With pending queue enabled, response is a PendingChangeRecord
        assert "change_id" in response_json
        assert response_json["model_name"] == model_name
        assert response_json["category"] == category.value
        assert response_json["operation"] == AuditOperation.DELETE.value
        assert response_json["status"] == PendingChangeStatus.PENDING.value

        # Verify the change is in the queue
        queue_service = v1_canonical_manager.pending_queue_service
        assert queue_service is not None
        change = queue_service.get_change(response_json["change_id"])
        assert change is not None
        assert change.model_name == model_name
        assert change.operation == AuditOperation.DELETE

        # Model should still exist (not yet deleted)
        legacy_data = _read_legacy_model_file(primary_base, category)
        assert model_name in legacy_data

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_not_found(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should return 404 when the model is absent."""
        model_name = "missing_model"
        _create_legacy_json_file(primary_base, category, {})

        response = api_client.delete(
            _legacy_model_url(RouteNames.delete_model, category, model_name), headers={"apikey": "test_key"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestLegacyPendingQueueAdmin:
    """Tests for pending queue management endpoints exposed under /v1."""

    _base_url = _V1_PENDING_QUEUE_BASE

    def test_list_pending_changes_includes_enqueued_records(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        mock_auth_success: None,
    ) -> None:
        """GET /pending_queue/changes should surface legacy queued updates."""
        change_id = _enqueue_legacy_pending_change(
            v1_canonical_manager,
            model_name="legacy_queue_list",
        )

        response = api_client.get(f"{self._base_url}/changes", headers=_queue_auth_headers())
        payload = _assert_success_response(response)
        returned_ids = {item["change_id"] for item in payload["items"]}
        assert change_id in returned_ids

    def test_apply_pending_change_updates_legacy_file(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        mock_auth_success: None,
    ) -> None:
        """POST /pending_queue/changes/{id}/apply should write to legacy JSON."""
        category = MODEL_REFERENCE_CATEGORY.image_generation
        model_name = "legacy_queue_apply"
        original_payload = _create_legacy_model_payload(model_name, category, description="old")
        updated_payload = _create_legacy_model_payload(model_name, category, description="new")
        _create_legacy_json_file(primary_base, category, {model_name: original_payload})

        change_id = _enqueue_legacy_pending_change(
            v1_canonical_manager,
            model_name=model_name,
            category=category,
            operation=AuditOperation.UPDATE,
            payload=updated_payload,
        )
        queue_service = v1_canonical_manager.pending_queue_service
        assert queue_service is not None
        queue_service.process_batch(
            approver_id=_V1_QUEUE_USER_ID,
            approver_username=_V1_QUEUE_USERNAME,
            batch_title="apply legacy",
            approved_ids=[change_id],
            rejected_ids=None,
            reject_reason=None,
        )

        response = api_client.post(
            f"{self._base_url}/changes/{change_id}/apply",
            headers=_queue_auth_headers(),
            json={"job_id": "legacy-job"},
        )
        payload = _assert_success_response(response)
        assert payload["record"]["status"] == PendingChangeStatus.APPLIED.value
        assert payload["record"]["applied_job_id"] == "legacy-job"

        legacy_data = _read_legacy_model_file(primary_base, category)
        assert legacy_data[model_name]["description"] == "new"

    def test_pending_queue_works_in_v2_mode(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        mock_auth_success: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pending queue should work in v2 mode - both legacy and v2 support enqueued changes."""
        _enqueue_legacy_pending_change(v1_canonical_manager, model_name="v2_queue_test")
        monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)

        response = api_client.get(f"{self._base_url}/changes", headers=_queue_auth_headers())
        # Both legacy and v2 modes support pending queue
        assert response.status_code == 200

    @pytest.mark.parametrize("category", ALL_MODEL_CATEGORIES)
    def test_delete_model_category_missing(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        legacy_canonical_mode: None,
        mock_auth_success: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """DELETE should return 404 when the category file is missing."""
        model_name = "missing_category"

        response = api_client.delete(
            _legacy_model_url(RouteNames.delete_model, category, model_name), headers={"apikey": "test_key"}
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestTextGenerationEndpoint:
    """Tests for GET /text_generation endpoint with optional include_group parameter."""

    def test_get_text_generation_success(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should return text generation models."""
        test_data = {
            "model_1": {
                "name": "model_1",
                "parameters": 7000000000,
                "text_model_group": "base_model_1",
            },
            "model_2": {
                "name": "model_2",
                "parameters": 13000000000,
                "text_model_group": "base_model_2",
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        data = _assert_success_response(response)

        assert "model_1" in data
        assert "model_2" in data

    def test_get_text_generation_excludes_group_by_default(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should not include text_model_group by default (legacy format)."""
        test_data = {
            "model_without_group": {
                "name": "model_without_group",
                "parameters": 7000000000,
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        data = _assert_success_response(response)

        assert "model_without_group" in data
        # Legacy format doesn't include text_model_group
        assert "text_model_group" not in data["model_without_group"]

    def test_get_text_generation_includes_group_when_requested(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation?include_group=true should compute and add text_model_group."""
        test_data = {
            "test-model-8B-v1": {
                "name": "test-model-8B-v1",
                "parameters": 8000000000,
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        url = route_registry.url_for(RouteNames.get_text_generation_reference, {}, v1_prefix)
        response = api_client.get(f"{url}?include_group=true")
        data = _assert_success_response(response)

        assert "test-model-8B-v1" in data
        assert "text_model_group" in data["test-model-8B-v1"]
        # The base name should be computed from the model name (removing size and version)
        assert data["test-model-8B-v1"]["text_model_group"] == "test-model-v1"

    def test_get_text_generation_explicit_false(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation?include_group=false should not include text_model_group."""
        test_data = {
            "model_name": {
                "name": "model_name",
                "parameters": 7000000000,
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        url = route_registry.url_for(RouteNames.get_text_generation_reference, {}, v1_prefix)
        response = api_client.get(f"{url}?include_group=false")
        data = _assert_success_response(response)

        assert "model_name" in data
        assert "text_model_group" not in data["model_name"]

    def test_get_text_generation_preserves_other_fields(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should preserve all other fields when adding text_model_group."""
        test_data = {
            "test-model-7B": {
                "name": "test-model-7B",
                "parameters": 7000000000,
                "description": "Test description",
                "url": "https://example.com/model",
                "custom_field": "custom_value",
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        url = route_registry.url_for(RouteNames.get_text_generation_reference, {}, v1_prefix)
        response = api_client.get(f"{url}?include_group=true")
        data = _assert_success_response(response)

        model = data["test-model-7B"]
        assert model["name"] == "test-model-7B"
        assert model["parameters"] == 7000000000
        assert model["description"] == "Test description"
        assert model["url"] == "https://example.com/model"
        assert model["custom_field"] == "custom_value"
        # text_model_group should be computed and added
        assert "text_model_group" in model
        assert model["text_model_group"] == "test-model"

    def test_get_text_generation_handles_missing_field(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should handle models without text_model_group."""
        test_data = {
            "model_without_group": {
                "name": "model_without_group",
                "parameters": 7000000000,
            },
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        data = _assert_success_response(response)

        assert "model_without_group" in data
        assert "text_model_group" not in data["model_without_group"]

    def test_get_text_generation_not_found(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """GET /text_generation should return 404 when no models exist."""
        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        _assert_not_found_response(response)

    def test_get_text_generation_empty(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should return 404 when category is empty."""
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, {})

        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        _assert_not_found_response(response, "empty")

    def test_get_text_generation_returns_json_content_type(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /text_generation should return application/json content type."""
        test_data = {
            "model_1": {"name": "model_1", "parameters": 7000000000},
        }
        _create_legacy_json_file(primary_base, MODEL_REFERENCE_CATEGORY.text_generation, test_data)

        response = _make_v1_get_request(api_client, RouteNames.get_text_generation_reference)
        assert response.headers["content-type"] == "application/json"


class TestRouteConditionalImport:
    """Tests for conditional import of CRUD routes.

    Note: In the test environment, canonical_format is set to 'LEGACY' at import time,
    so v1 CRUD routes ARE registered. In a production deployment with canonical_format='v2',
    these routes would not be registered at all (conditional import in references.py:143-144).

    These tests verify that read-only routes always work and that the conditional import
    logic exists in the codebase.
    """

    def test_read_routes_always_available(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """Read-only routes should be available regardless of canonical_format."""
        info_response = _make_v1_get_request(api_client, RouteNames.get_reference_info)
        assert info_response.status_code == 200

        categories_response = _make_v1_get_request(api_client, RouteNames.get_reference_names)
        assert categories_response.status_code == 200

    def test_crud_routes_registered_in_legacy_mode(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        mock_auth_success: None,
    ) -> None:
        """CRUD routes should be registered when canonical_format='LEGACY' at import time."""
        assert horde_model_reference_settings.canonical_format == "LEGACY"

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        route_name = _get_create_route_for_category(category)

        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 create endpoint")

        model_name = "test_model"
        payload = _create_legacy_model_payload(model_name, category)

        url = route_registry.url_for(route_name, {}, v1_prefix)
        response = api_client.post(url, json=payload, headers={"apikey": "test_key"})

        assert response.status_code in (201, 202, 409, 422, 500)
