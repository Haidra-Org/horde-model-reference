"""Tests for v1 API (legacy format) read-only operations."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, ReplicateMode
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.service.app import app
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v1_prefix
from horde_model_reference.service.v1.routers.references import (
    get_model_reference_manager,
)


@pytest.fixture
def api_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def primary_manager_for_v1_api(
    primary_base: Path,
    restore_manager_singleton: None,
) -> Iterator[ModelReferenceManager]:
    """Create a PRIMARY mode manager with legacy data for v1 API tests."""
    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    app.dependency_overrides[get_model_reference_manager] = lambda: manager

    yield manager

    app.dependency_overrides.clear()


def _create_legacy_json_file(base_path: Path, category: MODEL_REFERENCE_CATEGORY, data: dict[str, Any]) -> None:
    """Create a legacy format JSON file for testing.

    Args:
        base_path: Base path for the test
        category: Model category
        data: Legacy format data to write
    """
    legacy_path = base_path / "legacy"
    legacy_path.mkdir(parents=True, exist_ok=True)

    if category == MODEL_REFERENCE_CATEGORY.image_generation:
        filename = "stable_diffusion.json"
    elif category == MODEL_REFERENCE_CATEGORY.text_generation:
        filename = "text_generation.json"
    else:
        filename = f"{category.value}.json"

    file_path = legacy_path / filename
    file_path.write_text(json.dumps(data, indent=2))


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
        primary_manager_for_v1_api: ModelReferenceManager,
    ) -> None:
        """GET /info should return API information."""
        response = _make_v1_get_request(api_client, RouteNames.get_reference_info)
        data = _assert_success_response(response)
        assert "message" in data
        assert "legacy" in data["message"].lower()

    def test_get_info_mentions_github_repos(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
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
        primary_manager_for_v1_api: ModelReferenceManager,
    ) -> None:
        """GET /model_categories should return list of category names."""
        response = _make_v1_get_request(api_client, RouteNames.get_reference_names)
        data = _assert_success_response(response)
        assert isinstance(data, list)

    def test_get_model_categories_returns_valid_categories(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
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

    def test_get_legacy_reference_success(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /{category} should return legacy JSON for a category."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
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
        primary_manager_for_v1_api: ModelReferenceManager,
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
        primary_manager_for_v1_api: ModelReferenceManager,
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
        primary_manager_for_v1_api: ModelReferenceManager,
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

    def test_get_legacy_reference_not_found(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
    ) -> None:
        """GET /{category} should return 404 when category not found."""
        response = _get_category_url(api_client, MODEL_REFERENCE_CATEGORY.miscellaneous.value)
        _assert_not_found_response(response)

    def test_get_legacy_reference_empty_category(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /{category} should return 404 when category is empty."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        _create_legacy_json_file(primary_base, category, {})

        response = _get_category_url(api_client, category.value)
        _assert_not_found_response(response, "empty")

    def test_get_legacy_reference_returns_raw_json(
        self,
        api_client: TestClient,
        primary_manager_for_v1_api: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """GET /{category} should return raw JSON string (not parsed models)."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
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
        primary_manager_for_v1_api: ModelReferenceManager,
    ) -> None:
        """GET /{category} should return 422 for invalid category names."""
        response = _get_category_url(api_client, "invalid_category_name")
        _assert_validation_error(response)
