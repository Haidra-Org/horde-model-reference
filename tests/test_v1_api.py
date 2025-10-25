"""Tests for v1 API (legacy format) read-only operations."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    horde_model_reference_paths,
    horde_model_reference_settings,
)
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v1_prefix
from tests.helpers import ALL_MODEL_CATEGORIES

# Note: The v1_canonical_manager fixture is now defined in conftest.py
# It provides a PRIMARY mode manager with canonical_format='legacy' for v1 API tests


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
    """Tests for POST /{category}/create_model endpoint."""

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
        """POST should create a new legacy model file entry."""
        route_name = _get_create_route_for_category(category)
        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 create endpoint")

        model_name = "new_legacy_model"
        payload = _create_legacy_model_payload(model_name, category, description="Created via POST")

        url = route_registry.url_for(route_name, {}, v1_prefix)

        response = api_client.post(url, json=payload, headers={"apikey": "test_key"})

        assert response.status_code == 201

        response_json = response.json()
        for key, value in payload.items():
            if not isinstance(value, dict) and not isinstance(value, list):
                assert response_json[key] == value

        legacy_data = _read_legacy_model_file(primary_base, category)
        assert model_name in legacy_data
        assert legacy_data[model_name]["description"] == "Created via POST"

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
    canonical_format is set to 'legacy' at import time so routes are registered.
    In production with canonical_format='v2', the v1 CRUD routes would not be registered at all.
    """

    def test_backend_supports_legacy_writes_in_legacy_mode(
        self,
        v1_canonical_manager: ModelReferenceManager,
    ) -> None:
        """Backend should support legacy writes when canonical_format='legacy'."""
        assert horde_model_reference_settings.canonical_format == "legacy"
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
    """Tests for PUT /{category} endpoint."""

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
        """PUT should update an existing legacy model."""
        route_name = _get_create_route_for_category(category)
        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 update endpoint")

        model_name = "update_me"
        original_payload = _create_legacy_model_payload(model_name, category, description="Original")
        _create_legacy_json_file(primary_base, category, {model_name: original_payload})

        updated_payload = _create_legacy_model_payload(model_name, category, description="Updated")

        url = route_registry.url_for(route_name, {}, v1_prefix)

        response = api_client.put(url, json=updated_payload, headers={"apikey": "test_key"})

        assert response.status_code == 200
        response_json = response.json()

        assert response_json["description"] == "Updated"

        legacy_data = _read_legacy_model_file(primary_base, category)
        assert legacy_data[model_name]["description"] == "Updated"


class TestDeleteLegacyModel:
    """Tests for DELETE /{category}/{model_name} endpoint."""

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
        """DELETE should remove the model from the legacy file."""
        model_name = "delete_me"
        payload = _create_legacy_model_payload(model_name, category)
        _create_legacy_json_file(primary_base, category, {model_name: payload})

        response = api_client.delete(
            _legacy_model_url(RouteNames.delete_model, category, model_name), headers={"apikey": "test_key"}
        )

        assert response.status_code == 200

        legacy_data = _read_legacy_model_file(primary_base, category)
        assert model_name not in legacy_data

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


class TestRouteConditionalImport:
    """Tests for conditional import of CRUD routes.

    Note: In the test environment, canonical_format is set to 'legacy' at import time,
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
        """CRUD routes should be registered when canonical_format='legacy' at import time."""
        assert horde_model_reference_settings.canonical_format == "legacy"

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        route_name = _get_create_route_for_category(category)

        if route_name is None:
            pytest.skip(f"Category {category} does not have a v1 create endpoint")

        model_name = "test_model"
        payload = _create_legacy_model_payload(model_name, category)

        url = route_registry.url_for(route_name, {}, v1_prefix)
        response = api_client.post(url, json=payload, headers={"apikey": "test_key"})

        assert response.status_code in (201, 409, 422, 500)
