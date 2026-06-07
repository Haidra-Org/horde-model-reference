"""Tests for Phase 4 API design corrections.

Covers:
- API-1/IV-3: Invalid category -> 422 (not 404)
- API-5: POST creates include Location header
- API-6: Direct DELETE returns 204 No Content
- IV-2: Model name validation (empty, whitespace, path separators)
- API-3: SearchResponse includes has_more field
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    CanonicalFormat,
    ModelReferenceManager,
    PrefetchStrategy,
    ReplicateMode,
    horde_model_reference_paths,
    horde_model_reference_settings,
)
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.service.shared import (
    PathVariables,
    RouteNames,
    get_model_reference_manager,
    route_registry,
    v1_prefix,
    validate_model_name,
)

_V2 = "/model_references/v2"

pytestmark = pytest.mark.usefixtures("mock_auth_success")


# - Helpers ------------------------


def _v1_model_url(route_name: RouteNames, category: MODEL_REFERENCE_CATEGORY, model_name: str) -> str:
    return route_registry.url_for(
        route_name,
        {
            PathVariables.model_category_name: category.value,
            PathVariables.model_name: model_name,
        },
        v1_prefix,
    )


def _create_legacy_json_file(base_path: Path, category: MODEL_REFERENCE_CATEGORY, data: dict[str, Any]) -> None:
    legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(category, base_path=base_path)
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        legacy_file_path = legacy_file_path.with_name("text_generation.json")
    legacy_file_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_file_path.write_text(json.dumps(data, indent=2))


def _legacy_payload(name: str, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "version": "1.0",
        "config": {
            "files": [{"path": f"{name}.ckpt", "sha256sum": "a" * 64}],
            "download": [
                {"file_name": f"{name}.ckpt", "file_url": f"https://example.com/{name}.ckpt", "file_path": ""}
            ],
        },
    }
    if category == MODEL_REFERENCE_CATEGORY.image_generation:
        payload.update(type="ckpt", baseline="stable diffusion 1", inpainting=False)
    elif category == MODEL_REFERENCE_CATEGORY.text_generation:
        payload["parameters"] = 7_000_000_000
    elif category == MODEL_REFERENCE_CATEGORY.miscellaneous:
        payload["type"] = "layer_diffuse"
    return payload


def _v2_model_payload(
    name: str,
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous,
) -> dict[str, Any]:
    model_dict: dict[str, Any] = {
        "name": name,
        "record_type": category.value,
        "model_classification": {"domain": "image", "purpose": "miscellaneous"},
    }
    if category == MODEL_REFERENCE_CATEGORY.image_generation:
        model_dict["model_classification"] = {"domain": "image", "purpose": "generation"}
        model_dict["baseline"] = "stable_diffusion_1"
        model_dict["inpainting"] = False
    return model_dict


# - Fixtures ------------------------


@pytest.fixture
def v1_manager_no_queue(
    primary_base: Path,
    restore_manager_singleton: None,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    monkeypatch: pytest.MonkeyPatch,
    mock_auth_success: None,
) -> Iterator[ModelReferenceManager]:
    """PRIMARY manager with no pending queue for direct write tests."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.LEGACY)
    monkeypatch.setattr(horde_model_reference_settings.pending_queue, "enabled", False)

    # auth_against_horde is imported as a local name in v1/routers/shared.py,
    # so the conftest monkeypatch on the source module doesn't reach it.
    from horde_model_reference.service.shared import auth_against_horde as _patched

    monkeypatch.setattr(
        "horde_model_reference.service.v1.routers.shared.auth_against_horde",
        _patched,
    )

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        prefetch_strategy=PrefetchStrategy.LAZY,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    assert manager.pending_queue_service is None, "Queue should be disabled for this test"
    dependency_override(get_model_reference_manager, lambda: manager)
    yield manager


@pytest.fixture
def v2_manager(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """PRIMARY manager with pending queue for v2 API tests."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)
    yield manager


@pytest.fixture
def search_manager(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> ModelReferenceManager:
    """PRIMARY manager seeded with models for search/pagination tests."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)

    for i in range(5):
        manager.backend.update_model(
            MODEL_REFERENCE_CATEGORY.image_generation,
            f"model_{i}",
            {
                "name": f"model_{i}",
                "record_type": "image_generation",
                "model_classification": {"domain": "image", "purpose": "generation"},
                "baseline": "stable_diffusion_1",
                "nsfw": False,
                "inpainting": False,
            },
        )
    return manager


# - IV-2: Model Name Validation ------------------------


class TestModelNameValidation:
    """IV-2: Reject empty, whitespace-only, and path-separator model names."""

    def test_validate_model_name_rejects_empty(self) -> None:
        """Empty string raises 422."""
        with pytest.raises(Exception, match="empty"):
            validate_model_name("")

    def test_validate_model_name_rejects_whitespace_only(self) -> None:
        """Whitespace-only string raises 422."""
        with pytest.raises(Exception, match="empty"):
            validate_model_name("   ")

    def test_validate_model_name_rejects_backslash(self) -> None:
        """Backslash raises 422."""
        with pytest.raises(Exception, match="invalid character"):
            validate_model_name("path\\model")

    def test_validate_model_name_accepts_valid(self) -> None:
        """Normal name passes without error."""
        validate_model_name("valid_model-name.v1")

    @pytest.mark.parametrize("bad_name", ["", "   ", "\t", "c\\d"])
    def test_v2_create_rejects_bad_model_names(
        self,
        api_client: TestClient,
        v2_manager: ModelReferenceManager,
        bad_name: str,
    ) -> None:
        """V2 POST with invalid model name returns 422."""
        payload = _v2_model_payload(bad_name)
        url = f"{_V2}/{MODEL_REFERENCE_CATEGORY.miscellaneous}/add"
        resp = api_client.post(url, json=payload, headers={"apikey": "test_key"})
        assert resp.status_code == 422

    @pytest.mark.parametrize("bad_name", ["", "   ", "c\\d"])
    def test_v2_delete_rejects_bad_model_names(
        self,
        api_client: TestClient,
        v2_manager: ModelReferenceManager,
        bad_name: str,
    ) -> None:
        """V2 DELETE with invalid model name returns an error (not 2xx)."""
        url = f"{_V2}/{MODEL_REFERENCE_CATEGORY.miscellaneous}/{bad_name}"
        resp = api_client.delete(url, headers={"apikey": "test_key"})
        # Empty/slash names may fail at the routing level (404/405) before
        # reaching our validate_model_name check (422). Either way the
        # request must not succeed.
        assert resp.status_code in (404, 405, 422)

    @pytest.mark.parametrize("bad_name", ["", "   "])
    def test_v1_create_rejects_bad_model_names(
        self,
        api_client: TestClient,
        v1_canonical_manager: ModelReferenceManager,
        primary_base: Path,
        bad_name: str,
    ) -> None:
        """V1 POST with invalid model name in payload returns 422."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        _create_legacy_json_file(primary_base, category, {})
        payload = _legacy_payload(bad_name, category)
        url = f"{v1_prefix}/{category.value}"
        resp = api_client.post(url, json=payload, headers={"apikey": "test_key"})
        assert resp.status_code == 422


# - API-1/IV-3: Category Validation -> 422 ------------------------


class TestCategoryValidation:
    """API-1/IV-3: Invalid category names return 422 (not 404)."""

    def test_search_invalid_category_returns_422(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """Search with bogus category returns 422."""
        resp = api_client.get(f"{_V2}/bogus_category/search")
        assert resp.status_code == 422

    def test_popular_invalid_category_returns_422(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """Popular with bogus category returns 422."""
        resp = api_client.get(f"{_V2}/bogus_category/popular")
        assert resp.status_code == 422


# - API-6: DELETE -> 204 ------------------------


class TestDeleteReturns204:
    """API-6: Direct (non-queue) DELETE returns 204 No Content."""

    def test_v1_direct_delete_returns_204(
        self,
        api_client: TestClient,
        v1_manager_no_queue: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """DELETE with queue disabled should return 204 with empty body."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "to_delete"
        payload = _legacy_payload(model_name, category)
        _create_legacy_json_file(primary_base, category, {model_name: payload})

        url = _v1_model_url(RouteNames.delete_model, category, model_name)
        resp = api_client.delete(url, headers={"apikey": "test_key"})

        assert resp.status_code == 204
        assert resp.content == b""


# - API-5: POST -> 201 + Location header ------------------------


class TestPostLocation:
    """API-5: Direct v1 POST creates return 201 with Location header."""

    def test_v1_direct_create_returns_201_with_location(
        self,
        api_client: TestClient,
        v1_manager_no_queue: ModelReferenceManager,
        primary_base: Path,
    ) -> None:
        """POST with queue disabled returns 201 and a Location header."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_name = "new_model"
        _create_legacy_json_file(primary_base, category, {})
        payload = _legacy_payload(model_name, category)

        url = f"{v1_prefix}/{category.value}"
        resp = api_client.post(url, json=payload, headers={"apikey": "test_key"})

        assert resp.status_code == 201
        assert "location" in resp.headers
        location = resp.headers["location"]
        assert category.value in location
        assert model_name in location


# - API-3: has_more in SearchResponse ------------------------


class TestSearchHasMore:
    """API-3: SearchResponse includes has_more boolean."""

    def test_has_more_true_when_more_results(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """has_more is True when offset+limit < total."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"limit": 2, "offset": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["has_more"] is True

    def test_has_more_false_at_end(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """has_more is False when offset+limit >= total."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"limit": 10, "offset": 0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["has_more"] is False

    def test_has_more_false_at_exact_boundary(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """has_more is False when offset+limit == total."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"limit": 3, "offset": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["has_more"] is False

    def test_cross_category_search_has_more(
        self,
        api_client: TestClient,
        search_manager: ModelReferenceManager,
    ) -> None:
        """Cross-category search also includes has_more."""
        resp = api_client.get(f"{_V2}/search", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "has_more" in data
        assert data["has_more"] is True
