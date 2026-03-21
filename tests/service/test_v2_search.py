"""Tests for v2 search and popularity endpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
)
from horde_model_reference.service.shared import get_model_reference_manager

_V2 = "/model_references/v2"


@pytest.fixture
def primary_manager_for_search(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """PRIMARY manager seeded with test data for search tests."""
    from horde_model_reference import horde_model_reference_settings

    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "v2")
    manager = primary_manager_override_factory(get_model_reference_manager)

    backend = manager.backend

    backend.update_model(
        MODEL_REFERENCE_CATEGORY.image_generation,
        "img_safe_sd1",
        {
            "name": "img_safe_sd1",
            "record_type": "image_generation",
            "model_classification": {"domain": "image", "purpose": "generation"},
            "baseline": "stable_diffusion_1",
            "nsfw": False,
            "inpainting": False,
            "tags": ["landscape", "photo"],
        },
    )
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.image_generation,
        "img_nsfw_xl",
        {
            "name": "img_nsfw_xl",
            "record_type": "image_generation",
            "model_classification": {"domain": "image", "purpose": "generation"},
            "baseline": "stable_diffusion_xl",
            "nsfw": True,
            "inpainting": False,
            "tags": ["anime"],
        },
    )
    backend.update_model(
        MODEL_REFERENCE_CATEGORY.image_generation,
        "img_inpaint_sd1",
        {
            "name": "img_inpaint_sd1",
            "record_type": "image_generation",
            "model_classification": {"domain": "image", "purpose": "generation"},
            "baseline": "stable_diffusion_1",
            "nsfw": False,
            "inpainting": True,
            "tags": ["landscape", "anime"],
        },
    )

    backend.update_model(
        MODEL_REFERENCE_CATEGORY.clip,
        "clip_vit",
        {
            "name": "clip_vit",
            "record_type": "clip",
            "model_classification": {"domain": "image", "purpose": "feature_extractor"},
        },
    )

    backend.update_model(
        MODEL_REFERENCE_CATEGORY.miscellaneous,
        "misc_util",
        {
            "name": "misc_util",
            "record_type": "miscellaneous",
            "model_classification": {"domain": "image", "purpose": "miscellaneous"},
        },
    )

    manager._invalidate_cache()
    yield manager


class TestCategorySearch:
    """Tests for the per-category search endpoint."""

    def test_search_basic(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate search returns all models in a category with total count."""
        resp = api_client.get(f"{_V2}/image_generation/search")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 3

    def test_search_nsfw_filter(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate nsfw=false filter excludes NSFW models from results."""
        resp = api_client.get(f"{_V2}/image_generation/search", params={"nsfw": "false"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        names = {r["name"] for r in data["results"]}
        assert "img_nsfw_xl" not in names

    def test_search_baseline_filter(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate baseline filter returns only models matching the specified baseline."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"baseline": "stable_diffusion_xl"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "img_nsfw_xl"

    def test_search_inpainting_filter(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate inpainting=true filter returns only inpainting models."""
        resp = api_client.get(f"{_V2}/image_generation/search", params={"inpainting": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "img_inpaint_sd1"

    def test_search_tags_any(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate tags_any filter returns models having at least one of the specified tags."""
        resp = api_client.get(f"{_V2}/image_generation/search", params={"tags_any": ["anime"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        names = {r["name"] for r in data["results"]}
        assert names == {"img_nsfw_xl", "img_inpaint_sd1"}

    def test_search_tags_all(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate tags_all filter returns only models having all specified tags."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"tags_all": ["landscape", "anime"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "img_inpaint_sd1"

    def test_search_tags_none(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate tags_none filter excludes models having any of the specified tags."""
        resp = api_client.get(f"{_V2}/image_generation/search", params={"tags_none": ["anime"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "img_safe_sd1"

    def test_search_name_contains(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate name_contains filter performs case-insensitive substring matching."""
        resp = api_client.get(f"{_V2}/image_generation/search", params={"name_contains": "NSFW"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["name"] == "img_nsfw_xl"

    def test_search_sort_by(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate sort_by=name returns results in ascending alphabetical order."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"sort_by": "name"},
        )
        assert resp.status_code == 200
        data = resp.json()
        names = [r["name"] for r in data["results"]]
        assert names == sorted(names)

    def test_search_sort_by_invalid(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate sort_by with a nonexistent field returns 400."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"sort_by": "nonexistent_field"},
        )
        assert resp.status_code == 400

    def test_search_pagination(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate limit and offset params paginate results correctly across pages."""
        resp = api_client.get(
            f"{_V2}/image_generation/search",
            params={"limit": 1, "offset": 0, "sort_by": "name"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 1
        assert data["offset"] == 0
        assert data["limit"] == 1
        first_name = data["results"][0]["name"]

        resp2 = api_client.get(
            f"{_V2}/image_generation/search",
            params={"limit": 1, "offset": 1, "sort_by": "name"},
        )
        data2 = resp2.json()
        assert data2["total"] == 3
        assert len(data2["results"]) == 1
        assert data2["results"][0]["name"] != first_name

    def test_search_invalid_category(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate search on a nonexistent category returns 404."""
        resp = api_client.get(f"{_V2}/bogus_category/search")
        assert resp.status_code == 404

    def test_search_unsupported_filter_returns_400(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Clip has no 'nsfw' field — should get 400, not 500."""
        resp = api_client.get(f"{_V2}/clip/search", params={"nsfw": "false"})
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"].lower() or "does not exist" in resp.json()["detail"].lower()


class TestCrossCategorySearch:
    """Tests for the cross-category search endpoint."""

    def test_search_all_basic(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate cross-category search returns models from all categories."""
        resp = api_client.get(f"{_V2}/search")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 5  # 3 image + 1 clip + 1 misc

    def test_search_all_name_contains(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate cross-category name_contains filter works across all categories."""
        resp = api_client.get(f"{_V2}/search", params={"name_contains": "img_"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        names = {r["name"] for r in data["results"]}
        assert names == {"img_safe_sd1", "img_nsfw_xl", "img_inpaint_sd1"}

    def test_search_all_nsfw_filter(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Cross-category nsfw filter should gracefully handle records without the field."""
        resp = api_client.get(f"{_V2}/search", params={"nsfw": "true"})
        assert resp.status_code == 200
        data = resp.json()
        names = {r["name"] for r in data["results"]}
        assert "img_nsfw_xl" in names
        # clip/misc have no nsfw field → excluded, not errored
        assert "clip_vit" not in names
        assert "misc_util" not in names

    def test_search_all_pagination(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate cross-category search respects limit and offset pagination."""
        resp = api_client.get(f"{_V2}/search", params={"limit": 2, "offset": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert data["total"] >= 5


class TestPopularModels:
    """Tests for the popular models endpoint."""

    def test_popular_unsupported_category(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate popular endpoint returns empty list for categories without popularity data."""
        resp = api_client.get(f"{_V2}/clip/popular")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_popular_invalid_category(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate popular endpoint returns 404 for a nonexistent category."""
        resp = api_client.get(f"{_V2}/bogus_category/popular")
        assert resp.status_code == 404

    def test_popular_mocked(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate popular endpoint ranks models by worker count using mocked Horde API data."""
        from horde_model_reference.integrations.horde_api_models import (
            HordeModelStatsResponse,
            HordeModelStatus,
            IndexedHordeModelStats,
            IndexedHordeModelStatus,
            IndexedHordeWorkers,
        )

        mock_status = IndexedHordeModelStatus(
            [
                HordeModelStatus(
                    name="img_safe_sd1", count=10, jobs=0, performance=1.0, eta=0, queued=0, type="image"
                ),
                HordeModelStatus(name="img_nsfw_xl", count=5, jobs=0, performance=1.0, eta=0, queued=0, type="image"),
                HordeModelStatus(
                    name="img_inpaint_sd1", count=1, jobs=0, performance=1.0, eta=0, queued=0, type="image"
                ),
            ]
        )
        mock_stats = IndexedHordeModelStats(HordeModelStatsResponse(day={}, month={}, total={}))
        mock_workers = IndexedHordeWorkers([])

        mock_integration = AsyncMock()
        mock_integration.get_combined_data_indexed = AsyncMock(
            return_value=(mock_status, mock_stats, mock_workers),
        )

        with patch(
            "horde_model_reference.integrations.horde_api_integration.HordeAPIIntegration",
            return_value=mock_integration,
        ):
            resp = api_client.get(f"{_V2}/image_generation/popular")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["name"] == "img_safe_sd1"

    def test_popular_sort_by_usage(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate popular endpoint sorts by daily usage stats when sort_by=usage_day."""
        from horde_model_reference.integrations.horde_api_models import (
            HordeModelStatsResponse,
            HordeModelStatus,
            IndexedHordeModelStats,
            IndexedHordeModelStatus,
            IndexedHordeWorkers,
        )

        mock_status = IndexedHordeModelStatus(
            [
                HordeModelStatus(name="img_safe_sd1", count=1, jobs=0, performance=1.0, eta=0, queued=0, type="image"),
                HordeModelStatus(name="img_nsfw_xl", count=1, jobs=0, performance=1.0, eta=0, queued=0, type="image"),
            ]
        )
        mock_stats = IndexedHordeModelStats(
            HordeModelStatsResponse(
                day={"img_safe_sd1": 5, "img_nsfw_xl": 50},
                month={"img_safe_sd1": 100, "img_nsfw_xl": 200},
                total={"img_safe_sd1": 1000, "img_nsfw_xl": 2000},
            )
        )
        mock_workers = IndexedHordeWorkers([])

        mock_integration = AsyncMock()
        mock_integration.get_combined_data_indexed = AsyncMock(
            return_value=(mock_status, mock_stats, mock_workers),
        )

        with patch(
            "horde_model_reference.integrations.horde_api_integration.HordeAPIIntegration",
            return_value=mock_integration,
        ):
            resp = api_client.get(
                f"{_V2}/image_generation/popular",
                params={"sort_by": "usage_day"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2
        assert data[0]["name"] == "img_nsfw_xl"

    def test_popular_limit(
        self,
        api_client: TestClient,
        primary_manager_for_search: ModelReferenceManager,
    ) -> None:
        """Validate popular endpoint respects the limit parameter to cap result count."""
        from horde_model_reference.integrations.horde_api_models import (
            HordeModelStatsResponse,
            HordeModelStatus,
            IndexedHordeModelStats,
            IndexedHordeModelStatus,
            IndexedHordeWorkers,
        )

        mock_status = IndexedHordeModelStatus(
            [
                HordeModelStatus(
                    name="img_safe_sd1", count=10, jobs=0, performance=1.0, eta=0, queued=0, type="image"
                ),
                HordeModelStatus(name="img_nsfw_xl", count=5, jobs=0, performance=1.0, eta=0, queued=0, type="image"),
                HordeModelStatus(
                    name="img_inpaint_sd1", count=1, jobs=0, performance=1.0, eta=0, queued=0, type="image"
                ),
            ]
        )
        mock_stats = IndexedHordeModelStats(HordeModelStatsResponse(day={}, month={}, total={}))
        mock_workers = IndexedHordeWorkers([])

        mock_integration = AsyncMock()
        mock_integration.get_combined_data_indexed = AsyncMock(
            return_value=(mock_status, mock_stats, mock_workers),
        )

        with patch(
            "horde_model_reference.integrations.horde_api_integration.HordeAPIIntegration",
            return_value=mock_integration,
        ):
            resp = api_client.get(f"{_V2}/image_generation/popular", params={"limit": 2})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
