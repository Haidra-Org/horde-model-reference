"""Tests for HordeAPIIntegration with mocked Horde API responses."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from pytest_httpx import HTTPXMock

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.integrations import (
    HordeAPIIntegration,
    HordeModelStatsResponse,
    HordeModelStatus,
    HordeWorker,
)


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator[None, None, None]:
    """Reset the HordeAPIIntegration singleton before each test.

    This matches the project's existing pattern of restoring singletons after
    tests to avoid cross-test pollution (see ``restore_manager_singleton`` in conftest).
    """
    previous = HordeAPIIntegration._instance
    HordeAPIIntegration._instance = None
    try:
        yield
    finally:
        HordeAPIIntegration._instance = previous


@pytest.fixture
def mock_horde_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure horde_model_reference_settings for unit tests.

    Uses monkeypatch to alter attributes on the real settings object, following
    the pattern used in other tests like ``legacy_canonical_mode`` in conftest.
    """
    monkeypatch.setattr(horde_model_reference_settings, "horde_api_cache_ttl", 60)
    monkeypatch.setattr(horde_model_reference_settings, "horde_api_timeout", 10)
    monkeypatch.setattr(horde_model_reference_settings.redis, "use_redis", False)


@pytest.fixture
def sample_status_response() -> list[dict[str, Any]]:
    """Sample model status response data."""
    return [
        {
            "name": "Deliberate",
            "count": 127,
            "performance": 15.2,
            "queued": 45000,
            "jobs": 45,
            "eta": 120,
            "type": "image",
        },
        {
            "name": "SDXL 1.0",
            "count": 89,
            "performance": 12.8,
            "queued": 32000,
            "jobs": 32,
            "eta": 95,
            "type": "image",
        },
    ]


@pytest.fixture
def sample_stats_response() -> dict[str, dict[str, int]]:
    """Sample model statistics response data."""
    return {
        "day": {
            "Deliberate": 1234,
            "SDXL 1.0": 2345,
        },
        "month": {
            "Deliberate": 45678,
            "SDXL 1.0": 56789,
        },
        "total": {
            "Deliberate": 987654,
            "SDXL 1.0": 876543,
        },
    }


@pytest.fixture
def sample_workers_response() -> list[dict[str, Any]]:
    """Sample workers response data."""
    return [
        {
            "id": "12345678-1234-1234-1234-123456789abc",
            "name": "Test Worker 1",
            "type": "image",
            "performance": "0.4 megapixelsteps per second",
            "requests_fulfilled": 1000,
            "kudos_rewards": 5000.0,
            "kudos_details": {"generated": 4500.0, "uptime": 500.0},
            "threads": 1,
            "uptime": 86400,
            "uncompleted_jobs": 0,
            "maintenance_mode": False,
            "nsfw": True,
            "trusted": True,
            "flagged": False,
            "online": True,
            "models": ["Deliberate", "SDXL 1.0"],
            "team": {"name": "Test Team", "id": "team-uuid"},
            "bridge_agent": "AI Horde Worker:5.0.0",
            "max_pixels": 4194304,
            "megapixelsteps_generated": 12345.0,
            "img2img": True,
            "painting": True,
            "post-processing": True,
            "lora": True,
            "controlnet": False,
            "sdxl_controlnet": False,
        },
    ]


class TestHordeAPIIntegrationSingleton:
    """Test singleton behavior of HordeAPIIntegration."""

    def test_singleton_same_instance(self, mock_horde_settings: None) -> None:
        """Test that multiple calls return the same instance."""
        integration1 = HordeAPIIntegration()
        integration2 = HordeAPIIntegration()

        assert integration1 is integration2

    def test_singleton_initialization_once(self, mock_horde_settings: None) -> None:
        """Test that initialization only happens once."""
        with patch.object(HordeAPIIntegration, "_initialize") as mock_init:
            _ = HordeAPIIntegration()
            _ = HordeAPIIntegration()

            # _initialize should only be called once
            assert mock_init.call_count == 1


class TestHordeAPIIntegrationStatus:
    """Test model status fetching."""

    @pytest.mark.asyncio
    async def test_get_model_status_success(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test successful model status fetching."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )

        result = await integration.get_model_status("image")

        assert len(result) == 2
        assert isinstance(result[0], HordeModelStatus)
        assert result[0].name == "Deliberate"
        assert result[0].count == 127
        assert result[1].name == "SDXL 1.0"

    @pytest.mark.asyncio
    async def test_get_model_status_caching(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test that status responses are cached."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )

        # First call - should hit API
        result1 = await integration.get_model_status("image")

        # Second call - should use cache
        result2 = await integration.get_model_status("image")

        # API should only be called once (verify with httpx_mock)
        requests = httpx_mock.get_requests()
        assert len(requests) == 1

        # Results should be identical
        assert result1[0].name == result2[0].name
        assert result1[0].count == result2[0].count

    @pytest.mark.asyncio
    async def test_get_model_status_force_refresh(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test force refresh bypasses cache."""
        integration = HordeAPIIntegration()

        # Register response twice since we expect two calls
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )

        # First call
        await integration.get_model_status("image")

        # Second call with force_refresh
        await integration.get_model_status("image", force_refresh=True)

        # API should be called twice
        requests = httpx_mock.get_requests()
        assert len(requests) == 2


class TestHordeAPIIntegrationStats:
    """Test model statistics fetching."""

    @pytest.mark.asyncio
    async def test_get_model_stats_success(
        self,
        mock_horde_settings: None,
        sample_stats_response: dict[str, dict[str, int]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test successful model stats fetching."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )

        result = await integration.get_model_stats("image")

        assert isinstance(result, HordeModelStatsResponse)
        assert "Deliberate" in result.day
        assert result.day["Deliberate"] == 1234
        assert result.month["Deliberate"] == 45678

    @pytest.mark.asyncio
    async def test_get_model_stats_caching(
        self,
        mock_horde_settings: None,
        sample_stats_response: dict[str, dict[str, int]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test that stats responses are cached."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )

        # First call - should hit API
        await integration.get_model_stats("image")

        # Second call - should use cache
        await integration.get_model_stats("image")

        # API should only be called once
        requests = httpx_mock.get_requests()
        assert len(requests) == 1


class TestHordeAPIIntegrationWorkers:
    """Test workers fetching."""

    @pytest.mark.asyncio
    async def test_get_workers_success(
        self,
        mock_horde_settings: None,
        sample_workers_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test successful workers fetching."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/workers?type=image",
            json=sample_workers_response,
        )

        result = await integration.get_workers("image")

        assert len(result) == 1
        assert isinstance(result[0], HordeWorker)
        assert result[0].name == "Test Worker 1"
        assert "Deliberate" in result[0].models

    @pytest.mark.asyncio
    async def test_get_workers_all_types(
        self,
        mock_horde_settings: None,
        sample_workers_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test fetching all workers without type filter."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/workers",
            json=sample_workers_response,
        )

        result = await integration.get_workers(None)

        assert len(result) == 1
        assert isinstance(result[0], HordeWorker)


class TestHordeAPIIntegrationCombined:
    """Test combined data fetching."""

    @pytest.mark.asyncio
    async def test_get_combined_data(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        sample_stats_response: dict[str, dict[str, int]],
        sample_workers_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test fetching all data in parallel."""
        integration = HordeAPIIntegration()

        # Register responses for all three endpoints
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/workers?type=image",
            json=sample_workers_response,
        )

        status, stats, workers = await integration.get_combined_data("image", include_workers=True)

        assert len(status) == 2
        assert isinstance(stats, HordeModelStatsResponse)
        assert workers is not None
        assert len(workers) == 1

    @pytest.mark.asyncio
    async def test_get_combined_data_without_workers(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        sample_stats_response: dict[str, dict[str, int]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test fetching combined data without workers."""
        integration = HordeAPIIntegration()

        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )

        status, stats, workers = await integration.get_combined_data("image", include_workers=False)

        assert len(status) == 2
        assert isinstance(stats, HordeModelStatsResponse)
        assert workers is None


class TestHordeAPIIntegrationCacheInvalidation:
    """Test cache invalidation."""

    @pytest.mark.asyncio
    async def test_invalidate_specific_type(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test invalidating cache for specific model type."""
        integration = HordeAPIIntegration()

        # Register response twice since we expect two calls
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )

        # Fetch and cache
        await integration.get_model_status("image")

        # Invalidate
        integration.invalidate_cache("image")

        # Fetch again - should hit API
        await integration.get_model_status("image")

        # API should be called twice (once before invalidation, once after)
        requests = httpx_mock.get_requests()
        assert len(requests) == 2

    @pytest.mark.asyncio
    async def test_invalidate_all(
        self,
        mock_horde_settings: None,
        sample_status_response: list[dict[str, Any]],
        sample_stats_response: dict[str, dict[str, int]],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test invalidating all caches."""
        integration = HordeAPIIntegration()

        # Register responses for initial fetches
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )

        # Fetch and cache both
        await integration.get_model_status("image")
        await integration.get_model_stats("image")

        # Invalidate all
        integration.invalidate_cache(None)

        # Register responses again for post-invalidation fetches
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/status/models?type=image&model_state=known",
            json=sample_status_response,
        )
        httpx_mock.add_response(
            url="https://aihorde.net/api/v2/stats/img/models?model_state=known",
            json=sample_stats_response,
        )

        # Fetch again - both should hit API
        await integration.get_model_status("image")
        await integration.get_model_stats("image")

        # API should be called 4 times total (2 before, 2 after invalidation)
        requests = httpx_mock.get_requests()
        assert len(requests) == 4
