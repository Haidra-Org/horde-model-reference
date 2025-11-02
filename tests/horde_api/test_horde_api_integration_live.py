"""Integration tests for HordeAPIIntegration against live production API.

These tests make real HTTP requests to the AI Horde public API.
They are marked with @pytest.mark.integration and can be run separately:

    pytest tests/test_horde_api_integration_live.py -m integration

IMPORTANT: These tests require network access and depend on the availability
of the live Horde API at https://aihorde.net/api/v2
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

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
def mock_redis_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure horde_model_reference_settings for live integration tests.

    Uses monkeypatch to alter attributes on the real settings object, following
    the pattern used in other tests like ``legacy_canonical_mode`` in conftest.
    Disables Redis and uses longer timeouts for live API calls.
    """
    monkeypatch.setattr(horde_model_reference_settings, "horde_api_cache_ttl", 60)
    monkeypatch.setattr(horde_model_reference_settings, "horde_api_timeout", 30)  # Longer timeout for live API
    monkeypatch.setattr(horde_model_reference_settings.redis, "use_redis", False)


@pytest.mark.integration
class TestHordeAPIIntegrationLiveStatus:
    """Test live model status fetching from production API."""

    @pytest.mark.asyncio
    async def test_get_image_model_status(self, mock_redis_disabled: None) -> None:
        """Test fetching live image model status."""
        integration = HordeAPIIntegration()

        result = await integration.get_model_status("image")

        # Should get a non-empty list
        assert len(result) > 0
        assert isinstance(result[0], HordeModelStatus)

        # Check that models have expected fields
        first_model = result[0]
        assert first_model.name
        assert first_model.type == "image"
        assert first_model.count >= 0
        assert first_model.performance >= 0

        print(f"Fetched {len(result)} image models from live API")
        print(f"Sample model: {first_model.name} ({first_model.count} workers)")

    @pytest.mark.asyncio
    async def test_get_text_model_status(self, mock_redis_disabled: None) -> None:
        """Test fetching live text model status."""
        integration = HordeAPIIntegration()

        result = await integration.get_model_status("text")

        # Should get a non-empty list
        assert len(result) > 0
        assert isinstance(result[0], HordeModelStatus)

        # Check that models have expected fields
        first_model = result[0]
        assert first_model.name
        assert first_model.type == "text"
        assert first_model.count >= 0

        print(f"Fetched {len(result)} text models from live API")
        print(f"Sample model: {first_model.name} ({first_model.count} workers)")

    @pytest.mark.asyncio
    async def test_status_caching_works(self, mock_redis_disabled: None) -> None:
        """Test that caching works with live API."""
        integration = HordeAPIIntegration()

        # First call - should hit API
        result1 = await integration.get_model_status("image")

        # Second call - should use cache (verify by checking it's instant)
        import time

        start = time.time()
        result2 = await integration.get_model_status("image")
        duration = time.time() - start

        # Cache should be much faster than API call (< 10ms)
        assert duration < 0.01

        # Results should have same length
        assert len(result1) == len(result2)

        print(f"Cache retrieval took {duration * 1000:.2f}ms")


@pytest.mark.integration
class TestHordeAPIIntegrationLiveStats:
    """Test live model statistics fetching from production API."""

    @pytest.mark.asyncio
    async def test_get_image_model_stats(self, mock_redis_disabled: None) -> None:
        """Test fetching live image model statistics."""
        integration = HordeAPIIntegration()

        result = await integration.get_model_stats("image")

        assert isinstance(result, HordeModelStatsResponse)

        # Should have data for day, month, and total
        assert isinstance(result.day, dict)
        assert isinstance(result.month, dict)
        assert isinstance(result.total, dict)

        # Should have at least some model data
        assert len(result.total) > 0

        print(f"Fetched stats for {len(result.total)} image models")

        # Show sample stats for a model
        if result.total:
            sample_model = next(iter(result.total.keys()))
            print(
                f"Sample: {sample_model} - "
                f"day: {result.day.get(sample_model, 0)}, "
                f"month: {result.month.get(sample_model, 0)}, "
                f"total: {result.total.get(sample_model, 0)}"
            )

    @pytest.mark.asyncio
    async def test_get_text_model_stats(self, mock_redis_disabled: None) -> None:
        """Test fetching live text model statistics."""
        integration = HordeAPIIntegration()

        result = await integration.get_model_stats("text")

        assert isinstance(result, HordeModelStatsResponse)
        assert len(result.total) > 0

        print(f"Fetched stats for {len(result.total)} text models")


@pytest.mark.integration
class TestHordeAPIIntegrationLiveWorkers:
    """Test live workers fetching from production API."""

    @pytest.mark.asyncio
    async def test_get_image_workers(self, mock_redis_disabled: None) -> None:
        """Test fetching live image workers.

        Note: This test may find 0 workers if no image workers are online.
        We use text workers as fallback since they're more consistently available.
        """
        integration = HordeAPIIntegration()

        # Try image workers first
        result = await integration.get_workers("image")

        # If no image workers, try text workers as they're more reliable
        if len(result) == 0:
            result = await integration.get_workers("text")
            if len(result) == 0:
                pytest.skip("No workers available on live API at this time")
            worker_type = "text"
        else:
            worker_type = "image"

        # Should get a non-empty list
        assert len(result) > 0
        assert isinstance(result[0], HordeWorker)

        # Check that workers have expected fields
        first_worker = result[0]
        assert first_worker.name
        assert first_worker.id
        assert first_worker.type in ("image", "text")
        assert isinstance(first_worker.models, list)

        print(f"Fetched {len(result)} {worker_type} workers from live API")
        print(f"Sample worker: {first_worker.name} (serving {len(first_worker.models)} models)")

    @pytest.mark.asyncio
    async def test_get_all_workers(self, mock_redis_disabled: None) -> None:
        """Test fetching all workers without type filter.

        Note: May not have both image and text workers at all times.
        """
        integration = HordeAPIIntegration()

        result = await integration.get_workers(None)

        # Should get a non-empty list
        if len(result) == 0:
            pytest.skip("No workers available on live API at this time")

        # Count workers by type
        image_workers = [w for w in result if w.type == "image"]
        text_workers = [w for w in result if w.type == "text"]

        # Should have at least one type of worker
        assert len(image_workers) > 0 or len(text_workers) > 0

        print(f"Fetched {len(result)} total workers ({len(image_workers)} image, {len(text_workers)} text)")


@pytest.mark.integration
class TestHordeAPIIntegrationLiveCombined:
    """Test live combined data fetching."""

    @pytest.mark.asyncio
    async def test_get_combined_data_image(self, mock_redis_disabled: None) -> None:
        """Test fetching all image data in parallel.

        Note: This test may find 0 workers if no image workers are online.
        We use text workers as fallback since they're more consistently available.
        """
        integration = HordeAPIIntegration()

        # Try image first
        status, stats, workers = await integration.get_combined_data("image", include_workers=True)

        # If no image workers, try text as they're more reliable
        if workers is not None and len(workers) == 0:
            status, stats, workers = await integration.get_combined_data("text", include_workers=True)
            if workers is not None and len(workers) == 0:
                pytest.skip("No workers available on live API at this time")

        # Verify all data was fetched
        assert len(status) > 0
        assert isinstance(stats, HordeModelStatsResponse)
        assert len(stats.total) > 0
        assert workers is not None
        assert len(workers) > 0

        print(f"Combined fetch: {len(status)} status, {len(stats.total)} stats, {len(workers)} workers")

    @pytest.mark.asyncio
    async def test_get_combined_data_text(self, mock_redis_disabled: None) -> None:
        """Test fetching all text data in parallel."""
        integration = HordeAPIIntegration()

        status, stats, workers = await integration.get_combined_data("text", include_workers=True)

        # Verify all data was fetched
        assert len(status) > 0
        assert isinstance(stats, HordeModelStatsResponse)
        assert workers is not None

        print(f"Combined fetch: {len(status)} status, {len(stats.total)} stats, {len(workers)} workers")


@pytest.mark.integration
class TestHordeAPIIntegrationLiveDataConsistency:
    """Test data consistency across different endpoints."""

    @pytest.mark.asyncio
    async def test_model_names_consistent(self, mock_redis_disabled: None) -> None:
        """Test that model names appear consistently across status and stats."""
        integration = HordeAPIIntegration()

        status = await integration.get_model_status("image")
        stats = await integration.get_model_stats("image")

        # Get model names from status
        status_names = {model.name for model in status if model.count > 0}

        # Get model names from stats (models with any total usage)
        stats_names = {name for name, count in stats.total.items() if count > 0}

        # There should be significant overlap
        # (not perfect because some models might have usage but no active workers, or vice versa)
        overlap = status_names & stats_names
        assert len(overlap) > 0

        print(f"Status models: {len(status_names)}, Stats models: {len(stats_names)}, Overlap: {len(overlap)}")

    @pytest.mark.asyncio
    async def test_worker_models_exist_in_status(self, mock_redis_disabled: None) -> None:
        """Test that models served by workers appear in status.

        Note: This test may find 0 workers if no image workers are online.
        We use text workers as fallback since they're more consistently available.
        """
        integration = HordeAPIIntegration()

        # Try image first
        status = await integration.get_model_status("image")
        workers = await integration.get_workers("image")

        # If no image workers, use text workers as they're more reliable
        if len(workers) == 0:
            status = await integration.get_model_status("text")
            workers = await integration.get_workers("text")
            if len(workers) == 0:
                pytest.skip("No workers available on live API at this time")

        # Get all model names from status
        status_names = {model.name.lower() for model in status}

        # Get all model names from workers
        worker_model_names = set()
        for worker in workers[:10]:  # Check first 10 workers
            for model_name in worker.models:
                worker_model_names.add(model_name.lower())

        # Most models served by workers should appear in status
        found_in_status = worker_model_names & status_names
        coverage = len(found_in_status) / len(worker_model_names) if worker_model_names else 0

        print(
            f"Worker models: {len(worker_model_names)}, "
            f"Found in status: {len(found_in_status)} ({coverage * 100:.1f}%)"
        )

        # Should have at least 50% coverage (some models might be custom/unknown)
        assert coverage > 0.5


@pytest.mark.integration
class TestHordeAPIIntegrationLiveErrorHandling:
    """Test error handling with live API."""

    @pytest.mark.asyncio
    async def test_invalid_parameters_handled(self, mock_redis_disabled: None) -> None:
        """Test that invalid parameters are handled gracefully."""
        integration = HordeAPIIntegration()

        # This should not crash even with unusual parameters
        result = await integration.get_model_status("image", min_count=99999)

        # Should get empty or near-empty list
        assert isinstance(result, list)
        print(f"High min_count filter returned {len(result)} models")

    @pytest.mark.asyncio
    async def test_force_refresh_works(self, mock_redis_disabled: None) -> None:
        """Test that force refresh fetches new data."""
        integration = HordeAPIIntegration()

        # First call
        result1 = await integration.get_model_status("image")

        # Force refresh
        result2 = await integration.get_model_status("image", force_refresh=True)

        # Should get results (may or may not be identical depending on timing)
        assert len(result1) > 0
        assert len(result2) > 0

        print(f"Normal fetch: {len(result1)} models, Force refresh: {len(result2)} models")
