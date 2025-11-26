"""Unit tests for the cache hydration module.

Tests the CacheHydrator background service, stale-while-revalidate behavior,
and cache hydration settings integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from horde_model_reference.analytics.audit_analysis import (
    CategoryAuditResponse,
    CategoryAuditSummary,
    DeletionRiskFlags,
    ModelAuditInfo,
    UsageTrend,
)
from horde_model_reference.analytics.audit_cache import AuditCache
from horde_model_reference.analytics.cache_hydrator import CacheHydrator, get_cache_hydrator
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


def create_mock_audit_response(
    category: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation,
    total_models: int = 5,
) -> CategoryAuditResponse:
    """Create a mock CategoryAuditResponse for testing.

    Args:
        category: The model reference category.
        total_models: Number of models in the response.

    Returns:
        A CategoryAuditResponse with mock data.
    """
    models = [
        ModelAuditInfo(
            name=f"test_model_{i}",
            category=category,
            deletion_risk_flags=DeletionRiskFlags(),
            at_risk=False,
            risk_score=0,
            worker_count=i + 1,
            usage_day=100 * (i + 1),
            usage_month=1000 * (i + 1),
            usage_total=10000 * (i + 1),
            usage_percentage_of_category=20.0,
            usage_trend=UsageTrend(),
            has_description=True,
            download_count=1,
            download_hosts=["huggingface.co"],
        )
        for i in range(total_models)
    ]

    summary = CategoryAuditSummary.from_audit_models(models)

    return CategoryAuditResponse(
        category=category,
        category_total_month_usage=sum(m.usage_month for m in models),
        total_count=total_models,
        returned_count=total_models,
        offset=0,
        limit=None,
        models=models,
        summary=summary,
    )


class TestCacheHydratorSingleton:
    """Tests for CacheHydrator singleton pattern."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> Generator[None, None, None]:
        """Reset CacheHydrator singleton between tests."""
        previous = CacheHydrator._instance
        CacheHydrator._instance = None
        try:
            yield
        finally:
            # Stop any running hydrator before restoring
            if CacheHydrator._instance is not None and CacheHydrator._instance._running:
                # Can't await in sync cleanup, just mark as stopped
                CacheHydrator._instance._running = False
                CacheHydrator._instance._shutdown_event.set()
            CacheHydrator._instance = previous

    def test_singleton_pattern(self) -> None:
        """Test that CacheHydrator is a singleton."""
        hydrator1 = CacheHydrator()
        hydrator2 = CacheHydrator()

        assert hydrator1 is hydrator2

    def test_get_cache_hydrator_returns_singleton(self) -> None:
        """Test that get_cache_hydrator() returns the singleton."""
        hydrator1 = get_cache_hydrator()
        hydrator2 = get_cache_hydrator()

        assert hydrator1 is hydrator2
        assert hydrator1 is CacheHydrator()

    def test_initial_state(self) -> None:
        """Test that hydrator starts in correct initial state."""
        hydrator = CacheHydrator()

        assert hydrator._running is False
        assert hydrator._task is None
        assert hydrator.is_running is False


class TestCacheHydratorStartStop:
    """Tests for CacheHydrator start/stop lifecycle."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> Generator[None, None, None]:
        """Reset CacheHydrator singleton between tests."""
        previous = CacheHydrator._instance
        CacheHydrator._instance = None
        try:
            yield
        finally:
            if CacheHydrator._instance is not None and CacheHydrator._instance._running:
                CacheHydrator._instance._running = False
                CacheHydrator._instance._shutdown_event.set()
            CacheHydrator._instance = previous

    @pytest.mark.asyncio
    async def test_start_when_disabled_does_nothing(self) -> None:
        """Test that start() does nothing when hydration is disabled."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = False

            hydrator = CacheHydrator()
            await hydrator.start()

            assert hydrator.is_running is False
            assert hydrator._task is None

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self) -> None:
        """Test that start() creates a background task when enabled."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 0

            hydrator = CacheHydrator()

            # Mock _hydrate_all_caches to prevent actual API calls
            hydrator._hydrate_all_caches = AsyncMock()  # type: ignore[method-assign]

            await hydrator.start()

            assert hydrator.is_running is True
            assert hydrator._task is not None

            # Clean up
            await hydrator.stop()

    @pytest.mark.asyncio
    async def test_start_twice_logs_warning(self) -> None:
        """Test that starting twice logs a warning and doesn't create duplicate tasks."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 0

            hydrator = CacheHydrator()
            hydrator._hydrate_all_caches = AsyncMock()  # type: ignore[method-assign]

            await hydrator.start()
            task1 = hydrator._task

            # Start again
            await hydrator.start()
            task2 = hydrator._task

            # Should be the same task
            assert task1 is task2

            await hydrator.stop()

    @pytest.mark.asyncio
    async def test_stop_gracefully_stops_task(self) -> None:
        """Test that stop() gracefully stops the background task."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 0

            hydrator = CacheHydrator()
            hydrator._hydrate_all_caches = AsyncMock()  # type: ignore[method-assign]

            await hydrator.start()
            assert hydrator.is_running is True

            await hydrator.stop()

            assert hydrator.is_running is False
            assert hydrator._task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running_does_nothing(self) -> None:
        """Test that stop() does nothing when not running."""
        hydrator = CacheHydrator()

        # Should not raise
        await hydrator.stop()

        assert hydrator.is_running is False


class TestCacheHydratorHydrationLoop:
    """Tests for the hydration loop behavior."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self) -> Generator[None, None, None]:
        """Reset CacheHydrator singleton between tests."""
        previous = CacheHydrator._instance
        CacheHydrator._instance = None
        try:
            yield
        finally:
            if CacheHydrator._instance is not None and CacheHydrator._instance._running:
                CacheHydrator._instance._running = False
                CacheHydrator._instance._shutdown_event.set()
            CacheHydrator._instance = previous

    @pytest.mark.asyncio
    async def test_hydration_loop_respects_startup_delay(self) -> None:
        """Test that hydration waits for startup delay before first run."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 1

            hydrator = CacheHydrator()
            hydrate_mock = AsyncMock()
            hydrator._hydrate_all_caches = hydrate_mock  # type: ignore[method-assign]

            await hydrator.start()

            # Should not have hydrated yet (within startup delay)
            await asyncio.sleep(0.1)
            assert hydrate_mock.call_count == 0

            # Wait past startup delay
            await asyncio.sleep(1.0)

            # Now should have hydrated
            assert hydrate_mock.call_count >= 1

            await hydrator.stop()

    @pytest.mark.asyncio
    async def test_hydration_loop_runs_at_interval(self) -> None:
        """Test that hydration runs repeatedly at the configured interval."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 0.5
            mock_settings.cache_hydration_startup_delay_seconds = 0

            hydrator = CacheHydrator()
            hydrate_mock = AsyncMock()
            hydrator._hydrate_all_caches = hydrate_mock  # type: ignore[method-assign]

            await hydrator.start()

            # Wait for multiple intervals
            await asyncio.sleep(1.2)

            # Should have run multiple times
            assert hydrate_mock.call_count >= 2

            await hydrator.stop()

    @pytest.mark.asyncio
    async def test_hydration_loop_handles_errors_gracefully(self) -> None:
        """Test that errors in hydration don't crash the loop."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 0.2
            mock_settings.cache_hydration_startup_delay_seconds = 0

            hydrator = CacheHydrator()

            # Make hydration fail on first call, succeed on second
            call_count = 0

            async def failing_hydrate() -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("Test error")

            hydrator._hydrate_all_caches = failing_hydrate  # type: ignore[method-assign]

            await hydrator.start()

            # Wait for multiple intervals
            await asyncio.sleep(0.5)

            # Should have continued despite error
            assert call_count >= 2
            assert hydrator.is_running is True

            await hydrator.stop()

    @pytest.mark.asyncio
    async def test_hydration_stops_on_shutdown_during_delay(self) -> None:
        """Test that hydration stops properly if shutdown is requested during startup delay."""
        with patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings:
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 5

            hydrator = CacheHydrator()
            hydrate_mock = AsyncMock()
            hydrator._hydrate_all_caches = hydrate_mock  # type: ignore[method-assign]

            await hydrator.start()
            await asyncio.sleep(0.1)  # Let it start

            # Stop before startup delay completes
            await hydrator.stop()

            # Should not have hydrated at all
            assert hydrate_mock.call_count == 0


class TestCacheHydratorHydration:
    """Tests for actual cache hydration behavior."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> Generator[None, None, None]:
        """Reset relevant singletons between tests."""
        # Reset CacheHydrator
        prev_hydrator = CacheHydrator._instance
        CacheHydrator._instance = None

        # Reset AuditCache
        prev_audit = AuditCache._instance  # type: ignore[misc]
        AuditCache._instance = None  # type: ignore[misc]

        try:
            yield
        finally:
            if CacheHydrator._instance is not None and CacheHydrator._instance._running:
                CacheHydrator._instance._running = False
                CacheHydrator._instance._shutdown_event.set()
            CacheHydrator._instance = prev_hydrator

            if AuditCache._instance is not None:  # type: ignore[misc]
                with contextlib.suppress(Exception):
                    AuditCache._instance.clear_all()  # type: ignore[misc]
            AuditCache._instance = prev_audit  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_hydrate_audit_cache_stores_response(self) -> None:
        """Test that _hydrate_audit_cache stores computed response in cache."""
        mock_response = create_mock_audit_response(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=3,
        )

        hydrator = CacheHydrator()

        with (
            patch.object(hydrator, "_compute_audit_response", return_value=mock_response) as mock_compute,
            patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings,
        ):
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 300
            mock_settings.cache_hydration_enabled = False  # Don't need hydration running

            await hydrator._hydrate_audit_cache(
                MODEL_REFERENCE_CATEGORY.image_generation,
                grouped=False,
                include_backend_variations=False,
            )

            mock_compute.assert_called_once_with(
                MODEL_REFERENCE_CATEGORY.image_generation,
                grouped=False,
                include_backend_variations=False,
            )

            # Verify cache was populated
            cache = AuditCache()
            cached = cache.get(
                MODEL_REFERENCE_CATEGORY.image_generation,
                grouped=False,
                include_backend_variations=False,
            )

            assert cached is not None
            assert cached.total_count == 3

    @pytest.mark.asyncio
    async def test_hydrate_audit_cache_handles_none_response(self) -> None:
        """Test that _hydrate_audit_cache handles None compute response gracefully."""
        hydrator = CacheHydrator()

        with (
            patch.object(hydrator, "_compute_audit_response", return_value=None),
            patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings,
        ):
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 300
            mock_settings.cache_hydration_enabled = False

            # Should not raise
            await hydrator._hydrate_audit_cache(
                MODEL_REFERENCE_CATEGORY.image_generation,
                grouped=False,
                include_backend_variations=False,
            )

    @pytest.mark.asyncio
    async def test_hydrate_all_caches_hydrates_all_variants(self) -> None:
        """Test that _hydrate_all_caches hydrates all cache variants."""
        hydrator = CacheHydrator()

        with patch.object(hydrator, "_hydrate_audit_cache", new_callable=AsyncMock) as mock_hydrate:
            hydrator._running = True  # Simulate running state

            await hydrator._hydrate_all_caches()

            # Should hydrate image_generation (grouped and ungrouped)
            # Should hydrate text_generation (grouped, ungrouped, and with backend variations)
            expected_calls = [
                # image_generation
                (
                    (MODEL_REFERENCE_CATEGORY.image_generation,),
                    {"grouped": False, "include_backend_variations": False},
                ),
                (
                    (MODEL_REFERENCE_CATEGORY.image_generation,),
                    {"grouped": True, "include_backend_variations": False},
                ),
                # text_generation
                (
                    (MODEL_REFERENCE_CATEGORY.text_generation,),
                    {"grouped": False, "include_backend_variations": False},
                ),
                (
                    (MODEL_REFERENCE_CATEGORY.text_generation,),
                    {"grouped": True, "include_backend_variations": False},
                ),
                (
                    (MODEL_REFERENCE_CATEGORY.text_generation,),
                    {"grouped": False, "include_backend_variations": True},
                ),
            ]

            assert mock_hydrate.call_count == len(expected_calls)

            for call, (args, kwargs) in zip(mock_hydrate.call_args_list, expected_calls, strict=False):
                assert call.args == args
                assert call.kwargs == kwargs

    @pytest.mark.asyncio
    async def test_hydrate_all_caches_stops_early_when_shutdown(self) -> None:
        """Test that _hydrate_all_caches stops early when shutdown is requested."""
        hydrator = CacheHydrator()

        call_count = 0

        async def counting_hydrate(*args: object, **kwargs: object) -> None:
            nonlocal call_count
            _ = args, kwargs  # Explicitly unused
            call_count += 1
            # Simulate shutdown request after first call
            if call_count == 1:
                hydrator._running = False

        with patch.object(hydrator, "_hydrate_audit_cache", side_effect=counting_hydrate):
            hydrator._running = True

            await hydrator._hydrate_all_caches()

            # Should have stopped after first call
            assert call_count == 1


class TestStaleWhileRevalidate:
    """Tests for stale-while-revalidate behavior in RedisCache."""

    @pytest.fixture(autouse=True)
    def reset_audit_cache(self) -> Generator[None, None, None]:
        """Reset AuditCache singleton between tests."""
        previous = AuditCache._instance  # type: ignore[misc]
        AuditCache._instance = None  # type: ignore[misc]
        try:
            yield
        finally:
            if AuditCache._instance is not None:  # type: ignore[misc]
                with contextlib.suppress(Exception):
                    AuditCache._instance.clear_all()  # type: ignore[misc]
            AuditCache._instance = previous  # type: ignore[misc]

    def test_get_returns_stale_data_when_hydration_enabled(self) -> None:
        """Test that get() returns stale data when hydration is enabled."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 1  # 1 second TTL
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_stale_ttl_seconds = 3600  # 1 hour stale TTL

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            # Wait for normal TTL to expire
            time.sleep(1.2)

            # Should still return stale data
            result = cache.get(MODEL_REFERENCE_CATEGORY.image_generation)
            assert result is not None
            assert result.total_count == mock_response.total_count

    def test_get_returns_none_when_stale_ttl_exceeded(self) -> None:
        """Test that get() returns None when stale TTL is exceeded."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 1  # 1 second TTL
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_stale_ttl_seconds = 2  # 2 second stale TTL

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            # Wait for stale TTL to expire
            time.sleep(2.2)

            # Should return None now
            result = cache.get(MODEL_REFERENCE_CATEGORY.image_generation)
            assert result is None

    def test_get_with_allow_stale_false_respects_ttl(self) -> None:
        """Test that get(allow_stale=False) respects normal TTL even with hydration enabled."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 1  # 1 second TTL
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_stale_ttl_seconds = 3600

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            # Wait for normal TTL to expire
            time.sleep(1.2)

            # Should return None when allow_stale=False
            result = cache.get(MODEL_REFERENCE_CATEGORY.image_generation, allow_stale=False)
            assert result is None

    def test_get_defaults_to_no_stale_when_hydration_disabled(self) -> None:
        """Test that get() defaults to normal TTL behavior when hydration is disabled."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 1
            mock_settings.cache_hydration_enabled = False
            mock_settings.cache_hydration_stale_ttl_seconds = 3600

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            # Wait for normal TTL to expire
            time.sleep(1.2)

            # Should return None (no stale data without hydration)
            result = cache.get(MODEL_REFERENCE_CATEGORY.image_generation)
            assert result is None

    def test_is_fresh_returns_true_within_ttl(self) -> None:
        """Test that is_fresh() returns True within TTL."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 300
            mock_settings.cache_hydration_enabled = False

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            assert cache.is_fresh(MODEL_REFERENCE_CATEGORY.image_generation) is True

    def test_is_fresh_returns_false_after_ttl(self) -> None:
        """Test that is_fresh() returns False after TTL expires."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 1
            mock_settings.cache_hydration_enabled = False

            cache = AuditCache()
            mock_response = create_mock_audit_response()

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, mock_response)

            # Wait for TTL to expire
            time.sleep(1.2)

            assert cache.is_fresh(MODEL_REFERENCE_CATEGORY.image_generation) is False

    def test_is_fresh_returns_false_for_missing_entry(self) -> None:
        """Test that is_fresh() returns False for missing entries."""
        with patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.audit_cache_ttl = 300
            mock_settings.cache_hydration_enabled = False

            cache = AuditCache()

            assert cache.is_fresh(MODEL_REFERENCE_CATEGORY.image_generation) is False


class TestCacheHydrationSettings:
    """Tests for cache hydration settings."""

    def test_default_settings(self) -> None:
        """Test default hydration settings values."""
        from horde_model_reference import HordeModelReferenceSettings

        settings = HordeModelReferenceSettings()

        assert settings.cache_hydration_enabled is False
        assert settings.cache_hydration_interval_seconds == 240
        assert settings.cache_hydration_stale_ttl_seconds == 3600
        assert settings.cache_hydration_startup_delay_seconds == 5

    def test_settings_can_be_overridden(self) -> None:
        """Test that hydration settings can be overridden."""
        from horde_model_reference import HordeModelReferenceSettings

        settings = HordeModelReferenceSettings(
            cache_hydration_enabled=True,
            cache_hydration_interval_seconds=120,
            cache_hydration_stale_ttl_seconds=1800,
            cache_hydration_startup_delay_seconds=10,
        )

        assert settings.cache_hydration_enabled is True
        assert settings.cache_hydration_interval_seconds == 120
        assert settings.cache_hydration_stale_ttl_seconds == 1800
        assert settings.cache_hydration_startup_delay_seconds == 10


class TestCacheHydrationIntegration:
    """Integration tests for cache hydration with mocked Horde API."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self) -> Generator[None, None, None]:
        """Reset relevant singletons between tests."""
        prev_hydrator = CacheHydrator._instance
        CacheHydrator._instance = None

        prev_audit = AuditCache._instance  # type: ignore[misc]
        AuditCache._instance = None  # type: ignore[misc]

        try:
            yield
        finally:
            if CacheHydrator._instance is not None and CacheHydrator._instance._running:
                CacheHydrator._instance._running = False
                CacheHydrator._instance._shutdown_event.set()
            CacheHydrator._instance = prev_hydrator

            if AuditCache._instance is not None:  # type: ignore[misc]
                with contextlib.suppress(Exception):
                    AuditCache._instance.clear_all()  # type: ignore[misc]
            AuditCache._instance = prev_audit  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_compute_audit_response_with_mocked_dependencies(self) -> None:
        """Test _compute_audit_response with fully mocked dependencies."""
        from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
        from horde_model_reference.integrations.horde_api_models import (
            IndexedHordeModelStats,
            IndexedHordeModelStatus,
        )
        from horde_model_reference.model_reference_records import ImageGenerationModelRecord

        hydrator = CacheHydrator()

        # Create mock model records
        mock_model_records = {
            "test_model": ImageGenerationModelRecord(
                name="test_model",
                description="A test model",
                baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
                inpainting=False,
                nsfw=False,
            ),
        }

        # Mock all dependencies
        mock_manager = MagicMock()
        mock_manager.get_model_names.return_value = ["test_model"]
        mock_manager.get_model_reference.return_value = mock_model_records

        mock_horde_api = MagicMock()
        mock_status = IndexedHordeModelStatus([])
        mock_stats = IndexedHordeModelStats(MagicMock(day={}, month={}, total={}))
        mock_horde_api.get_model_status_indexed = AsyncMock(return_value=mock_status)
        mock_horde_api.get_model_stats_indexed = AsyncMock(return_value=mock_stats)

        with (
            patch(
                "horde_model_reference.analytics.cache_hydrator.ModelReferenceManager",
                return_value=mock_manager,
            ),
            patch(
                "horde_model_reference.analytics.cache_hydrator.HordeAPIIntegration",
                return_value=mock_horde_api,
            ),
        ):
            result = await hydrator._compute_audit_response(
                MODEL_REFERENCE_CATEGORY.image_generation,
                grouped=False,
                include_backend_variations=False,
            )

            assert result is not None
            assert result.category == MODEL_REFERENCE_CATEGORY.image_generation
            assert result.total_count == 1
            assert len(result.models) == 1
            assert result.models[0].name == "test_model"

    @pytest.mark.asyncio
    async def test_full_hydration_cycle_with_mocks(self) -> None:
        """Test a full hydration cycle with mocked external services."""
        mock_response = create_mock_audit_response()

        hydrator = CacheHydrator()

        # Mock _compute_audit_response to return our mock response
        with (
            patch.object(hydrator, "_compute_audit_response", return_value=mock_response),
            patch("horde_model_reference.analytics.cache_hydrator.horde_model_reference_settings") as mock_settings,
            patch("horde_model_reference.analytics.base_cache.horde_model_reference_settings") as mock_base_settings,
        ):
            mock_settings.cache_hydration_enabled = True
            mock_settings.cache_hydration_interval_seconds = 60
            mock_settings.cache_hydration_startup_delay_seconds = 0

            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_base_settings.redis = mock_redis
            mock_base_settings.audit_cache_ttl = 300
            mock_base_settings.cache_hydration_enabled = True
            mock_base_settings.cache_hydration_stale_ttl_seconds = 3600

            # Run single hydration cycle
            hydrator._running = True
            await hydrator._hydrate_all_caches()
            hydrator._running = False

            # Verify all caches are populated
            cache = AuditCache()

            # Image generation caches
            assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation, grouped=False) is not None
            assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation, grouped=True) is not None

            # Text generation caches
            assert cache.get(MODEL_REFERENCE_CATEGORY.text_generation, grouped=False) is not None
            assert cache.get(MODEL_REFERENCE_CATEGORY.text_generation, grouped=True) is not None
            assert (
                cache.get(
                    MODEL_REFERENCE_CATEGORY.text_generation,
                    grouped=False,
                    include_backend_variations=True,
                )
                is not None
            )
