"""Unit tests for the statistics cache module."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest

from horde_model_reference.analytics.statistics import CategoryStatistics
from horde_model_reference.analytics.statistics_cache import StatisticsCache
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class TestStatisticsCache:
    """Tests for StatisticsCache singleton and caching behavior."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self) -> Generator[None, None, None]:
        """Reset singleton between tests.

        This matches the project's existing pattern of restoring singletons after
        tests to avoid cross-test pollution (see `restore_manager_singleton` in conftest).
        """
        previous = StatisticsCache._instance  # type: ignore[misc]
        StatisticsCache._instance = None  # type: ignore[misc]
        try:
            yield
        finally:
            if StatisticsCache._instance is not None:  # type: ignore[misc]
                with contextlib.suppress(Exception):
                    StatisticsCache._instance.clear_all()
            StatisticsCache._instance = previous  # type: ignore[misc]

    def test_singleton_pattern(self) -> None:
        """Test that StatisticsCache is a singleton."""
        cache1 = StatisticsCache()
        cache2 = StatisticsCache()

        assert cache1 is cache2

    def test_cache_get_miss(self) -> None:
        """Test cache miss returns None."""
        cache = StatisticsCache()

        result = cache.get(MODEL_REFERENCE_CATEGORY.image_generation)

        assert result is None

    def test_cache_set_and_get(self) -> None:
        """Test storing and retrieving statistics from cache."""
        cache = StatisticsCache()

        stats = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            sfw_count=8,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats)
        retrieved = cache.get(MODEL_REFERENCE_CATEGORY.image_generation)

        assert retrieved is not None
        assert retrieved.category == MODEL_REFERENCE_CATEGORY.image_generation
        assert retrieved.total_models == 10
        assert retrieved.nsfw_count == 2
        assert retrieved.sfw_count == 8

    def test_cache_invalidate(self) -> None:
        """Test cache invalidation removes entry."""
        cache = StatisticsCache()

        stats = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            sfw_count=8,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats)
        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is not None

        cache.invalidate(MODEL_REFERENCE_CATEGORY.image_generation)
        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is None

    def test_cache_ttl_expiration(self) -> None:
        """Test that cache entries expire after TTL."""
        with patch("horde_model_reference.analytics.statistics_cache.horde_model_reference_settings") as mock_settings:
            mock_redis = Mock()
            mock_redis.use_redis = False
            mock_settings.redis = mock_redis
            mock_settings.statistics_cache_ttl = 1

            cache = StatisticsCache()

            stats = CategoryStatistics(
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                total_models=10,
                returned_models=10,
                offset=0,
                limit=None,
                nsfw_count=2,
                sfw_count=8,
                computed_at=int(time.time()),
            )

            cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats)
            assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is not None

            time.sleep(1.1)

            assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is None

    def test_cache_clear_all(self) -> None:
        """Test clearing all cache entries."""
        cache = StatisticsCache()

        stats1 = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            sfw_count=8,
            computed_at=int(time.time()),
        )

        stats2 = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            total_models=20,
            returned_models=20,
            offset=0,
            limit=None,
            nsfw_count=0,
            sfw_count=20,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats1)
        cache.set(MODEL_REFERENCE_CATEGORY.text_generation, stats2)

        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is not None
        assert cache.get(MODEL_REFERENCE_CATEGORY.text_generation) is not None

        cache.clear_all()

        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is None
        assert cache.get(MODEL_REFERENCE_CATEGORY.text_generation) is None

    def test_cache_info(self) -> None:
        """Test get_cache_info returns correct information."""
        cache = StatisticsCache()

        info = cache.get_cache_info()

        assert "cache_size" in info
        assert "redis_enabled" in info
        assert "ttl_seconds" in info
        assert "keys_cached" in info

        assert isinstance(info["cache_size"], int)
        assert isinstance(info["redis_enabled"], bool)
        assert isinstance(info["ttl_seconds"], int)
        assert isinstance(info["keys_cached"], list)

    def test_cache_info_with_entries(self) -> None:
        """Test get_cache_info reflects cached entries."""
        cache = StatisticsCache()

        stats = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats)

        info = cache.get_cache_info()

        assert info["cache_size"] == 1
        assert isinstance(info["keys_cached"], list)
        # Check that the category key is in the cached keys
        assert any(MODEL_REFERENCE_CATEGORY.image_generation.value in key for key in info["keys_cached"])

    def test_on_category_invalidated_callback(self) -> None:
        """Test that the invalidation callback works correctly."""
        cache = StatisticsCache()

        stats = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            sfw_count=8,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats)
        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is not None

        cache._on_category_invalidated(MODEL_REFERENCE_CATEGORY.image_generation)

        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is None

    def test_cache_handles_multiple_categories(self) -> None:
        """Test that cache correctly handles multiple categories independently."""
        cache = StatisticsCache()

        stats_image = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            total_models=10,
            returned_models=10,
            offset=0,
            limit=None,
            nsfw_count=2,
            sfw_count=8,
            computed_at=int(time.time()),
        )

        stats_text = CategoryStatistics(
            category=MODEL_REFERENCE_CATEGORY.text_generation,
            total_models=20,
            returned_models=20,
            offset=0,
            limit=None,
            nsfw_count=0,
            sfw_count=20,
            computed_at=int(time.time()),
        )

        cache.set(MODEL_REFERENCE_CATEGORY.image_generation, stats_image)
        cache.set(MODEL_REFERENCE_CATEGORY.text_generation, stats_text)

        cache.invalidate(MODEL_REFERENCE_CATEGORY.image_generation)

        assert cache.get(MODEL_REFERENCE_CATEGORY.image_generation) is None
        assert cache.get(MODEL_REFERENCE_CATEGORY.text_generation) is not None

    @patch("horde_model_reference.analytics.statistics_cache.horde_model_reference_settings")
    def test_redis_disabled_uses_memory_only(self, mock_settings: Mock) -> None:
        """Test that cache uses in-memory only when Redis is disabled."""
        mock_redis = Mock()
        mock_redis.use_redis = False
        mock_settings.redis = mock_redis
        mock_settings.statistics_cache_ttl = 300

        cache = StatisticsCache()

        assert cache._redis_client is None
