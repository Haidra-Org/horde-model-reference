"""Caching layer for category statistics with Redis support.

Provides a singleton cache for CategoryStatistics that integrates with the backend
invalidation system. Automatically invalidates when model reference data changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.analytics.base_cache import RedisCache
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

if TYPE_CHECKING:
    from horde_model_reference.analytics.statistics import CategoryStatistics


class StatisticsCache(RedisCache["CategoryStatistics"]):
    """Singleton cache for category statistics.

    Integrates with existing backend invalidation system to automatically
    clear statistics when model data changes. Uses Redis for distributed
    caching when available, with in-memory fallback.

    Inherits from RedisCache[CategoryStatistics] for common caching infrastructure.
    """

    def _get_cache_key_prefix(self) -> str:
        """Get the Redis key prefix for statistics cache.

        Returns:
            Redis key prefix string.
        """
        return f"{horde_model_reference_settings.redis.key_prefix}:stats"

    def _get_ttl(self) -> int:
        """Get the TTL in seconds for statistics cache entries.

        Returns:
            TTL in seconds from settings.
        """
        return horde_model_reference_settings.statistics_cache_ttl

    def _get_model_class(self) -> type[CategoryStatistics]:
        """Get the Pydantic model class for deserialization.

        Returns:
            CategoryStatistics class.
        """
        from horde_model_reference.analytics.statistics import CategoryStatistics

        return CategoryStatistics

    def _register_invalidation_callback(self) -> None:
        """Register callback with ModelReferenceManager backend for automatic invalidation."""
        try:
            from horde_model_reference import ModelReferenceManager

            manager = ModelReferenceManager()
            if hasattr(manager.backend, "register_invalidation_callback"):
                manager.backend.register_invalidation_callback(self._on_category_invalidated)
                logger.info("StatisticsCache registered invalidation callback with backend")
            else:
                logger.warning(f"Backend {type(manager.backend).__name__} does not support invalidation callbacks")
        except Exception as e:
            logger.warning(f"Failed to register invalidation callback: {e}")
            logger.info("Statistics cache will rely on TTL-based expiration only")

    def _on_category_invalidated(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Invalidate statistics cache when model reference data changes.

        Invalidates both grouped and ungrouped variants for the category.

        Args:
            category: The category that was invalidated.
        """
        logger.debug(f"Invalidating statistics cache for category: {category}")
        self.invalidate(category, grouped=None)  # Invalidate both variants
