"""Caching layer for category deletion risk results with Redis support.

Provides a singleton cache for CategoryDeletionRiskResponse that integrates with the backend
invalidation system. Automatically invalidates when model reference data changes.
"""

from __future__ import annotations

from typing import ClassVar

from loguru import logger

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.analytics.base_cache import RedisCache
from horde_model_reference.analytics.deletion_risk_analysis import CategoryDeletionRiskResponse
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class DeletionRiskCache(RedisCache[CategoryDeletionRiskResponse]):
    """Singleton cache for category deletion risk results.

    Integrates with existing backend invalidation system to automatically
    clear deletion risk results when model data changes. Uses Redis for distributed
    caching when available, with in-memory fallback.

    Inherits from RedisCache[CategoryDeletionRiskResponse] for common caching infrastructure.
    """

    _instance: ClassVar[DeletionRiskCache | None] = None

    def _get_cache_key_prefix(self) -> str:
        """Get the Redis key prefix for deletion risk cache.

        Returns:
            Redis key prefix string.

        """
        return f"{horde_model_reference_settings.redis.key_prefix}:deletion_risk"

    def _get_ttl(self) -> int:
        """Get the TTL in seconds for deletion risk cache entries.

        Returns:
            TTL in seconds from settings.

        """
        return horde_model_reference_settings.deletion_risk_cache_ttl

    def _get_model_class(self) -> type[CategoryDeletionRiskResponse]:
        """Get the Pydantic model class for deserialization.

        Returns:
            CategoryDeletionRiskResponse class.

        """
        from horde_model_reference.analytics.deletion_risk_analysis import CategoryDeletionRiskResponse

        return CategoryDeletionRiskResponse

    def _register_invalidation_callback(self) -> None:
        """Register callback with ModelReferenceManager backend for automatic invalidation."""
        try:
            from horde_model_reference import ModelReferenceManager

            manager = ModelReferenceManager()
            if hasattr(manager.backend, "register_invalidation_callback"):
                manager.backend.register_invalidation_callback(self._on_category_invalidated)
                logger.info("DeletionRiskCache registered invalidation callback with backend")
            else:
                logger.warning(f"Backend {type(manager.backend).__name__} does not support invalidation callbacks")
        except Exception as e:
            logger.warning(f"Failed to register invalidation callback: {e}")
            logger.info("Deletion risk cache will rely on TTL-based expiration only")

    def _on_category_invalidated(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Invalidate deletion risk cache when model reference data changes.

        Invalidates both grouped and ungrouped variants for the category.

        Args:
            category: The category that was invalidated.

        """
        logger.debug(f"Invalidating deletion risk cache for category: {category}")
        self.invalidate(category, grouped=None)  # Invalidate both variants
