"""Generic base cache class for analytics results with Redis support.

Provides a thread-safe singleton cache that can store typed Pydantic models
with Redis distributed caching and in-memory fallback.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

if TYPE_CHECKING:
    import redis

T = TypeVar("T", bound=BaseModel)


class RedisCache(ABC, Generic[T]):
    """Generic base class for Redis-backed singleton caches.

    Provides common caching infrastructure with Redis distributed caching
    and in-memory fallback. Thread-safe with RLock pattern.

    Subclasses must implement:
    - _get_cache_key_prefix(): Return the Redis key prefix for this cache type
    - _get_ttl(): Return the TTL in seconds for cache entries
    - _get_model_class(): Return the Pydantic model class for deserialization
    - _register_invalidation_callback(): Set up automatic invalidation hooks

    Type parameter T must be a Pydantic BaseModel subclass.
    """

    _instance: RedisCache[T] | None = None
    _lock: RLock = RLock()

    _cache: dict[str, T]
    _timestamps: dict[str, float]
    _redis_client: redis.Redis[bytes] | None
    _redis_key_prefix: str
    _ttl: int

    def __new__(cls) -> RedisCache[T]:
        """Singleton pattern matching ModelReferenceManager."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initialize caching infrastructure."""
        self._cache = {}
        self._timestamps = {}
        self._redis_client = None
        self._redis_key_prefix = self._get_cache_key_prefix()
        self._ttl = self._get_ttl()

        logger.debug(f"Initializing {self.__class__.__name__} with TTL={self._ttl}s")

        # Try to connect to Redis if configured
        if horde_model_reference_settings.redis.use_redis:
            try:
                import redis

                self._redis_client = redis.from_url(
                    horde_model_reference_settings.redis.url,
                    socket_timeout=horde_model_reference_settings.redis.socket_timeout,
                    decode_responses=False,
                )
                self._redis_client.ping()
                logger.info(f"{self.__class__.__name__} using Redis for distributed caching")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for {self.__class__.__name__}: {e}")
                logger.info(f"{self.__class__.__name__} falling back to in-memory cache")
                self._redis_client = None
        else:
            logger.info(f"{self.__class__.__name__} using in-memory cache (Redis disabled)")

        # Register invalidation callback
        self._register_invalidation_callback()

    @abstractmethod
    def _get_cache_key_prefix(self) -> str:
        """Get the Redis key prefix for this cache type.

        Returns:
            Redis key prefix string (e.g., "horde:stats" or "horde:audit").
        """
        ...

    @abstractmethod
    def _get_ttl(self) -> int:
        """Get the TTL in seconds for cache entries.

        Returns:
            TTL in seconds.
        """
        ...

    @abstractmethod
    def _get_model_class(self) -> type[T]:
        """Get the Pydantic model class for deserialization.

        Returns:
            Pydantic model class (e.g., CategoryStatistics or CategoryAuditResponse).
        """
        ...

    @abstractmethod
    def _register_invalidation_callback(self) -> None:
        """Register callback with ModelReferenceManager for automatic invalidation.

        Should call manager.backend.register_invalidation_callback() if available.
        """
        ...

    def _build_cache_key(self, category: MODEL_REFERENCE_CATEGORY, grouped: bool = False) -> str:
        """Build cache key from category and grouping state.

        Args:
            category: The model reference category.
            grouped: Whether this is for grouped text models.

        Returns:
            Cache key string.
        """
        group_suffix = ":grouped=true" if grouped else ":grouped=false"
        return f"{category.value}{group_suffix}"

    def _get_redis_key(self, cache_key: str) -> str:
        """Generate Redis key from cache key.

        Args:
            cache_key: The cache key (category + grouping state).

        Returns:
            Full Redis key string with prefix.
        """
        return f"{self._redis_key_prefix}:{cache_key}"

    def get(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        grouped: bool = False,
    ) -> T | None:
        """Get cached result for a category.

        Checks Redis first (if available), then in-memory cache.
        Returns None if no valid cache entry exists.

        Args:
            category: The model reference category.
            grouped: Whether to get grouped text models variant.

        Returns:
            Cached result or None if not cached or expired.
        """
        cache_key = self._build_cache_key(category, grouped)

        # Try Redis first
        if self._redis_client:
            try:
                redis_key = self._get_redis_key(cache_key)
                cached_bytes = self._redis_client.get(redis_key)

                if cached_bytes:
                    model_class = self._get_model_class()
                    cached_dict = json.loads(cached_bytes.decode("utf-8"))
                    result = model_class(**cached_dict)
                    logger.debug(f"{self.__class__.__name__} cache hit (Redis): {cache_key}")
                    return result
            except Exception as e:
                logger.warning(f"Failed to get from Redis for {cache_key}: {e}")

        # Try in-memory cache
        with self._lock:
            if cache_key in self._cache:
                age = time.time() - self._timestamps.get(cache_key, 0)
                if age < self._ttl:
                    logger.debug(f"{self.__class__.__name__} cache hit (memory): {cache_key}")
                    return self._cache[cache_key]
                logger.debug(f"{self.__class__.__name__} cache expired (memory): {cache_key}, age={age:.1f}s")
                self._cache.pop(cache_key, None)
                self._timestamps.pop(cache_key, None)

        logger.debug(f"{self.__class__.__name__} cache miss: {cache_key}")
        return None

    def set(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        result: T,
        grouped: bool = False,
    ) -> None:
        """Store result in cache.

        Stores in both Redis (if available) and in-memory cache.

        Args:
            category: The model reference category.
            result: The computed result to cache.
            grouped: Whether this is the grouped text models variant.
        """
        cache_key = self._build_cache_key(category, grouped)

        # Store in Redis
        if self._redis_client:
            try:
                redis_key = self._get_redis_key(cache_key)
                serialized = result.model_dump_json()
                self._redis_client.setex(redis_key, self._ttl, serialized)
                logger.debug(f"Stored in Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to store in Redis for {cache_key}: {e}")

        # Store in-memory (always, as fallback)
        with self._lock:
            self._cache[cache_key] = result
            self._timestamps[cache_key] = time.time()
            logger.debug(f"Stored in memory: {cache_key}")

    def invalidate(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        grouped: bool | None = None,
    ) -> None:
        """Invalidate cached results for a category.

        Removes from both Redis and in-memory cache. If grouped is None,
        invalidates both grouped and ungrouped variants.

        Args:
            category: The model reference category to invalidate.
            grouped: Whether to invalidate grouped variant (None = both).
        """
        # Determine which variants to invalidate
        variants = [False, True] if grouped is None else [grouped]

        for variant in variants:
            cache_key = self._build_cache_key(category, variant)
            logger.debug(f"Invalidating cache: {cache_key}")

            # Invalidate Redis
            if self._redis_client:
                try:
                    redis_key = self._get_redis_key(cache_key)
                    deleted_count = self._redis_client.delete(redis_key)
                    if deleted_count > 0:
                        logger.debug(f"Deleted Redis key: {redis_key}")
                except Exception as e:
                    logger.warning(f"Failed to delete from Redis for {cache_key}: {e}")

            # Invalidate in-memory
            with self._lock:
                removed = self._cache.pop(cache_key, None) is not None
                self._timestamps.pop(cache_key, None)
                if removed:
                    logger.debug(f"Removed from memory cache: {cache_key}")

    def clear_all(self) -> None:
        """Clear all cached results.

        Useful for testing or when a global cache reset is needed.
        """
        logger.info(f"Clearing all {self.__class__.__name__} entries")

        # Clear Redis
        if self._redis_client:
            try:
                for category in MODEL_REFERENCE_CATEGORY:
                    for grouped in [False, True]:
                        cache_key = self._build_cache_key(category, grouped)
                        redis_key = self._get_redis_key(cache_key)
                        self._redis_client.delete(redis_key)
                logger.debug("Cleared all Redis keys")
            except Exception as e:
                logger.warning(f"Failed to clear Redis cache: {e}")

        # Clear in-memory
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.debug("Cleared in-memory cache")

    def get_cache_info(self) -> dict[str, int | float | bool | list[str]]:
        """Get information about the current cache state.

        Returns:
            Dictionary with cache statistics including size, Redis status, TTL.
        """
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "redis_enabled": self._redis_client is not None,
                "ttl_seconds": self._ttl,
                "keys_cached": list(self._cache.keys()),
            }
