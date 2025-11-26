"""Generic base cache class for analytics results with Redis support.

Provides a thread-safe singleton cache that can store typed Pydantic models
with Redis distributed caching and in-memory fallback. Supports stale-while-revalidate
pattern when cache hydration is enabled.
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

    When cache hydration is enabled (via settings), implements stale-while-revalidate:
    - Normal TTL controls when background hydration refreshes the cache
    - Stale TTL controls maximum age before returning None (forcing computation)
    - Clients always receive cached data immediately while hydration runs in background

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

    def _build_cache_key(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        grouped: bool = False,
        include_backend_variations: bool = False,
    ) -> str:
        """Build cache key from category and options.

        Args:
            category: The model reference category.
            grouped: Whether this is for grouped text models.
            include_backend_variations: Whether backend variations are included.

        Returns:
            Cache key string.
        """
        group_suffix = ":grouped=true" if grouped else ":grouped=false"
        variations_suffix = ":variations=true" if include_backend_variations else ""
        return f"{category.value}{group_suffix}{variations_suffix}"

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
        include_backend_variations: bool = False,
        allow_stale: bool | None = None,
    ) -> T | None:
        """Get cached result for a category.

        Checks Redis first (if available), then in-memory cache.

        When cache hydration is enabled (settings.cache_hydration_enabled=True) and
        allow_stale is True (or None with hydration enabled), implements stale-while-revalidate:
        - Returns cached data even if past normal TTL
        - Only returns None if data exceeds stale_ttl (default 1 hour)
        - Background hydration is expected to refresh data before stale_ttl

        Args:
            category: The model reference category.
            grouped: Whether to get grouped text models variant.
            include_backend_variations: Whether backend variations are included.
            allow_stale: Whether to return stale data beyond normal TTL.
                If None, defaults to True when hydration is enabled, False otherwise.

        Returns:
            Cached result or None if not cached or expired beyond stale TTL.
        """
        cache_key = self._build_cache_key(category, grouped, include_backend_variations)

        # Determine stale behavior
        hydration_enabled = horde_model_reference_settings.cache_hydration_enabled
        effective_allow_stale = allow_stale if allow_stale is not None else hydration_enabled
        stale_ttl = horde_model_reference_settings.cache_hydration_stale_ttl_seconds

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

                # Fresh data - always return
                if age < self._ttl:
                    logger.debug(f"{self.__class__.__name__} cache hit (memory): {cache_key}")
                    return self._cache[cache_key]

                # Stale data - return if stale allowed and within stale TTL
                if effective_allow_stale and age < stale_ttl:
                    logger.debug(
                        f"{self.__class__.__name__} returning stale data (memory): {cache_key}, "
                        f"age={age:.1f}s (TTL={self._ttl}s, stale_ttl={stale_ttl}s)"
                    )
                    return self._cache[cache_key]

                # Data too old - remove and return None
                logger.debug(
                    f"{self.__class__.__name__} cache expired (memory): {cache_key}, "
                    f"age={age:.1f}s (stale_allowed={effective_allow_stale})"
                )
                self._cache.pop(cache_key, None)
                self._timestamps.pop(cache_key, None)

        logger.debug(f"{self.__class__.__name__} cache miss: {cache_key}")
        return None

    def is_fresh(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        grouped: bool = False,
        include_backend_variations: bool = False,
    ) -> bool:
        """Check if cached data is fresh (within normal TTL).

        Useful for determining if background hydration should run.

        Args:
            category: The model reference category.
            grouped: Whether to check grouped text models variant.
            include_backend_variations: Whether backend variations are included.

        Returns:
            True if fresh data exists within TTL, False otherwise.
        """
        cache_key = self._build_cache_key(category, grouped, include_backend_variations)

        # Check Redis TTL
        if self._redis_client:
            try:
                redis_key = self._get_redis_key(cache_key)
                ttl = self._redis_client.ttl(redis_key)
                if ttl > 0:
                    return True
            except Exception as e:
                logger.warning(f"Failed to check Redis TTL for {cache_key}: {e}")

        # Check in-memory
        with self._lock:
            if cache_key in self._cache:
                age = time.time() - self._timestamps.get(cache_key, 0)
                return age < self._ttl

        return False

    def set(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        result: T,
        grouped: bool = False,
        include_backend_variations: bool = False,
    ) -> None:
        """Store result in cache.

        Stores in both Redis (if available) and in-memory cache.

        Args:
            category: The model reference category.
            result: The computed result to cache.
            grouped: Whether this is the grouped text models variant.
            include_backend_variations: Whether backend variations are included.
        """
        cache_key = self._build_cache_key(category, grouped, include_backend_variations)

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
        include_backend_variations: bool | None = None,
    ) -> None:
        """Invalidate cached results for a category.

        Removes from both Redis and in-memory cache. If grouped is None,
        invalidates all grouped/ungrouped variants. If include_backend_variations
        is None, invalidates all variation states.

        Args:
            category: The model reference category to invalidate.
            grouped: Whether to invalidate grouped variant (None = both).
            include_backend_variations: Whether to invalidate variation states (None = both).
        """
        # Determine which variants to invalidate
        grouped_variants = [False, True] if grouped is None else [grouped]
        variations_variants = [False, True] if include_backend_variations is None else [include_backend_variations]

        for gv in grouped_variants:
            for vv in variations_variants:
                cache_key = self._build_cache_key(category, gv, vv)
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
                        for variations in [False, True]:
                            cache_key = self._build_cache_key(category, grouped, variations)
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
