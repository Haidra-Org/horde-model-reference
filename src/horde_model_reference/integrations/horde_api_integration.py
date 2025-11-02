"""Integration with AI Horde public API for fetching runtime model data.

This module provides a singleton class for fetching and caching model status, statistics,
and worker information from the AI Horde public API. It integrates with the existing Redis
infrastructure when available and falls back to in-memory caching otherwise.
"""

from __future__ import annotations

import asyncio
import json
import time
from threading import RLock
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from horde_model_reference.integrations.horde_api_models import (
    HordeModelState,
    HordeModelStatsResponse,
    HordeModelStatus,
    HordeModelType,
    HordeWorker,
    IndexedHordeModelStats,
    IndexedHordeModelStatus,
    IndexedHordeWorkers,
)

if TYPE_CHECKING:
    import redis


class HordeAPIIntegration:
    """Singleton for Horde API data fetching and caching.

    Integrates with existing Redis backend when available in PRIMARY mode.
    Falls back to in-memory TTL cache for REPLICA mode or non-Redis PRIMARY.

    This class follows the same patterns as ModelReferenceManager:
    - Singleton pattern with thread-safe initialization
    - Settings-based configuration
    - Redis-aware caching with in-memory fallback
    - Consistent key naming with existing backend
    """

    _instance: HordeAPIIntegration | None = None
    _lock: RLock = RLock()

    # In-memory cache structures (fallback when Redis unavailable)
    _status_cache: dict[HordeModelType, list[HordeModelStatus]]
    _stats_cache: dict[HordeModelType, HordeModelStatsResponse]
    _workers_cache: dict[HordeModelType | None, list[HordeWorker]]
    _cache_timestamps: dict[str, float]

    # Redis integration (when available)
    _redis_client: redis.Redis[bytes] | None
    _redis_key_prefix: str
    _ttl: int
    _base_url: str
    _timeout: int

    def __new__(cls) -> HordeAPIIntegration:
        """Singleton pattern matching ModelReferenceManager."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self) -> None:
        """Initialize caching infrastructure, detect Redis availability."""
        from horde_model_reference import ai_horde_worker_settings, horde_model_reference_settings

        # Load settings
        self._ttl = horde_model_reference_settings.horde_api_cache_ttl
        self._base_url = str(ai_horde_worker_settings.ai_horde_url).rstrip("/") + "/v2"
        self._timeout = horde_model_reference_settings.horde_api_timeout

        # Initialize in-memory caches
        self._status_cache = {}
        self._stats_cache = {}
        self._workers_cache = {}
        self._cache_timestamps = {}

        # Try to connect to Redis if configured
        self._redis_client = None
        if horde_model_reference_settings.redis.use_redis:
            try:
                import redis

                self._redis_client = redis.from_url(
                    horde_model_reference_settings.redis.url,
                    socket_timeout=horde_model_reference_settings.redis.socket_timeout,
                    decode_responses=False,
                )
                self._redis_key_prefix = f"{horde_model_reference_settings.redis.key_prefix}:horde_api"

                # Test connection
                self._redis_client.ping()
                logger.info("HordeAPIIntegration using Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for HordeAPI cache, using in-memory: {e}")
                self._redis_client = None
        else:
            logger.info("HordeAPIIntegration using in-memory cache")

    def _get_cache_key(self, cache_type: str, model_type: HordeModelType | None = None) -> str:
        """Generate cache key for Redis or in-memory dict.

        Args:
            cache_type: Type of cache (status, stats, workers)
            model_type: Model type (image or text), None for all workers

        Returns:
            Cache key string
        """
        if model_type is None:
            return f"{cache_type}:all"
        return f"{cache_type}:{model_type}"

    def _get_redis_key(self, cache_key: str) -> str:
        """Generate full Redis key with prefix.

        Args:
            cache_key: Base cache key

        Returns:
            Full Redis key with prefix
        """
        return f"{self._redis_key_prefix}:{cache_key}"

    def _get_from_redis(self, cache_key: str) -> bytes | None:
        """Get data from Redis cache.

        Args:
            cache_key: Cache key to retrieve

        Returns:
            Cached data as bytes, or None if not found or error
        """
        if not self._redis_client:
            return None

        try:
            redis_key = self._get_redis_key(cache_key)
            data = self._redis_client.get(redis_key)
            if data:
                logger.debug(f"Redis cache hit: {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Redis get failed for {cache_key}: {e}")
            return None

    def _store_in_redis(self, cache_key: str, data: bytes) -> None:
        """Store data in Redis cache with TTL.

        Args:
            cache_key: Cache key to store under
            data: Data to store (serialized bytes)
        """
        if not self._redis_client:
            return

        try:
            redis_key = self._get_redis_key(cache_key)
            self._redis_client.setex(redis_key, self._ttl, data)
            logger.debug(f"Stored in Redis cache: {cache_key} (TTL: {self._ttl}s)")
        except Exception as e:
            logger.warning(f"Failed to store in Redis: {e}")

    async def get_model_status(
        self,
        model_type: HordeModelType,
        min_count: int | None = None,
        model_state: HordeModelState = "known",
        force_refresh: bool = False,
    ) -> list[HordeModelStatus]:
        """Fetch model status with caching (Redis if available, else in-memory).

        Args:
            model_type: Type of models to fetch (image or text)
            min_count: Minimum worker count filter
            model_state: Model state filter (known, custom, all)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            List of model status objects
        """
        cache_key = self._get_cache_key("status", model_type)

        # Try Redis first
        if not force_refresh:
            cached_bytes = self._get_from_redis(cache_key)
            if cached_bytes:
                try:
                    cached_data = json.loads(cached_bytes)
                    return [HordeModelStatus.model_validate(item) for item in cached_data]
                except Exception as e:
                    logger.warning(f"Failed to deserialize Redis cache for {cache_key}: {e}")

        # Try in-memory cache
        if not force_refresh:
            with self._lock:
                if cache_key in self._cache_timestamps:
                    age = time.time() - self._cache_timestamps[cache_key]
                    if age < self._ttl and model_type in self._status_cache:
                        logger.debug(f"In-memory cache hit: {cache_key}")
                        return self._status_cache[model_type]

        # Fetch from Horde API
        logger.debug(f"Fetching from Horde API: {cache_key}")
        data = await self._fetch_status_from_api(model_type, min_count, model_state)

        # Store in cache
        self._store_status_in_cache(cache_key, model_type, data)

        return data

    async def _fetch_status_from_api(
        self,
        model_type: HordeModelType,
        min_count: int | None = None,
        model_state: HordeModelState = "known",
    ) -> list[HordeModelStatus]:
        """Fetch model status from Horde API.

        Args:
            model_type: Type of models to fetch
            min_count: Minimum worker count filter
            model_state: Model state filter

        Returns:
            List of model status objects

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        url = f"{self._base_url}/status/models"
        params: dict[str, str] = {"type": model_type, "model_state": model_state}
        if min_count is not None:
            params["min_count"] = str(min_count)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return [HordeModelStatus.model_validate(item) for item in data]

    def _store_status_in_cache(
        self,
        cache_key: str,
        model_type: HordeModelType,
        data: list[HordeModelStatus],
    ) -> None:
        """Store status data in both Redis (if available) and in-memory cache.

        Args:
            cache_key: Cache key to store under
            model_type: Model type
            data: Status data to cache
        """
        # Serialize for Redis
        serialized = json.dumps([item.model_dump() for item in data])
        self._store_in_redis(cache_key, serialized.encode("utf-8"))

        # Store in-memory (always, as fallback)
        with self._lock:
            self._status_cache[model_type] = data
            self._cache_timestamps[cache_key] = time.time()
            logger.debug(f"Stored in memory cache: {cache_key}")

    async def get_model_stats(
        self,
        model_type: HordeModelType,
        model_state: HordeModelState = "known",
        force_refresh: bool = False,
    ) -> HordeModelStatsResponse:
        """Fetch model statistics with caching.

        Args:
            model_type: Type of models to fetch stats for
            model_state: Model state filter
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            Model statistics response
        """
        cache_key = self._get_cache_key("stats", model_type)

        # Try Redis first
        if not force_refresh:
            cached_bytes = self._get_from_redis(cache_key)
            if cached_bytes:
                try:
                    cached_data = json.loads(cached_bytes)
                    return HordeModelStatsResponse.model_validate(cached_data)
                except Exception as e:
                    logger.warning(f"Failed to deserialize Redis cache for {cache_key}: {e}")

        # Try in-memory cache
        if not force_refresh:
            with self._lock:
                if cache_key in self._cache_timestamps:
                    age = time.time() - self._cache_timestamps[cache_key]
                    if age < self._ttl and model_type in self._stats_cache:
                        logger.debug(f"In-memory cache hit: {cache_key}")
                        return self._stats_cache[model_type]

        # Fetch from Horde API
        logger.debug(f"Fetching from Horde API: {cache_key}")
        data = await self._fetch_stats_from_api(model_type, model_state)

        # Store in cache
        self._store_stats_in_cache(cache_key, model_type, data)

        return data

    async def _fetch_stats_from_api(
        self,
        model_type: HordeModelType,
        model_state: HordeModelState = "known",
    ) -> HordeModelStatsResponse:
        """Fetch model stats from Horde API.

        Args:
            model_type: Type of models to fetch
            model_state: Model state filter

        Returns:
            Model statistics response

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        endpoint = "stats/img/models" if model_type == "image" else "stats/text/models"
        url = f"{self._base_url}/{endpoint}"
        params = {"model_state": model_state}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return HordeModelStatsResponse.model_validate(data)

    def _store_stats_in_cache(
        self,
        cache_key: str,
        model_type: HordeModelType,
        data: HordeModelStatsResponse,
    ) -> None:
        """Store stats data in both Redis (if available) and in-memory cache.

        Args:
            cache_key: Cache key to store under
            model_type: Model type
            data: Stats data to cache
        """
        # Serialize for Redis
        serialized = json.dumps(data.model_dump())
        self._store_in_redis(cache_key, serialized.encode("utf-8"))

        # Store in-memory (always, as fallback)
        with self._lock:
            self._stats_cache[model_type] = data
            self._cache_timestamps[cache_key] = time.time()
            logger.debug(f"Stored in memory cache: {cache_key}")

    async def get_workers(
        self,
        model_type: HordeModelType | None = None,
        force_refresh: bool = False,
    ) -> list[HordeWorker]:
        """Fetch workers with caching.

        Args:
            model_type: Type of workers to fetch (image, text, or None for all)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            List of worker objects
        """
        cache_key = self._get_cache_key("workers", model_type)

        # Try Redis first
        if not force_refresh:
            cached_bytes = self._get_from_redis(cache_key)
            if cached_bytes:
                try:
                    cached_data = json.loads(cached_bytes)
                    return [HordeWorker.model_validate(item) for item in cached_data]
                except Exception as e:
                    logger.warning(f"Failed to deserialize Redis cache for {cache_key}: {e}")

        # Try in-memory cache
        if not force_refresh:
            with self._lock:
                if cache_key in self._cache_timestamps:
                    age = time.time() - self._cache_timestamps[cache_key]
                    if age < self._ttl and model_type in self._workers_cache:
                        logger.debug(f"In-memory cache hit: {cache_key}")
                        return self._workers_cache[model_type]

        # Fetch from Horde API
        logger.debug(f"Fetching from Horde API: {cache_key}")
        data = await self._fetch_workers_from_api(model_type)

        # Store in cache
        self._store_workers_in_cache(cache_key, model_type, data)

        return data

    async def _fetch_workers_from_api(
        self,
        model_type: HordeModelType | None = None,
    ) -> list[HordeWorker]:
        """Fetch workers from Horde API.

        Args:
            model_type: Type of workers to fetch (or None for all)

        Returns:
            List of worker objects

        Raises:
            httpx.HTTPError: On network or HTTP errors
        """
        url = f"{self._base_url}/workers"
        params = {}
        if model_type is not None:
            params["type"] = model_type

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Fetched {len(data)} workers from {url} with params {params}")
            return [HordeWorker.model_validate(item) for item in data]

    def _store_workers_in_cache(
        self,
        cache_key: str,
        model_type: HordeModelType | None,
        data: list[HordeWorker],
    ) -> None:
        """Store workers data in both Redis (if available) and in-memory cache.

        Args:
            cache_key: Cache key to store under
            model_type: Model type (or None)
            data: Workers data to cache
        """
        # Serialize for Redis
        serialized = json.dumps([item.model_dump() for item in data])
        self._store_in_redis(cache_key, serialized.encode("utf-8"))

        # Store in-memory (always, as fallback)
        with self._lock:
            self._workers_cache[model_type] = data
            self._cache_timestamps[cache_key] = time.time()
            logger.debug(f"Stored in memory cache: {cache_key}")

    async def get_combined_data(
        self,
        model_type: HordeModelType,
        include_workers: bool = True,
        force_refresh: bool = False,
    ) -> tuple[list[HordeModelStatus], HordeModelStatsResponse, list[HordeWorker] | None]:
        """Fetch all data for a model type in parallel.

        Args:
            model_type: Type of models to fetch data for
            include_workers: Whether to fetch workers (can be slow)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            Tuple of (status, stats, workers)
        """
        status_task: asyncio.Task[list[HordeModelStatus]] = asyncio.create_task(
            self.get_model_status(model_type, force_refresh=force_refresh)
        )
        stats_task: asyncio.Task[HordeModelStatsResponse] = asyncio.create_task(
            self.get_model_stats(model_type, force_refresh=force_refresh)
        )

        workers_task: asyncio.Task[list[HordeWorker]] | None = None
        if include_workers:
            workers_task = asyncio.create_task(self.get_workers(model_type, force_refresh=force_refresh))

        try:
            status = await status_task
            stats = await stats_task
            workers = await workers_task if workers_task is not None else None
        except Exception:
            for t in (status_task, stats_task, workers_task):
                if t is not None and not t.done():
                    t.cancel()
            raise

        return status, stats, workers

    async def get_model_status_indexed(
        self,
        model_type: HordeModelType,
        min_count: int | None = None,
        model_state: HordeModelState = "known",
        force_refresh: bool = False,
    ) -> IndexedHordeModelStatus:
        """Fetch model status as pre-indexed dictionary for O(1) lookups.

        **Performance**: Returns IndexedHordeModelStatus which provides O(1) lookups
        by model name instead of O(n) list iteration. Use this when merging with
        model reference data for optimal performance.

        Args:
            model_type: Type of models to fetch (image or text)
            min_count: Minimum worker count filter
            model_state: Model state filter (known, custom, all)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            IndexedHordeModelStatus with O(1) lookup by model name
        """
        status_list = await self.get_model_status(model_type, min_count, model_state, force_refresh)
        return IndexedHordeModelStatus(status_list)

    async def get_model_stats_indexed(
        self,
        model_type: HordeModelType,
        model_state: HordeModelState = "known",
        force_refresh: bool = False,
    ) -> IndexedHordeModelStats:
        """Fetch model statistics as pre-indexed dictionary for O(1) lookups.

        **Performance**: Returns IndexedHordeModelStats which provides O(1) lookups
        by model name instead of O(n) dict iteration. Use this when merging with
        model reference data for optimal performance.

        Args:
            model_type: Type of models to fetch stats for
            model_state: Model state filter
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            IndexedHordeModelStats with O(1) lookup by model name
        """
        stats = await self.get_model_stats(model_type, model_state, force_refresh)
        return IndexedHordeModelStats(stats)

    async def get_workers_indexed(
        self,
        model_type: HordeModelType | None = None,
        force_refresh: bool = False,
    ) -> IndexedHordeWorkers:
        """Fetch workers as pre-indexed dictionary for O(1) lookups by model name.

        **Performance**: Returns IndexedHordeWorkers which provides O(1) lookups
        by model name instead of O(w*m) iteration over workers and their models.
        Use this when merging with model reference data for optimal performance.

        Args:
            model_type: Type of workers to fetch (image, text, or None for all)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            IndexedHordeWorkers with O(1) lookup by model name
        """
        workers_list = await self.get_workers(model_type, force_refresh)
        return IndexedHordeWorkers(workers_list)

    async def get_combined_data_indexed(
        self,
        model_type: HordeModelType,
        include_workers: bool = True,
        force_refresh: bool = False,
    ) -> tuple[IndexedHordeModelStatus, IndexedHordeModelStats, IndexedHordeWorkers | None]:
        """Fetch all data for a model type in parallel, pre-indexed for optimal merging.

        **Performance**: Returns pre-indexed data structures that provide O(1) lookups.
        This is the recommended method for fetching Horde data when merging with
        model reference data, as it eliminates the O(s+t+w*p) indexing overhead
        from the merge operation.

        Args:
            model_type: Type of models to fetch data for
            include_workers: Whether to fetch workers (can be slow)
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            Tuple of (indexed_status, indexed_stats, indexed_workers)
        """
        status, stats, workers = await self.get_combined_data(model_type, include_workers, force_refresh)

        indexed_status = IndexedHordeModelStatus(status)
        indexed_stats = IndexedHordeModelStats(stats)
        indexed_workers = IndexedHordeWorkers(workers) if workers else None

        return indexed_status, indexed_stats, indexed_workers

    def invalidate_cache(self, model_type: HordeModelType | None = None) -> None:
        """Invalidate cached data for a model type.

        Args:
            model_type: Model type to invalidate, or None to invalidate all
        """
        with self._lock:
            if model_type is None:
                # Invalidate all
                self._status_cache.clear()
                self._stats_cache.clear()
                self._workers_cache.clear()
                self._cache_timestamps.clear()
                logger.debug("Invalidated all HordeAPI caches")

                # Invalidate Redis if available
                if self._redis_client:
                    try:
                        pattern = f"{self._redis_key_prefix}:*"
                        keys = self._redis_client.keys(pattern)
                        if keys:
                            self._redis_client.delete(*keys)
                            logger.debug(f"Deleted {len(keys)} Redis keys matching {pattern}")
                    except Exception as e:
                        logger.warning(f"Failed to invalidate Redis keys: {e}")
            else:
                # Invalidate specific model type
                self._status_cache.pop(model_type, None)
                self._stats_cache.pop(model_type, None)
                self._workers_cache.pop(model_type, None)

                for key in [
                    self._get_cache_key("status", model_type),
                    self._get_cache_key("stats", model_type),
                    self._get_cache_key("workers", model_type),
                ]:
                    self._cache_timestamps.pop(key, None)

                    # Invalidate Redis if available
                    if self._redis_client:
                        try:
                            redis_key = self._get_redis_key(key)
                            self._redis_client.delete(redis_key)
                            logger.debug(f"Deleted Redis key: {redis_key}")
                        except Exception as e:
                            logger.warning(f"Failed to delete Redis key {key}: {e}")

                logger.debug(f"Invalidated HordeAPI cache for {model_type}")
