"""Redis-based distributed cache backend for PRIMARY mode.

This backend wraps a file-based backend and adds distributed caching via Redis.
It's designed for PRIMARY mode multi-worker deployments where multiple FastAPI
workers need to share cached model reference data.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
import redis.asyncio
from loguru import logger
from typing_extensions import override

from horde_model_reference import RedisSettings, ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_metadata import CategoryMetadata


class RedisBackend(ModelReferenceBackend):
    """Redis-backed distributed cache with file backend fallback.

    Architecture:
    - File backend is the source of truth
    - Redis provides distributed caching across multiple PRIMARY workers
    - On cache miss, reads from file backend and populates Redis
    - Pub/sub notifies other workers of cache invalidations
    - Only usable in PRIMARY mode
    """

    _file_backend: FileSystemBackend
    _redis_settings: RedisSettings
    _ttl: int

    _lock: RLock
    _sync_redis: redis.Redis[bytes]

    _pubsub: redis.client.PubSub
    _pubsub_thread: threading.Thread | None
    _pubsub_running: bool

    def __init__(
        self,
        *,
        file_backend: FileSystemBackend,
        redis_settings: RedisSettings,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Initialize Redis backend with filesystem backend.

        Args:
            file_backend: Filesystem backend to wrap.
            redis_settings: Redis connection settings.
            cache_ttl_seconds: TTL for cache entries.
                If None, uses redis_settings.ttl_seconds.

        Raises:
            ValueError: If file_backend is not in PRIMARY mode.
        """
        if file_backend.replicate_mode != ReplicateMode.PRIMARY:
            raise ValueError(
                "RedisBackend can only be used with a FileSystemBackend in PRIMARY mode. "
                "For REPLICA mode, use GitHubBackend or HTTPBackend."
            )

        super().__init__(mode=ReplicateMode.PRIMARY)

        self._file_backend = file_backend
        self._redis_settings = redis_settings
        self._ttl = redis_settings.ttl_seconds or cache_ttl_seconds or 60

        self._lock = RLock()

        try:
            self._sync_redis = self._create_sync_pool()
            logger.info(f"Redis connection established: {redis_settings.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        self._pubsub_running = False
        self._pubsub_thread: threading.Thread | None = None
        if redis_settings.use_pubsub:
            self._setup_pubsub()

    def _create_sync_pool(self) -> redis.Redis[bytes]:
        """Create synchronous Redis connection pool."""
        return redis.from_url(
            self._redis_settings.url,
            max_connections=self._redis_settings.pool_size,
            socket_timeout=self._redis_settings.socket_timeout,
            socket_connect_timeout=self._redis_settings.socket_connect_timeout,
            decode_responses=False,
        )

    def _category_key(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Generate Redis key for a category."""
        return f"{self._redis_settings.key_prefix}:category:{category.value}"

    def _legacy_metadata_key(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Generate Redis key for legacy metadata."""
        return f"{self._redis_settings.key_prefix}:meta:legacy:{category.value}"

    def _v2_metadata_key(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Generate Redis key for v2 metadata."""
        return f"{self._redis_settings.key_prefix}:meta:v2:{category.value}"

    def _invalidation_channel(self) -> str:
        """Get the Redis pub/sub channel for invalidations."""
        return f"{self._redis_settings.key_prefix}:invalidate"

    def _setup_pubsub(self) -> None:
        """Set up pub/sub for cache invalidation events."""
        try:
            self._pubsub = self._sync_redis.pubsub(ignore_subscribe_messages=True)
            channel = self._invalidation_channel()
            self._pubsub.subscribe(channel)

            self._pubsub_running = True
            self._pubsub_thread = threading.Thread(
                target=self._listen_for_invalidations,
                daemon=True,
                name="RedisBackend-PubSub",
            )
            self._pubsub_thread.start()
            logger.info(f"Redis pub/sub listening on {channel}")
        except Exception as e:
            logger.warning(f"Failed to setup Redis pub/sub: {e}")
            self._pubsub_running = False

    def _listen_for_invalidations(self) -> None:
        """Listen for cache invalidation events from other workers."""
        logger.debug("Redis pub/sub listener started")
        try:
            messages = self._pubsub.listen()  # type: ignore[no-untyped-call]
            if not isinstance(messages, Iterable):
                raise ValueError("Expected iterable from pubsub.listen()")
            for message in messages:
                if not self._pubsub_running:
                    break

                if message["type"] == "message":
                    try:
                        data = message["data"]
                        category_str = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                        category = MODEL_REFERENCE_CATEGORY(category_str)
                        logger.debug(f"Received invalidation for {category} from another worker")

                        key = self._category_key(category)
                        try:
                            self._sync_redis.delete(key)
                            logger.debug(f"Invalidated local Redis cache for {category}")
                        except Exception as delete_error:
                            logger.warning(f"Failed to invalidate Redis cache for {category}: {delete_error}")

                        self._file_backend.mark_stale(category)
                        self._notify_invalidation(category)

                    except Exception as e:
                        logger.warning(f"Failed to process invalidation message: {e}")
        except Exception as e:
            logger.error(f"Redis pub/sub listener error: {e}")
        finally:
            logger.debug("Redis pub/sub listener stopped")

    def _retry_redis_operation(
        self,
        operation: Callable[..., str | bool | bytes | int | None],
        *args: str | int | float | None,
        **kwargs: str | int | float | None,
    ) -> str | bool | int | bytes | None:
        """Retry a Redis operation with exponential backoff."""
        for attempt in range(self._redis_settings.retry_max_attempts):
            try:
                return operation(*args, **kwargs)
            except redis.ConnectionError as e:
                if attempt == self._redis_settings.retry_max_attempts - 1:
                    logger.error(f"Redis operation failed after {attempt + 1} attempts: {e}")
                    raise
                wait_time = self._redis_settings.retry_backoff_seconds * (2**attempt)
                logger.warning(f"Redis connection error, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
        return None

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch from Redis cache, fallback to file backend on miss.

        Args:
            category: The category to fetch.
            force_refresh: If True, bypass Redis cache and fetch from files.

        Returns:
            dict[str, Any] | None: The model reference data.
        """
        key = self._category_key(category)
        data: dict[str, Any] | None = None

        if not force_refresh:
            try:
                cached = self._retry_redis_operation(self._sync_redis.get, key)
                if cached:
                    if not isinstance(cached, str):
                        raise ValueError("Expected str from Redis")

                    data = json.loads(cached)
                    logger.debug(f"Redis cache hit for {category}")
                    return data
                logger.debug(f"Redis cache miss for {category}")
            except Exception as e:
                logger.warning(f"Redis fetch failed for {category}, falling back to file: {e}")

        data = self._file_backend.fetch_category(category, force_refresh=force_refresh)

        if data is not None:
            try:
                json_data = json.dumps(data)
                self._retry_redis_operation(self._sync_redis.setex, key, self._ttl, json_data)
                logger.debug(f"Populated Redis cache for {category}")
            except Exception as e:
                logger.warning(f"Failed to cache {category} in Redis: {e}")

        return data

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch all categories, using Redis cache where available."""
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = self.fetch_category(category, force_refresh=force_refresh)

        return result

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch from Redis, fallback to file backend.

        Args:
            category: The category to fetch.
            httpx_client: Optional shared HTTPX client for file backend.
            force_refresh: If True, bypass Redis cache.

        Returns:
            dict[str, Any] | None: The model reference data.
        """
        key = self._category_key(category)
        data: dict[str, Any] | None = None

        if not force_refresh:
            cached: bytes | None = None
            try:
                async with redis.asyncio.from_url(
                    self._redis_settings.url,
                    max_connections=self._redis_settings.pool_size,
                    socket_timeout=self._redis_settings.socket_timeout,
                    socket_connect_timeout=self._redis_settings.socket_connect_timeout,
                    decode_responses=False,
                ) as async_redis:
                    cached = await async_redis.get(key)

                if cached:
                    data = json.loads(cached)
                    logger.debug(f"Redis cache hit for {category} (async)")
                    return data
                logger.debug(f"Redis cache miss for {category} (async)")
            except Exception as e:
                logger.warning(f"Async Redis fetch failed for {category}, falling back to file: {e}")

        data = await self._file_backend.fetch_category_async(
            category,
            httpx_client=httpx_client,
            force_refresh=force_refresh,
        )

        if data is not None:
            try:
                async with redis.asyncio.from_url(
                    self._redis_settings.url,
                    max_connections=self._redis_settings.pool_size,
                    socket_timeout=self._redis_settings.socket_timeout,
                    socket_connect_timeout=self._redis_settings.socket_connect_timeout,
                    decode_responses=False,
                ) as async_redis:
                    json_data = json.dumps(data)
                    await async_redis.setex(key, self._ttl, json_data)
                logger.debug(f"Populated Redis cache for {category} (async)")
            except Exception as e:
                logger.warning(f"Failed to cache {category} in Redis (async): {e}")

        return data

    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories from Redis with PRIMARY API fallback."""
        tasks = [
            self.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )
            for category in MODEL_REFERENCE_CATEGORY
        ]

        results = await asyncio.gather(*tasks)

        return dict(zip(MODEL_REFERENCE_CATEGORY, results, strict=False))

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if category needs refresh (delegates to file backend)."""
        return self._file_backend.needs_refresh(category)

    @override
    def _mark_stale_impl(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark category as stale and invalidate Redis cache.

        Also publishes invalidation event to notify other workers.
        """
        key = self._category_key(category)

        if self._redis_settings.use_pubsub:
            try:
                channel = self._invalidation_channel()
                self._retry_redis_operation(
                    self._sync_redis.publish,
                    channel,
                    category.value,
                )
                logger.debug(f"Published invalidation for {category}")
            except Exception as e:
                logger.warning(f"Failed to publish invalidation for {category}: {e}")

        try:
            self._retry_redis_operation(self._sync_redis.delete, key)
            logger.debug(f"Invalidated Redis cache for {category}")
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache for {category}: {e}")

        self._file_backend.mark_stale(category)

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get file path (delegates to file backend)."""
        return self._file_backend.get_category_file_path(category)

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get all file paths (delegates to file backend)."""
        return self._file_backend.get_all_category_file_paths()

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get legacy JSON (delegates to file backend)."""
        return self._file_backend.get_legacy_json(category, redownload=redownload)

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get legacy JSON string (delegates to file backend)."""
        return self._file_backend.get_legacy_json_string(category, redownload=redownload)

    @override
    def get_replicate_mode(self) -> ReplicateMode:
        """Get replication mode (always PRIMARY for Redis backend)."""
        return self._replicate_mode

    @override
    def supports_writes(self) -> bool:
        """Check if backend supports writes (delegates to file backend)."""
        return self._file_backend.supports_writes()

    @override
    def supports_metadata(self) -> bool:
        """Check if backend supports metadata tracking (delegates to file backend).

        Returns:
            bool: True if file backend supports metadata.
        """
        return self._file_backend.supports_metadata()

    @override
    def supports_cache_warming(self) -> bool:
        """Check if backend supports cache warming (True for Redis).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def supports_health_checks(self) -> bool:
        """Check if backend supports health checks (True for Redis).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def supports_statistics(self) -> bool:
        """Check if backend supports statistics (True for Redis).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update model via file backend, then invalidate Redis cache."""
        self._file_backend.update_model(category, model_name, record_dict)

        self.mark_stale(category)

    @override
    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete model via file backend, then invalidate Redis cache."""
        self._file_backend.delete_model(category, model_name)

        self.mark_stale(category)

    @override
    def warm_cache(self) -> None:
        """Pre-populate Redis cache from files on startup."""
        logger.info("Warming Redis cache...")
        for category in MODEL_REFERENCE_CATEGORY:
            try:
                self.fetch_category(category, force_refresh=True)
            except Exception as e:
                logger.warning(f"Failed to warm cache for {category}: {e}")
        logger.info("Redis cache warming complete")

    @override
    async def warm_cache_async(self) -> None:
        """Asynchronously pre-populate Redis cache."""
        logger.info("Warming Redis cache (async)...")
        async with httpx.AsyncClient() as client:
            tasks = [
                self.fetch_category_async(category, httpx_client=client, force_refresh=True)
                for category in MODEL_REFERENCE_CATEGORY
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for category, result in zip(MODEL_REFERENCE_CATEGORY, results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to warm cache for {category}: {result}")

        logger.info("Redis cache warming complete (async)")

    @override
    def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            self._sync_redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    @override
    def get_statistics(self) -> dict[str, Any]:
        """Get Redis cache statistics.

        Returns:
            dict containing:
                - connected: Whether Redis is connected
                - keys_count: Number of keys in Redis database
                - total_connections: Total connections received
                - total_commands: Total commands processed
                - memory_used_bytes: Memory used in bytes
                - memory_used_human: Human-readable memory usage
        """
        try:
            info = self._sync_redis.info("stats")
            memory = self._sync_redis.info("memory")

            return {
                "connected": True,
                "keys_count": self._sync_redis.dbsize(),
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "memory_used_bytes": memory.get("used_memory", 0),
                "memory_used_human": memory.get("used_memory_human", "unknown"),
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis stats: {e}")
            return {"connected": False, "error": str(e)}

    @override
    def get_legacy_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get legacy format metadata, using Redis cache.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The legacy metadata, or None if not available.
        """
        key = self._legacy_metadata_key(category)

        try:
            cached = self._retry_redis_operation(self._sync_redis.get, key)
            if cached:
                if not isinstance(cached, str):
                    raise ValueError("Expected str from Redis")
                data = json.loads(cached)
                return CategoryMetadata(**data)
        except Exception as e:
            logger.warning(f"Redis fetch failed for legacy metadata {category}, falling back to file: {e}")

        # Fall back to file backend
        metadata = self._file_backend.get_legacy_metadata(category)

        # Cache the result in Redis
        if metadata is not None:
            try:
                json_data = json.dumps(metadata.model_dump(mode="json"))
                self._retry_redis_operation(self._sync_redis.setex, key, self._ttl, json_data)
            except Exception as e:
                logger.warning(f"Failed to cache legacy metadata for {category} in Redis: {e}")

        return metadata

    @override
    async def get_legacy_metadata_async(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Asynchronously get legacy format metadata, using Redis cache.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The legacy metadata, or None if not available.
        """
        # For now, delegate to sync version
        return self.get_legacy_metadata(category)

    @override
    def get_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get v2 format metadata, using Redis cache.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The v2 metadata, or None if not available.
        """
        key = self._v2_metadata_key(category)

        try:
            cached = self._retry_redis_operation(self._sync_redis.get, key)
            if cached:
                if not isinstance(cached, str):
                    raise ValueError("Expected str from Redis")
                data = json.loads(cached)
                return CategoryMetadata(**data)
        except Exception as e:
            logger.warning(f"Redis fetch failed for v2 metadata {category}, falling back to file: {e}")

        # Fall back to file backend
        metadata = self._file_backend.get_metadata(category)

        # Cache the result in Redis
        if metadata is not None:
            try:
                json_data = json.dumps(metadata.model_dump(mode="json"))
                self._retry_redis_operation(self._sync_redis.setex, key, self._ttl, json_data)
            except Exception as e:
                logger.warning(f"Failed to cache v2 metadata for {category} in Redis: {e}")

        return metadata

    @override
    async def get_metadata_async(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Asynchronously get v2 format metadata, using Redis cache.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The v2 metadata, or None if not available.
        """
        # For now, delegate to sync version
        return self.get_metadata(category)

    @override
    def get_all_legacy_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get legacy format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their legacy metadata.
        """
        result = {}
        for category in MODEL_REFERENCE_CATEGORY:
            metadata = self.get_legacy_metadata(category)
            if metadata is not None:
                result[category] = metadata
        return result

    @override
    async def get_all_legacy_metadata_async(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Asynchronously get legacy format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their legacy metadata.
        """
        return self.get_all_legacy_metadata()

    @override
    def get_all_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get v2 format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their v2 metadata.
        """
        result = {}
        for category in MODEL_REFERENCE_CATEGORY:
            metadata = self.get_metadata(category)
            if metadata is not None:
                result[category] = metadata
        return result

    @override
    async def get_all_metadata_async(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Asynchronously get v2 format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their v2 metadata.
        """
        return self.get_all_metadata()

    def __del__(self) -> None:
        """Clean up Redis connections on deletion."""
        if hasattr(self, "_pubsub_running"):
            self._pubsub_running = False

        if hasattr(self, "_pubsub"):
            with contextlib.suppress(Exception):
                self._pubsub.close()

        if hasattr(self, "_sync_redis"):
            with contextlib.suppress(Exception):
                self._sync_redis.close()
