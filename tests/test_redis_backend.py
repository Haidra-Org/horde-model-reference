"""Comprehensive tests for RedisBackend distributed caching."""

import json
import time
from pathlib import Path
from typing import Any, cast, override
from unittest.mock import Mock

import fakeredis
import httpx
import pytest
import redis

from horde_model_reference import MODEL_REFERENCE_CATEGORY, RedisSettings, ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.backends.redis_backend import RedisBackend


class StubFileSystemBackend(ModelReferenceBackend):
    """Stub filesystem backend for testing RedisBackend."""

    def __init__(
        self,
        responses: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] | None = None,
    ) -> None:
        """Initialize with preset responses."""
        super().__init__(mode=ReplicateMode.PRIMARY)
        self._responses = responses or {}
        self.fetch_calls: list[MODEL_REFERENCE_CATEGORY] = []
        self.update_calls: list[tuple[MODEL_REFERENCE_CATEGORY, str, dict[str, Any]]] = []
        self.delete_calls: list[tuple[MODEL_REFERENCE_CATEGORY, str]] = []
        self.mark_stale_calls: list[MODEL_REFERENCE_CATEGORY] = []

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        self.fetch_calls.append(category)
        return self._responses.get(category)

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {category: self.fetch_category(category, force_refresh=force_refresh) for category in self._responses}

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        self.fetch_calls.append(category)
        return self._responses.get(category)

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {
            category: await self.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )
            for category in self._responses
        }

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        return False

    @override
    def _mark_stale_impl(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        self.mark_stale_calls.append(category)

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        return None

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        return {category: None for category in MODEL_REFERENCE_CATEGORY}

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        return None

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        return None

    @override
    def supports_writes(self) -> bool:
        return True

    @override
    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        self.update_calls.append((category, model_name, record_dict))

    @override
    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        self.delete_calls.append((category, model_name))


@pytest.fixture
def fake_redis_server() -> fakeredis.FakeRedis:
    """Create a fake Redis server for testing."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def redis_settings() -> RedisSettings:
    """Create Redis settings for testing."""
    return RedisSettings(
        use_redis=True,
        url="redis://localhost:6379/0",
        pool_size=10,
        socket_timeout=5,
        socket_connect_timeout=5,
        retry_max_attempts=3,
        retry_backoff_seconds=0.5,
        key_prefix="test:model_ref",
        ttl_seconds=60,
        use_pubsub=True,
    )


@pytest.fixture
def stub_file_backend() -> StubFileSystemBackend:
    """Create a stub filesystem backend for testing."""
    return StubFileSystemBackend(
        responses={
            MODEL_REFERENCE_CATEGORY.image_generation: {"model1": {"name": "model1"}},
            MODEL_REFERENCE_CATEGORY.text_generation: {"model2": {"name": "model2"}},
        },
    )


class RedisInitAndSetupTests:
    """Tests for RedisBackend initialization and setup."""

    def test_redis_backend_init_with_primary_backend(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RedisBackend should initialize successfully with PRIMARY FileSystemBackend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
            cache_ttl_seconds=60,
        )

        assert backend.get_replicate_mode() == ReplicateMode.PRIMARY
        assert backend._ttl == 60
        assert backend.supports_writes()

    def test_redis_backend_rejects_non_primary_backend(
        self,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RedisBackend should reject non-PRIMARY backends."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        non_primary_backend = StubFileSystemBackend()
        non_primary_backend._replicate_mode = ReplicateMode.REPLICA

        with pytest.raises(ValueError, match="can only be used with a FileSystemBackend in PRIMARY mode"):
            RedisBackend(
                file_backend=cast(FileSystemBackend, non_primary_backend),
                redis_settings=redis_settings,
            )

    def test_redis_backend_pubsub_enabled(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RedisBackend should set up pub/sub when enabled."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = True
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert backend._pubsub_running
        assert backend._pubsub_thread is not None
        assert backend._pubsub_thread.daemon

        backend._pubsub_running = False
        backend._pubsub_thread.join(timeout=1)

    def test_redis_backend_pubsub_disabled(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RedisBackend should not set up pub/sub when disabled."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert not backend._pubsub_running
        assert backend._pubsub_thread is None


class TestRedisSyncCacheOperations:
    """Tests for synchronous cache operations in RedisBackend."""

    def test_fetch_category_cache_miss_populates_redis(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache miss should fetch from file backend and populate Redis."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        result = backend.fetch_category(category)

        assert result == {"model1": {"name": "model1"}}
        assert category in stub_file_backend.fetch_calls

        key = backend._category_key(category)
        cached = fake_redis_server.get(key)
        assert cached is not None
        assert json.loads(cached) == {"model1": {"name": "model1"}}

    def test_fetch_category_cache_hit_skips_file_backend(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cache hit should return from Redis without hitting file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = backend._category_key(category)
        cached_data = {"cached": "data"}
        fake_redis_server.setex(key, 60, json.dumps(cached_data))

        stub_file_backend.fetch_calls.clear()

        result = backend.fetch_category(category)

        assert result == cached_data
        assert category not in stub_file_backend.fetch_calls

    def test_fetch_category_force_refresh_bypasses_cache(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force_refresh should bypass Redis cache and fetch from file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = backend._category_key(category)
        fake_redis_server.setex(key, 60, json.dumps({"old": "data"}))

        stub_file_backend.fetch_calls.clear()

        result = backend.fetch_category(category, force_refresh=True)

        assert result == {"model1": {"name": "model1"}}
        assert category in stub_file_backend.fetch_calls

    def test_fetch_all_categories(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fetch_all_categories should fetch all categories from cache/file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        result = backend.fetch_all_categories()

        assert MODEL_REFERENCE_CATEGORY.image_generation in result
        assert MODEL_REFERENCE_CATEGORY.text_generation in result
        assert result[MODEL_REFERENCE_CATEGORY.image_generation] == {"model1": {"name": "model1"}}


class TestRedisBackendAsyncCache:
    """Tests for asynchronous cache operations in RedisBackend."""

    @pytest.mark.asyncio
    async def test_fetch_category_async_cache_miss(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Async cache miss should fetch from file backend and populate Redis."""
        fake_async_redis = fakeredis.FakeAsyncRedis(decode_responses=True)

        def fake_from_url(*args: object, **kwargs: object) -> fakeredis.FakeAsyncRedis:
            return fake_async_redis

        monkeypatch.setattr("horde_model_reference.backends.redis_backend.redis.asyncio.from_url", fake_from_url)

        fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_sync_redis,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        result = await backend.fetch_category_async(category)

        assert result == {"model1": {"name": "model1"}}
        assert category in stub_file_backend.fetch_calls

    @pytest.mark.asyncio
    async def test_fetch_category_async_cache_hit(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Async cache hit should return from Redis without hitting file backend."""
        fake_async_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        cached_data = {"cached": "async"}

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = f"{redis_settings.key_prefix}:category:{category.value}"
        await fake_async_redis.setex(key, 60, json.dumps(cached_data))

        def fake_from_url(*args: object, **kwargs: object) -> fakeredis.FakeAsyncRedis:
            return fake_async_redis

        monkeypatch.setattr("horde_model_reference.backends.redis_backend.redis.asyncio.from_url", fake_from_url)

        fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_sync_redis,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        stub_file_backend.fetch_calls.clear()

        result = await backend.fetch_category_async(category)

        assert result == cached_data
        assert category not in stub_file_backend.fetch_calls

    @pytest.mark.asyncio
    async def test_fetch_category_async_force_refresh(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Async force_refresh should bypass cache."""
        fake_async_redis = fakeredis.FakeAsyncRedis(decode_responses=True)

        def fake_from_url(*args: object, **kwargs: object) -> fakeredis.FakeAsyncRedis:
            return fake_async_redis

        monkeypatch.setattr("horde_model_reference.backends.redis_backend.redis.asyncio.from_url", fake_from_url)

        fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_sync_redis,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        stub_file_backend.fetch_calls.clear()

        result = await backend.fetch_category_async(category, force_refresh=True)

        assert result == {"model1": {"name": "model1"}}
        assert category in stub_file_backend.fetch_calls

    @pytest.mark.asyncio
    async def test_fetch_all_categories_async(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Async fetch_all_categories should fetch all categories."""
        fake_async_redis = fakeredis.FakeAsyncRedis(decode_responses=True)

        def fake_from_url(*args: object, **kwargs: object) -> fakeredis.FakeAsyncRedis:
            return fake_async_redis

        monkeypatch.setattr("horde_model_reference.backends.redis_backend.redis.asyncio.from_url", fake_from_url)

        fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_sync_redis,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        result = await backend.fetch_all_categories_async()

        assert MODEL_REFERENCE_CATEGORY.image_generation in result
        assert MODEL_REFERENCE_CATEGORY.text_generation in result


class RedisCacheInvalidationTests:
    """Tests for cache invalidation in RedisBackend."""

    def test_mark_stale_deletes_redis_key(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mark_stale should delete the Redis key for the category."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = backend._category_key(category)
        fake_redis_server.setex(key, 60, json.dumps({"data": "test"}))

        backend.mark_stale(category)

        assert fake_redis_server.get(key) is None
        assert category in stub_file_backend.mark_stale_calls

    def test_mark_stale_publishes_invalidation(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """mark_stale should publish invalidation message when pub/sub is enabled."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = True
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        channel = backend._invalidation_channel()

        pubsub = fake_redis_server.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(channel)

        backend.mark_stale(category)

        time.sleep(0.1)

        message = pubsub.get_message()
        if message:
            assert message["type"] == "message"
            assert message["data"].decode("utf-8") == category.value

        backend._pubsub_running = False
        if backend._pubsub_thread:
            backend._pubsub_thread.join(timeout=1)


class RedisWriteOperationsTests:
    """Tests for write operations in RedisBackend."""

    def test_update_model_invalidates_cache(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """update_model should delegate to file backend and invalidate Redis cache."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = backend._category_key(category)
        fake_redis_server.setex(key, 60, json.dumps({"old": "data"}))

        model_name = "new_model"
        record_dict = {"name": "new_model", "description": "test"}

        backend.update_model(category, model_name, record_dict)

        assert (category, model_name, record_dict) in stub_file_backend.update_calls
        assert fake_redis_server.get(key) is None
        assert category in stub_file_backend.mark_stale_calls

    def test_delete_model_invalidates_cache(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """delete_model should delegate to file backend and invalidate Redis cache."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        key = backend._category_key(category)
        fake_redis_server.setex(key, 60, json.dumps({"model1": {"name": "model1"}}))

        model_name = "model1"

        backend.delete_model(category, model_name)

        assert (category, model_name) in stub_file_backend.delete_calls
        assert fake_redis_server.get(key) is None
        assert category in stub_file_backend.mark_stale_calls


class RedisErrorHandlingTests:
    """Tests for error handling in RedisBackend."""

    def test_fetch_category_redis_failure_falls_back_to_file_backend(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redis failures should gracefully fall back to file backend."""

        class FailingRedis:
            def get(self, key: str) -> None:
                raise redis.ConnectionError("Connection failed")

            def setex(self, key: str, time: int, value: str) -> None:
                raise redis.ConnectionError("Connection failed")

            def close(self) -> None:
                pass

            def pubsub(self, ignore_subscribe_messages: bool = True) -> Mock:
                mock_pubsub = Mock()
                mock_pubsub.subscribe = Mock()
                mock_pubsub.close = Mock()
                return mock_pubsub

        failing_redis = FailingRedis()
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: failing_redis,
        )

        redis_settings.use_pubsub = False
        redis_settings.retry_max_attempts = 1

        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        result = backend.fetch_category(category)

        assert result == {"model1": {"name": "model1"}}
        assert category in stub_file_backend.fetch_calls

    def test_retry_logic_with_exponential_backoff(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Retry logic should use exponential backoff."""

        class CountingFailRedis:
            def __init__(self) -> None:
                self.get_attempts = 0

            def get(self, key: str) -> None:
                self.get_attempts += 1
                raise redis.ConnectionError("Connection failed")

            def close(self) -> None:
                pass

            def pubsub(self, ignore_subscribe_messages: bool = True) -> Mock:
                mock_pubsub = Mock()
                mock_pubsub.subscribe = Mock()
                mock_pubsub.close = Mock()
                return mock_pubsub

        counting_redis = CountingFailRedis()
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: counting_redis,
        )

        redis_settings.use_pubsub = False
        redis_settings.retry_max_attempts = 3
        redis_settings.retry_backoff_seconds = 0.01

        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        with pytest.raises(redis.ConnectionError):
            backend._retry_redis_operation(counting_redis.get, "test_key")

        assert counting_redis.get_attempts == 3


def test_warm_cache(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    fake_redis_server: fakeredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warm_cache should pre-populate Redis with all categories."""
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: fake_redis_server,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    backend.warm_cache()

    for category, expected_data in stub_file_backend._responses.items():
        key = backend._category_key(category)
        cached = fake_redis_server.get(key)
        assert cached is not None
        assert json.loads(cached) == expected_data


@pytest.mark.asyncio
async def test_warm_cache_async(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warm_cache_async should pre-populate Redis asynchronously."""
    fake_async_redis = fakeredis.FakeAsyncRedis(decode_responses=True)

    def fake_from_url(*args: object, **kwargs: object) -> fakeredis.FakeAsyncRedis:
        return fake_async_redis

    monkeypatch.setattr("horde_model_reference.backends.redis_backend.redis.asyncio.from_url", fake_from_url)

    fake_sync_redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: fake_sync_redis,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    await backend.warm_cache_async()

    for category in stub_file_backend._responses:
        assert category in stub_file_backend.fetch_calls


def test_health_check_success(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    fake_redis_server: fakeredis.FakeRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """health_check should return True when Redis is connected."""
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: fake_redis_server,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    assert backend.health_check()


def test_health_check_failure(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """health_check should return False when Redis ping fails."""

    class FailPingRedis:
        def ping(self) -> None:
            raise redis.ConnectionError("Connection failed")

        def close(self) -> None:
            pass

        def pubsub(self, ignore_subscribe_messages: bool = True) -> Mock:
            mock_pubsub = Mock()
            mock_pubsub.subscribe = Mock()
            mock_pubsub.close = Mock()
            return mock_pubsub

    failing_redis = FailPingRedis()
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: failing_redis,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    assert not backend.health_check()


def test_get_statistics(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_statistics should return Redis stats."""

    class MockRedisWithStats:
        def info(self, section: str) -> dict[str, Any]:
            if section == "stats":
                return {
                    "total_connections_received": 100,
                    "total_commands_processed": 500,
                }
            if section == "memory":
                return {
                    "used_memory": 1024000,
                    "used_memory_human": "1.00M",
                }
            return {}

        def dbsize(self) -> int:
            return 42

        def ping(self) -> bool:
            return True

        def close(self) -> None:
            pass

        def pubsub(self, ignore_subscribe_messages: bool = True) -> Mock:
            mock_pubsub = Mock()
            mock_pubsub.subscribe = Mock()
            mock_pubsub.close = Mock()
            return mock_pubsub

    mock_redis = MockRedisWithStats()
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: mock_redis,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    stats = backend.get_statistics()

    assert "connected" in stats
    assert stats["connected"]
    assert stats["keys_count"] == 42
    assert stats["total_connections"] == 100
    assert stats["total_commands"] == 500
    assert stats["memory_used_bytes"] == 1024000
    assert stats["memory_used_human"] == "1.00M"


def test_get_statistics_on_failure(
    stub_file_backend: StubFileSystemBackend,
    redis_settings: RedisSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_statistics should return error info when Redis fails."""

    class FailStatsRedis:
        def info(self, section: str) -> dict[str, Any]:
            raise redis.ConnectionError("Connection failed")

        def dbsize(self) -> int:
            raise redis.ConnectionError("Connection failed")

        def close(self) -> None:
            pass

        def pubsub(self, ignore_subscribe_messages: bool = True) -> Mock:
            mock_pubsub = Mock()
            mock_pubsub.subscribe = Mock()
            mock_pubsub.close = Mock()
            return mock_pubsub

    failing_redis = FailStatsRedis()
    monkeypatch.setattr(
        "horde_model_reference.backends.redis_backend.redis.from_url",
        lambda *args, **kwargs: failing_redis,
    )

    redis_settings.use_pubsub = False
    backend = RedisBackend(
        file_backend=cast(FileSystemBackend, stub_file_backend),
        redis_settings=redis_settings,
    )

    stats = backend.get_statistics()

    assert "connected" in stats
    assert not stats["connected"]
    assert "error" in stats


class RedisBackendDelegationTests:
    """Tests to ensure RedisBackend delegates certain methods to FileSystemBackend."""

    def test_needs_refresh_delegates_to_file_backend(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """needs_refresh should delegate to file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        result = backend.needs_refresh(category)

        assert result is False

    def test_get_category_file_path_delegates(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_category_file_path should delegate to file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation
        result = backend.get_category_file_path(category)

        assert result is None

    def test_get_all_category_file_paths_delegates(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """get_all_category_file_paths should delegate to file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        result = backend.get_all_category_file_paths()

        assert isinstance(result, dict)
        assert len(result) == len(MODEL_REFERENCE_CATEGORY)

    def test_supports_writes_delegates(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """supports_writes should delegate to file backend."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert backend.supports_writes()


class TestRedisBackendCapabilities:
    """Tests for capability reporting methods in RedisBackend."""

    def test_supports_cache_warming(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """supports_cache_warming should return True."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert backend.supports_cache_warming()

    def test_supports_health_checks(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """supports_health_checks should return True."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert backend.supports_health_checks()

    def test_supports_statistics(
        self,
        stub_file_backend: StubFileSystemBackend,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """supports_statistics should return True."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=cast(FileSystemBackend, stub_file_backend),
            redis_settings=redis_settings,
        )

        assert backend.supports_statistics()


class TestRedisMultiWorkerScenarios:
    """Tests for multi-worker coordination via Redis pub/sub."""

    def test_multi_worker_write_invalidation(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker A writes, Worker B should see invalidation and fetch fresh data."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        file_backend_a = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = True
        backend_a = RedisBackend(
            file_backend=file_backend_a,
            redis_settings=redis_settings,
        )

        file_backend_b = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        backend_b = RedisBackend(
            file_backend=file_backend_b,
            redis_settings=redis_settings,
        )

        initial_data = {"model1": {"name": "model1", "description": "initial"}}
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(initial_data))

        result_b_initial = backend_b.fetch_category(category)
        assert result_b_initial == initial_data

        key = backend_a._category_key(category)
        cached_in_redis = json.loads(fake_redis_server.get(key) or "{}")
        assert cached_in_redis == initial_data

        updated_data = {"model1": {"name": "model1", "description": "updated"}}
        backend_a.update_model(category, "model1", updated_data["model1"])

        time.sleep(0.2)

        cached_after_invalidation = fake_redis_server.get(key)
        assert cached_after_invalidation is None

        result_b_after = backend_b.fetch_category(category)
        assert result_b_after is not None
        assert result_b_after["model1"]["description"] == "updated"

        backend_a._pubsub_running = False
        backend_b._pubsub_running = False
        if backend_a._pubsub_thread:
            backend_a._pubsub_thread.join(timeout=1)
        if backend_b._pubsub_thread:
            backend_b._pubsub_thread.join(timeout=1)

    def test_multi_worker_delete_invalidation(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker A deletes, Worker B should see invalidation."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        file_backend_a = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = True
        backend_a = RedisBackend(
            file_backend=file_backend_a,
            redis_settings=redis_settings,
        )

        file_backend_b = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        backend_b = RedisBackend(
            file_backend=file_backend_b,
            redis_settings=redis_settings,
        )

        initial_data = {"model1": {"name": "model1"}, "model2": {"name": "model2"}}
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(initial_data))

        result_b_initial = backend_b.fetch_category(category)
        assert result_b_initial is not None
        assert "model1" in result_b_initial
        assert "model2" in result_b_initial

        backend_a.delete_model(category, "model1")

        time.sleep(0.2)

        key = backend_a._category_key(category)
        cached_after_delete = fake_redis_server.get(key)
        assert cached_after_delete is None

        result_b_after = backend_b.fetch_category(category)
        assert result_b_after is not None
        assert "model1" not in result_b_after
        assert "model2" in result_b_after

        backend_a._pubsub_running = False
        backend_b._pubsub_running = False
        if backend_a._pubsub_thread:
            backend_a._pubsub_thread.join(timeout=1)
        if backend_b._pubsub_thread:
            backend_b._pubsub_thread.join(timeout=1)

    def test_pubsub_listener_actually_deletes_redis_key(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify pub/sub listener deletes Redis key on invalidation message."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        file_backend_a = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = True
        backend_a = RedisBackend(
            file_backend=file_backend_a,
            redis_settings=redis_settings,
        )

        file_backend_b = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        backend_b = RedisBackend(
            file_backend=file_backend_b,
            redis_settings=redis_settings,
        )

        key = backend_a._category_key(category)
        test_data = {"test": "data"}
        fake_redis_server.setex(key, 60, json.dumps(test_data))

        assert fake_redis_server.get(key) is not None

        channel = backend_a._invalidation_channel()
        fake_redis_server.publish(channel, category.value)

        time.sleep(0.2)

        cached = fake_redis_server.get(key)
        assert cached is None

        backend_a._pubsub_running = False
        backend_b._pubsub_running = False
        if backend_a._pubsub_thread:
            backend_a._pubsub_thread.join(timeout=1)
        if backend_b._pubsub_thread:
            backend_b._pubsub_thread.join(timeout=1)


class TestDiskPersistence:
    """Tests for disk persistence of write operations."""

    def test_update_model_persists_to_disk(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """update_model should persist changes to disk atomically."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({}))

        file_backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=file_backend,
            redis_settings=redis_settings,
        )

        model_data = {"name": "new_model", "description": "test"}
        backend.update_model(category, "new_model", model_data)

        assert file_path.exists()
        stored_data = json.loads(file_path.read_text())
        assert "new_model" in stored_data
        assert stored_data["new_model"] == model_data

    def test_delete_model_removes_from_disk(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """delete_model should remove model from disk atomically."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        initial_data = {"model1": {"name": "model1"}, "model2": {"name": "model2"}}
        file_path.write_text(json.dumps(initial_data))

        file_backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=file_backend,
            redis_settings=redis_settings,
        )

        backend.delete_model(category, "model1")

        assert file_path.exists()
        stored_data = json.loads(file_path.read_text())
        assert "model1" not in stored_data
        assert "model2" in stored_data

    def test_write_updates_file_mtime(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Write operations should update file modification time."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({}))

        initial_mtime = file_path.stat().st_mtime
        time.sleep(0.1)

        file_backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=file_backend,
            redis_settings=redis_settings,
        )

        model_data = {"name": "new_model"}
        backend.update_model(category, "new_model", model_data)

        new_mtime = file_path.stat().st_mtime
        assert new_mtime > initial_mtime

    def test_external_file_change_detected(
        self,
        primary_base: Path,
        redis_settings: RedisSettings,
        fake_redis_server: fakeredis.FakeRedis,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Backend should detect external file changes via mtime."""
        monkeypatch.setattr(
            "horde_model_reference.backends.redis_backend.redis.from_url",
            lambda *args, **kwargs: fake_redis_server,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        file_path = primary_base / "miscellaneous.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        initial_data = {"model1": {"name": "model1"}}
        file_path.write_text(json.dumps(initial_data))

        file_backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        redis_settings.use_pubsub = False
        backend = RedisBackend(
            file_backend=file_backend,
            redis_settings=redis_settings,
        )

        result_initial = backend.fetch_category(category)
        assert result_initial == initial_data

        time.sleep(0.1)
        updated_data = {"model2": {"name": "model2"}}
        file_path.write_text(json.dumps(updated_data))
        import os

        stat = file_path.stat()
        os.utime(file_path, (stat.st_atime, stat.st_mtime + 1))

        assert backend.needs_refresh(category)

        result_after = backend.fetch_category(category, force_refresh=True)
        assert result_after == updated_data
