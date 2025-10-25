from pathlib import Path
from typing import Any

import httpx
import pytest

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class _ReplicaBackendProbe(ReplicaBackendBase):
    """Minimal concrete backend used for exercising ReplicaBackendBase."""

    def __init__(self, *, cache_ttl_seconds: int | None) -> None:
        super().__init__(mode=ReplicateMode.REPLICA, cache_ttl_seconds=cache_ttl_seconds)

    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        raise NotImplementedError

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        raise NotImplementedError

    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        return None

    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        return dict.fromkeys(MODEL_REFERENCE_CATEGORY)

    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        return None

    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        return None


@pytest.fixture
def replica_probe() -> _ReplicaBackendProbe:
    """Provide a replica backend instance with a finite TTL."""
    return _ReplicaBackendProbe(cache_ttl_seconds=10)


def test_replica_backend_cache_lifecycle(replica_probe: _ReplicaBackendProbe, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise mark/stale flows and TTL expiry handling.

    Tests the semantic distinction that needs_refresh() indicates whether CACHED data
    needs to be refreshed, not whether initial data needs to be fetched.
    - Initially (no cache): is_cache_valid=False, needs_refresh=False
    - After caching: is_cache_valid=True, needs_refresh=False
    - After TTL expires: is_cache_valid=False, needs_refresh=True
    - After marking stale: needs_refresh=True
    """
    category = MODEL_REFERENCE_CATEGORY.image_generation
    current_time = 1_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    # Initially, cache is invalid but doesn't need refresh (not yet fetched)
    # Semantic: needs_refresh is about RE-fetching stale data, not initial fetching
    assert not replica_probe.is_cache_valid(category)
    assert not replica_probe.needs_refresh(category)

    # Store data in cache to make it valid
    replica_probe._store_in_cache(category, {"test": "data"})
    assert replica_probe.is_cache_valid(category)
    assert not replica_probe.needs_refresh(category)

    current_time += 11
    assert not replica_probe.is_cache_valid(category)
    assert replica_probe.needs_refresh(category)

    # Store data again and mark stale
    replica_probe._store_in_cache(category, {"test": "data"})
    replica_probe.mark_stale(category)
    assert replica_probe.needs_refresh(category)


def test_replica_backend_without_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure caches remain valid when TTL is unset."""
    probe = _ReplicaBackendProbe(cache_ttl_seconds=None)
    category = MODEL_REFERENCE_CATEGORY.image_generation
    current_time = 5_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    # Store data in cache to make it valid
    probe._store_in_cache(category, {"test": "data"})
    current_time += 10_000
    assert probe.is_cache_valid(category)
    assert not probe.needs_refresh(category)


def test_helper_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the new helper methods for clearer API."""
    probe = _ReplicaBackendProbe(cache_ttl_seconds=10)
    category = MODEL_REFERENCE_CATEGORY.image_generation
    current_time = 1_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    # Test has_cached_data
    assert not probe.has_cached_data(category)

    # Test should_fetch_data - should be True when no cache
    assert probe.should_fetch_data(category)

    # Store data
    probe._store_in_cache(category, {"test": "data"})

    # Now has cached data
    assert probe.has_cached_data(category)
    assert not probe.should_fetch_data(category)

    # Advance time to expire TTL
    current_time += 11

    # Still has cached data, but should fetch (stale)
    assert probe.has_cached_data(category)
    assert probe.should_fetch_data(category)


def test_fetch_with_cache_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the _fetch_with_cache convenience method."""
    probe = _ReplicaBackendProbe(cache_ttl_seconds=10)
    category = MODEL_REFERENCE_CATEGORY.image_generation
    current_time = 1_000.0
    fetch_count = 0

    def fake_time() -> float:
        return current_time

    def mock_fetch() -> dict[str, Any]:
        nonlocal fetch_count
        fetch_count += 1
        return {"data": f"fetch_{fetch_count}"}

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    # First call should fetch
    result = probe._fetch_with_cache(category, mock_fetch, force_refresh=False)
    assert result == {"data": "fetch_1"}
    assert fetch_count == 1

    # Second call should use cache
    result = probe._fetch_with_cache(category, mock_fetch, force_refresh=False)
    assert result == {"data": "fetch_1"}  # Same data
    assert fetch_count == 1  # Not fetched again

    # Force refresh should fetch again
    result = probe._fetch_with_cache(category, mock_fetch, force_refresh=True)
    assert result == {"data": "fetch_2"}
    assert fetch_count == 2

    # After TTL expires, should fetch automatically
    current_time += 11
    result = probe._fetch_with_cache(category, mock_fetch, force_refresh=False)
    assert result == {"data": "fetch_3"}
    assert fetch_count == 3
