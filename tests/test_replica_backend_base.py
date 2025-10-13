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
        return {category: None for category in MODEL_REFERENCE_CATEGORY}

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
    """Exercise mark/stale flows and TTL expiry handling."""
    category = MODEL_REFERENCE_CATEGORY.image_generation
    current_time = 1_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr("horde_model_reference.backends.replica_backend_base.time.time", fake_time)

    assert not replica_probe.is_cache_valid(category)
    assert replica_probe.needs_refresh(category)

    replica_probe._mark_category_fresh(category)
    assert replica_probe.is_cache_valid(category)
    assert not replica_probe.needs_refresh(category)

    current_time += 11
    assert not replica_probe.is_cache_valid(category)
    assert replica_probe.needs_refresh(category)

    replica_probe._mark_category_fresh(category)
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

    probe._mark_category_fresh(category)
    current_time += 10_000
    assert probe.is_cache_valid(category)
    assert not probe.needs_refresh(category)
