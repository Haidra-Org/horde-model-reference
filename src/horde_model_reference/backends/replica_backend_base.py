"""Shared functionality for REPLICA-oriented model reference backends."""

from __future__ import annotations

import time
from asyncio import Lock as AsyncLock
from threading import RLock
from typing import override

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ReplicaBackendBase(ModelReferenceBackend):
    """Intermediate base class for read-only replica backends."""

    def __init__(
        self,
        *,
        mode: ReplicateMode = ReplicateMode.REPLICA,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Configure shared cache tracking for replica backends."""
        super().__init__(mode=mode)

        self._cache_ttl_seconds = cache_ttl_seconds
        self._stale_categories: set[MODEL_REFERENCE_CATEGORY] = set()
        self._category_timestamps: dict[MODEL_REFERENCE_CATEGORY, float] = {}

        # Shared coordination primitives for subclasses.
        self._lock = RLock()
        self._async_lock: AsyncLock | None = AsyncLock()

    def _mark_category_fresh(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Record that we hold a fresh cache entry for *category*."""
        self._category_timestamps[category] = time.time()
        self._stale_categories.discard(category)

    def _invalidate_category_timestamp(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Drop timestamp knowledge for *category* without adjusting payloads."""
        self._category_timestamps.pop(category, None)

    def is_cache_valid(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Return True if the cache for *category* is considered fresh."""
        with self._lock:
            if category in self._stale_categories:
                return False

            last_updated = self._category_timestamps.get(category)

        if last_updated is None:
            return False

        if self._cache_ttl_seconds is None:
            return True

        return (time.time() - last_updated) <= self._cache_ttl_seconds

    @override
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:  # type: ignore[override]
        """Mark a category as stale so subclasses refetch on next access."""
        with self._lock:
            self._stale_categories.add(category)

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:  # type: ignore[override]
        """Determine whether a category should be refreshed."""
        with self._lock:
            if category in self._stale_categories:
                return True

            last_updated = self._category_timestamps.get(category)

        if last_updated is None:
            return True

        if self._cache_ttl_seconds is None:
            return False

        return (time.time() - last_updated) > self._cache_ttl_seconds

    @property
    def cache_ttl_seconds(self) -> int | None:
        """The cache TTL currently enforced for category payloads."""
        return self._cache_ttl_seconds

    @property
    def lock(self) -> RLock:
        """Thread-safe lock shared by subclasses for critical sections."""
        return self._lock

    @property
    def async_lock(self) -> AsyncLock | None:
        """Asyncio lock usable by subclasses when coordinating coroutines."""
        return self._async_lock

    def _set_cache_ttl_seconds(self, ttl_seconds: int | None) -> None:
        """Allow subclasses to tweak TTL after initialization if desired."""
        self._cache_ttl_seconds = ttl_seconds
