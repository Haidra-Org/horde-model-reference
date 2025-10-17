"""Shared functionality for REPLICA-oriented model reference backends.

This base class provides comprehensive caching infrastructure for all backend types,
not just REPLICA mode backends. It includes TTL-based caching, mtime validation,
and extensible hooks for custom cache validation logic.
"""

from __future__ import annotations

import time
from asyncio import Lock as AsyncLock
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any, override

from loguru import logger

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ReplicaBackendBase(ModelReferenceBackend):
    """Base class providing caching infrastructure for all backend types.

    Despite the name, this class provides caching for both REPLICA and PRIMARY backends.
    It implements TTL-based caching, mtime validation, and extensible validation hooks.

    **Dual Cache System:**
    Maintains separate caches for v2/converted format and legacy format:
    - V2 cache: `_cache` (dict), used by fetch_category()
    - Legacy cache: `_legacy_json_cache` (dict) + `_legacy_json_string_cache` (str)

    Each cache has its own timestamps, mtime tracking, and staleness management.

    **For V2/Converted Format:**
    - Call `_get_from_cache()` to retrieve cached v2 data
    - Call `_store_in_cache()` to store fetched v2 data
    - Override `_get_file_path_for_validation()` for v2 file mtime validation

    **For Legacy Format:**
    - Call `_get_legacy_from_cache()` to retrieve cached legacy data (dict + string)
    - Call `_store_legacy_in_cache()` to store fetched legacy data
    - Override `_get_legacy_file_path_for_validation()` for legacy file mtime validation

    **Custom Validation:**
    - Override `_additional_cache_validation()` for custom v2 validation logic
    """

    def __init__(
        self,
        *,
        mode: ReplicateMode = ReplicateMode.REPLICA,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Configure shared cache tracking for all backends.

        Args:
            mode: The replication mode (REPLICA or PRIMARY).
            cache_ttl_seconds: TTL for cache entries in seconds. None means no expiration.
        """
        super().__init__(mode=mode)

        self._cache_ttl_seconds = cache_ttl_seconds

        # V2/Converted format cache infrastructure
        self._cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        self._category_timestamps: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._last_known_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._stale_categories: set[MODEL_REFERENCE_CATEGORY] = set()

        # Legacy format cache infrastructure (dict and string)
        self._legacy_json_cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        self._legacy_json_string_cache: dict[MODEL_REFERENCE_CATEGORY, str | None] = {}
        self._legacy_cache_timestamps: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._legacy_last_known_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._stale_legacy_categories: set[MODEL_REFERENCE_CATEGORY] = set()

        self._lock = RLock()
        self._async_lock: AsyncLock | None = AsyncLock()

    def _mark_category_fresh(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Record that we hold a fresh cache entry for *category*.

        Also updates mtime if a file path is provided by the subclass.

        Args:
            category: The category to mark as fresh.
        """
        self._category_timestamps[category] = time.time()
        self._stale_categories.discard(category)

        file_path = self._get_file_path_for_validation(category)
        if file_path and file_path.exists():
            try:
                self._last_known_mtimes[category] = file_path.stat().st_mtime
            except Exception:
                self._last_known_mtimes[category] = 0.0

        logger.debug(f"Marked category {category} as fresh")

    def _invalidate_category_timestamp(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Drop timestamp knowledge for *category* without adjusting payloads."""
        self._category_timestamps.pop(category, None)

    def has_cached_data(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if any data has been cached for this category.

        This is a simple existence check that doesn't validate freshness.
        Use this for initial fetch detection: "Have we loaded this at least once?"

        Args:
            category: The category to check.

        Returns:
            bool: True if data exists in cache (may be stale), False if never loaded.
        """
        with self._lock:
            return category in self._cache

    def is_cache_valid(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Return True if the cache for *category* is considered fresh.

        This method checks if cached data EXISTS and is still valid. This is used
        internally for validation logic.

        Performs multiple validation checks:
        1. Staleness check (explicit invalidation)
        2. Cache existence check (initial fetch detection)
        3. TTL expiration check
        4. File mtime check (if file path provided by subclass)
        5. Additional custom validation (if overridden by subclass)

        Related methods:
        - `should_fetch_data()` - for "should I fetch?" checks
        - `has_cached_data()` - for "has data been loaded?" checks
        - `needs_refresh()` - checks if cached data should be RE-fetched (staleness only)
        - This method returns False for both "no data" and "stale data" cases
        - `needs_refresh()` returns False for "no data", True for "stale data"

        Args:
            category: The category to validate.

        Returns:
            bool: True if cache exists and is valid, False if cache doesn't exist or is invalid.
        """
        with self._lock:
            if category in self._stale_categories:
                logger.debug(f"Category {category} marked stale, cache invalid")
                return False

            if category not in self._cache:
                return False

            last_updated = self._category_timestamps.get(category)

        if last_updated is None:
            logger.debug(f"Category {category} has no timestamp, considering cache invalid")
            return False

        if self._cache_ttl_seconds is not None:
            elapsed = time.time() - last_updated
            if elapsed > self._cache_ttl_seconds:
                logger.debug(f"Category {category} TTL expired ({elapsed}s > {self._cache_ttl_seconds}s)")
                return False

        file_path = self._get_file_path_for_validation(category)
        if file_path and file_path.exists():
            try:
                current_mtime = file_path.stat().st_mtime
                last_known = self._last_known_mtimes.get(category, 0.0)
                if current_mtime != last_known:
                    logger.debug(
                        f"File {file_path.name} mtime changed "
                        f"(current={current_mtime}, cached={last_known}), cache invalid"
                    )
                    return False
            except Exception:
                return False

        if not self._additional_cache_validation(category):
            logger.debug(f"Category {category} failed additional validation")
            return False

        return True

    @override
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category as stale so subclasses refetch on next access.

        This causes needs_refresh() to return True for this category,
        triggering a refresh of cached data on the next access.
        """
        with self._lock:
            self._stale_categories.add(category)

    def should_fetch_data(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Determine if data should be fetched (initial load OR refresh).

        This is a convenience method that combines both initial fetch detection
        and refresh detection into a single check. Use this when you want to
        know "should I fetch data now?" regardless of whether it's an initial
        load or a refresh.

        This is equivalent to: `not is_cache_valid(category) or needs_refresh(category)`
        but handles the logic more efficiently.

        Args:
            category: The category to check.

        Returns:
            bool: True if data should be fetched (either initial or refresh),
                  False if cached data is valid and fresh.
        """
        return not self.is_cache_valid(category)

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Determine whether a category should be refreshed.

        This method checks if EXISTING cached data has become stale and needs to be
        re-fetched. It does NOT indicate whether an initial fetch is needed.

        Semantic distinction:
        - Returns False when no data has been fetched yet (no initial load needed via this method)
        - Returns True when cached data exists but has become stale due to:
          * Explicit staleness marking (mark_stale())
          * TTL expiration
          * File mtime changes

        For determining if data exists in cache (initial fetch check), use:
        - `is_cache_valid()` - checks if cache exists and is fresh
        - `_get_from_cache()` - retrieves data if cache exists and is valid
        - Direct cache existence check: `category not in self._cache`
        - `should_fetch_data()` - combined check for initial fetch OR refresh

        Args:
            category: The category to check.

        Returns:
            bool: True if cached data needs to be refreshed, False if no cached data exists
                  or if cached data is still fresh.
        """
        with self._lock:
            if category in self._stale_categories:
                logger.debug(f"Category {category} marked stale, needs refresh")
                return True

            last_updated = self._category_timestamps.get(category)

        if last_updated is None:
            # No timestamp means no data has been fetched/cached yet.
            # This is not a "refresh" scenario - it's an initial fetch scenario.
            # Callers should handle initial fetch separately from refresh logic.
            logger.debug(f"Category {category} has no timestamp, no refresh needed (not yet fetched)")
            return False

        if self._cache_ttl_seconds is not None:
            cache_stale = (time.time() - last_updated) > self._cache_ttl_seconds
            if cache_stale:
                logger.debug(f"Category {category} cache is stale, needs refresh")
                return True

        file_path = self._get_file_path_for_validation(category)
        if file_path and file_path.exists():
            try:
                current_mtime = file_path.stat().st_mtime
                last_known = self._last_known_mtimes.get(category, 0.0)
                if current_mtime != last_known:
                    logger.debug(f"File {file_path.name} mtime changed, needs refresh")
                    return True
            except Exception:
                return True

        return False

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

    def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return file path for mtime validation.

        Subclasses should override this if they want automatic mtime validation.
        If a path is returned, the cache will be invalidated if the file's mtime changes.

        Args:
            category: The category to get the file path for.

        Returns:
            Path | None: File path to check for mtime, or None to skip mtime validation.
        """
        return None

    def _additional_cache_validation(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Perform additional cache validation.

        Subclasses can override this to add custom validation logic beyond
        TTL and mtime checks. This is called during `is_cache_valid()`.

        Args:
            category: The category to validate.

        Returns:
            bool: True if cache is valid, False to invalidate.
        """
        return True

    def _fetch_with_cache(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        fetch_fn: Callable[[], dict[str, Any] | None],
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Implement standard fetch pattern with automatic caching.

        This helper method implements the recommended fetch pattern:
        1. Check cache if not forcing refresh
        2. Return cached data if valid
        3. Fetch data using provided function
        4. Store in cache and return

        Use this in your fetch_category() implementations to avoid boilerplate.

        Args:
            category: The category to fetch.
            fetch_fn: Callable that fetches the data (no args, returns dict or None).
            force_refresh: If True, skip cache check and force fetch.

        Returns:
            dict[str, Any] | None: The fetched/cached data.

        Example:
            def fetch_category(self, category, *, force_refresh=False):
                return self._fetch_with_cache(
                    category,
                    lambda: self._fetch_from_source(category),
                    force_refresh=force_refresh
                )
        """
        # Check cache first unless force refresh
        if not force_refresh:
            cached_data = self._get_from_cache(category)
            if cached_data is not None:
                return cached_data

        # Fetch data
        data = fetch_fn()

        # Store in cache
        if data is not None:
            self._store_in_cache(category, data)
        else:
            # Store None to indicate "checked but not found"
            self._store_in_cache(category, None)

        return data

    def _get_from_cache(self, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any] | None:
        """Get data from cache if valid.

        This is the primary method subclasses should use to retrieve cached data.
        It handles all validation logic internally, including initial fetch detection
        (returns None if data has never been loaded).

        This method determines if an INITIAL fetch is needed by checking cache existence.
        Use `needs_refresh()` to check if existing cached data should be RE-fetched.

        Args:
            category: The category to retrieve from cache.

        Returns:
            dict[str, Any] | None: Cached data if valid, None if cache miss (initial fetch needed)
                                   or cache invalid (refresh needed).
        """
        with self._lock:
            if self.is_cache_valid(category):
                logger.debug(f"Cache hit for {category}")
                return self._cache.get(category)

            logger.debug(f"Cache miss for {category}")
            return None

    def _store_in_cache(self, category: MODEL_REFERENCE_CATEGORY, data: dict[str, Any] | None) -> None:
        """Store data in cache and mark category as fresh.

        This is the primary method subclasses should use to store fetched data.
        It handles timestamp updates and mtime tracking internally.

        Args:
            category: The category to store.
            data: The data to cache, or None if category has no data.
        """
        with self._lock:
            self._cache[category] = data
            self._mark_category_fresh(category)
            logger.debug(f"Stored {category} in cache")

    def _invalidate_cache(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Invalidate cache for a category without deleting the data.

        This marks the category as stale, forcing a refetch on next access.

        Args:
            category: The category to invalidate.
        """
        with self._lock:
            self._stale_categories.add(category)
            self._category_timestamps.pop(category, None)
            logger.debug(f"Invalidated cache for {category}")

    def _get_legacy_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return legacy file path for mtime validation.

        Subclasses should override this if they want automatic mtime validation
        for legacy format files. This is separate from the v2 file path.

        Args:
            category: The category to get the legacy file path for.

        Returns:
            Path | None: Legacy file path to check for mtime, or None to skip mtime validation.
        """
        return None

    def _mark_legacy_category_fresh(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Record that we hold a fresh legacy cache entry for *category*.

        Also updates legacy file mtime if a path is provided by the subclass.

        Args:
            category: The category to mark as fresh.
        """
        self._legacy_cache_timestamps[category] = time.time()
        self._stale_legacy_categories.discard(category)

        legacy_file_path = self._get_legacy_file_path_for_validation(category)
        if legacy_file_path and legacy_file_path.exists():
            try:
                self._legacy_last_known_mtimes[category] = legacy_file_path.stat().st_mtime
            except Exception:
                self._legacy_last_known_mtimes[category] = 0.0

        logger.debug(f"Marked legacy category {category} as fresh")

    def is_legacy_cache_valid(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Return True if the legacy cache for *category* is considered fresh.

        Performs validation checks for legacy format cache:
        1. Staleness check (explicit invalidation)
        2. Cache existence check (dict or string)
        3. TTL expiration check
        4. File mtime check (if legacy file path provided by subclass)

        Args:
            category: The category to validate.

        Returns:
            bool: True if legacy cache is valid and can be used.
        """
        with self._lock:
            if category in self._stale_legacy_categories:
                logger.debug(f"Legacy category {category} marked stale, cache invalid")
                return False

            if category not in self._legacy_json_cache and category not in self._legacy_json_string_cache:
                return False

            last_updated = self._legacy_cache_timestamps.get(category)

        if last_updated is None:
            logger.debug(f"Legacy category {category} has no timestamp, considering cache invalid")
            return False

        if self._cache_ttl_seconds is not None:
            elapsed = time.time() - last_updated
            if elapsed > self._cache_ttl_seconds:
                logger.debug(f"Legacy category {category} TTL expired ({elapsed}s > {self._cache_ttl_seconds}s)")
                return False

        legacy_file_path = self._get_legacy_file_path_for_validation(category)
        if legacy_file_path and legacy_file_path.exists():
            try:
                current_mtime = legacy_file_path.stat().st_mtime
                last_known = self._legacy_last_known_mtimes.get(category, 0.0)
                if current_mtime != last_known:
                    logger.debug(
                        f"Legacy file {legacy_file_path.name} mtime changed "
                        f"(current={current_mtime}, cached={last_known}), cache invalid"
                    )
                    return False
            except Exception:
                return False

        return True

    def _get_legacy_from_cache(
        self,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Get legacy data from cache if valid.

        Returns both dict and string representations of legacy JSON.

        Args:
            category: The category to retrieve from cache.

        Returns:
            tuple[dict | None, str | None]: (legacy_dict, legacy_string) or (None, None) if cache miss.
        """
        with self._lock:
            if self.is_legacy_cache_valid(category):
                logger.debug(f"Legacy cache hit for {category}")
                return (
                    self._legacy_json_cache.get(category),
                    self._legacy_json_string_cache.get(category),
                )

            logger.debug(f"Legacy cache miss for {category}")
            return None, None

    def _store_legacy_in_cache(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        legacy_dict: dict[str, Any] | None,
        legacy_string: str | None,
    ) -> None:
        """Store legacy data in cache and mark category as fresh.

        Stores both dict and string representations of legacy JSON.

        Args:
            category: The category to store.
            legacy_dict: The legacy JSON as a dict, or None.
            legacy_string: The legacy JSON as a string, or None.
        """
        with self._lock:
            self._legacy_json_cache[category] = legacy_dict
            self._legacy_json_string_cache[category] = legacy_string
            self._mark_legacy_category_fresh(category)
            logger.debug(f"Stored legacy {category} in cache")

    def _invalidate_legacy_cache(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Invalidate legacy cache for a category without deleting the data.

        This marks the category as stale, forcing a refetch on next access.

        Args:
            category: The category to invalidate.
        """
        with self._lock:
            self._stale_legacy_categories.add(category)
            self._legacy_cache_timestamps.pop(category, None)
            logger.debug(f"Invalidated legacy cache for {category}")
