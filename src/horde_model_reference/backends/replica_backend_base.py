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
from typing import Any

from loguru import logger
from typing_extensions import override

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ReplicaBackendBase(ModelReferenceBackend):
    """Base class providing comprehensive caching infrastructure for model reference backends.

    Despite its name, this class serves as a universal caching layer for both REPLICA and PRIMARY
    backend modes. It implements a dual-cache architecture supporting both modern (v2/converted)
    and legacy JSON formats, with sophisticated validation mechanisms including TTL expiration,
    file modification time tracking, and extensible custom validation hooks.

    Cache Architecture:
        The class maintains two independent cache systems:

        **V2/Converted Format Cache:**
            - `_cache`: Primary storage for converted model reference data
            - `_category_timestamps`: Tracks when each category was last cached
            - `_last_known_mtimes`: File modification times for invalidation
            - `_stale_categories`: Set of categories marked for refresh

        **Legacy Format Cache:**
            - `_legacy_json_cache`: Dict representation of legacy JSON
            - `_legacy_json_string_cache`: String representation of legacy JSON
            - `_legacy_cache_timestamps`: Tracks when each legacy category was cached
            - `_legacy_last_known_mtimes`: Legacy file modification times
            - `_stale_legacy_categories`: Set of legacy categories marked for refresh

    Cache Validation:
        The validation system performs multiple checks to ensure cache freshness:

        1. **Explicit Staleness**: Categories marked via `mark_stale()` or `_invalidate_cache()`
        2. **TTL Expiration**: Time-based expiration if `cache_ttl_seconds` is set
        3. **File Modification**: Automatic invalidation when source file mtime changes
        4. **Custom Validation**: Extensible via `_additional_cache_validation()` override

    Thread Safety:
        All cache operations are protected by:
            - `_lock`: `RLock` for synchronous operations
            - `_async_lock`: `AsyncLock` for async operations (if needed)

    Subclass Integration:
        **For V2/Converted Format:**
            - Use `_fetch_with_cache()` for standard fetch-and-cache pattern
            - Call `_get_from_cache()` to retrieve cached v2 data
            - Call `_store_in_cache()` to store fetched v2 data
            - Override `_get_file_path_for_validation()` to enable mtime validation
            - Override `_additional_cache_validation()` for custom validation logic

        **For Legacy Format:**
            - Call `_get_legacy_from_cache()` to retrieve cached legacy data (dict + string)
            - Call `_store_legacy_in_cache()` to store fetched legacy data
            - Override `_get_legacy_file_path_for_validation()` for legacy mtime validation

    Cache Query Methods:
        - `has_cached_data()`: Check if data exists (ignores validity)
        - `is_cache_valid()`: Check if cached data exists AND is valid
        - `needs_refresh()`: Check if existing cached data should be refetched
        - `should_fetch_data()`: Combined check for initial fetch OR refresh


    Examples:
        Basic fetch implementation using the cache helper::

            def fetch_category(
                self,
                category: MODEL_REFERENCE_CATEGORY,
                *,
                force_refresh: bool = False
            ) -> dict[str, Any] | None:
                return self._fetch_with_cache(
                    category,
                    lambda: self._fetch_from_source(category),
                    force_refresh=force_refresh
                )

        Custom validation with file path override::

            def _get_file_path_for_validation(
                self,
                category: MODEL_REFERENCE_CATEGORY
            ) -> Path | None:
                return self.data_dir / f"{category.value}.json"

            def _additional_cache_validation(
                self,
                category: MODEL_REFERENCE_CATEGORY
            ) -> bool:
                # Custom validation logic
                return self._check_data_integrity(category)
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
        self._async_lock: AsyncLock = AsyncLock()

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
        """Check if cached data exists and is still valid for the given category.

        This method performs comprehensive validation to determine if cached data can be
        used without refetching. It's primarily used internally by cache retrieval methods
        but can also be called directly for validation checks.

        Args:
            category: The category to validate.

        Returns:
            True if cache exists and all validation checks pass, False otherwise.

        Validation Steps:
            The method performs checks in the following order:

            1. **Explicit Staleness**: Returns False if category is in `_stale_categories`
            2. **Cache Existence**: Returns False if category has never been cached
            3. **Timestamp Existence**: Returns False if no timestamp recorded
            4. **TTL Expiration**: Checks if `cache_ttl_seconds` exceeded (calls `mark_stale()` if expired)
            5. **File Modification**: Compares current file mtime with cached mtime (calls `mark_stale()` if changed)
            6. **Custom Validation**: Calls `_additional_cache_validation()` for subclass-specific checks

        Side Effects:
            When staleness is detected (TTL expiration or mtime change), this method calls
            `mark_stale()` to trigger invalidation callbacks and notify the manager.

        Related Methods:
            - `has_cached_data()`: Simple existence check, ignores validity
            - `should_fetch_data()`: Combined check for "should I fetch?" (initial OR refresh)
            - `needs_refresh()`: Checks if cached data should be refetched (staleness only)

        Return Value Semantics:
            - Returns `False` for both "no data" and "stale data" cases
            - Use `has_cached_data()` to distinguish between these cases
            - Use `needs_refresh()` to check staleness without considering initial fetch


        Note:
            This method is thread-safe and uses the internal `_lock` for synchronization.
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
                self.mark_stale(category)
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
                    self.mark_stale(category)
                    return False
            except Exception:
                return False

        if not self._additional_cache_validation(category):
            logger.debug(f"Category {category} failed additional validation")
            return False

        return True

    @override
    def _mark_stale_impl(self, category: MODEL_REFERENCE_CATEGORY) -> None:
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
                self.mark_stale(category)
                return True

        file_path = self._get_file_path_for_validation(category)
        if file_path and file_path.exists():
            try:
                current_mtime = file_path.stat().st_mtime
                last_known = self._last_known_mtimes.get(category, 0.0)
                if current_mtime != last_known:
                    logger.debug(f"File {file_path.name} mtime changed, needs refresh")
                    self.mark_stale(category)
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
            # Only mark as fresh if we actually have data
            # None values indicate failed loads and should not prevent retries
            if data is not None:
                self._mark_category_fresh(category)
                logger.debug(f"Stored {category} in cache")
            else:
                logger.debug(f"Stored None for {category}, not marking as fresh")

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
            # Only mark as fresh if we actually have data
            # None values indicate failed loads and should not prevent retries
            if legacy_dict is not None or legacy_string is not None:
                self._mark_legacy_category_fresh(category)
                logger.debug(f"Stored legacy {category} in cache")
            else:
                logger.debug(f"Stored None for legacy {category}, not marking as fresh")

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
