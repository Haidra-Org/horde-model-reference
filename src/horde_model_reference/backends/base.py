"""Abstract base class for model reference backend providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel

from horde_model_reference import ReplicateMode
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ModelReferenceBackend(ABC):
    """Abstract interface for model reference data providers.

    This interface defines the contract that all backend implementations must fulfill.
    Backends are responsible for fetching raw model reference data from their source
    (GitHub, database, API, etc.) and providing it in a standardized dictionary format.

    The ModelReferenceManager uses backends as pluggable data sources and handles
    caching, TTL management, and conversion to pydantic models.
    """

    _replicate_mode = ReplicateMode.REPLICA
    _invalidation_callbacks: list[Callable[[MODEL_REFERENCE_CATEGORY], None]]

    def __init__(
        self,
        mode: ReplicateMode = ReplicateMode.REPLICA,
    ) -> None:
        """Initialize the backend."""
        super().__init__()

        self._replicate_mode = mode
        self._invalidation_callbacks = []

    @property
    def replicate_mode(self) -> ReplicateMode:
        """Get the replicate mode of this backend."""
        return self._replicate_mode

    @abstractmethod
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch model reference data for a specific category.

        Args:
            category: The category to fetch.
            force_refresh: If True, bypass any backend-level caching
                and fetch fresh data. Defaults to False.

        Returns:
            dict[str, Any] | None: The model reference data as a dictionary mapping
                model names to their attributes, or None if the category cannot be fetched.

        Implementation Requirements:
            - Return data as a dictionary: `{model_name: {attribute: value, ...}, ...}`
            - Return `None` if category cannot be fetched
            - Honor `force_refresh` to bypass internal caches
            - Handle errors gracefully (log and return `None`)

        Example Implementation:
            ```python
            def fetch_category(self, category, *, force_refresh=False):
                def fetch():
                    # Your fetch logic here
                    response = httpx.get(f"{self.base_url}/{category}")
                    return response.json() if response.status_code == 200 else None

                return self._fetch_with_cache(category, fetch, force_refresh=force_refresh)
            ```

        See Also:
            - [fetch_all_categories()][(c).fetch_all_categories]: Batch fetching of all categories
            - [fetch_category_async()][(c).fetch_category_async]: Async variant
            - [ReplicaBackendBase._fetch_with_cache()]
              [^^^.replica_backend_base.ReplicaBackendBase._fetch_with_cache]:
              Helper for cache management
        """

    @abstractmethod
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch model reference data for all categories.

        Args:
            force_refresh: If True, bypass any backend-level caching
                and fetch fresh data. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]: A mapping of categories
                to their model reference data. Categories that cannot be fetched have None values.

        Implementation Requirements:
            - Return a dictionary mapping each category to its data
            - Use `None` values for categories that cannot be fetched
            - Typically implemented as a loop over [fetch_category()][(c).fetch_category]

        Example Implementation:
            ```python
            def fetch_all_categories(self, *, force_refresh=False):
                return {
                    category: self.fetch_category(category, force_refresh=force_refresh)
                    for category in MODEL_REFERENCE_CATEGORY
                }
            ```

        See Also:
            - [fetch_category()][(c).fetch_category]: Single category fetching
            - [fetch_all_categories_async()][(c).fetch_all_categories_async]: Async variant
        """

    @abstractmethod
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a specific category.

        Args:
            category: The category to fetch.
            httpx_client: An optional httpx async client for connection pooling.
            force_refresh: If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[str, Any] | None: The model reference data, or None if fetch failed.

        Implementation Requirements:
            - Use async I/O where possible (network requests, file operations with aiofiles)
            - Accept optional `httpx_client` for connection pooling
            - Create temporary client if not provided
            - Same return format as synchronous version
            - Share cache with synchronous methods when using
              [ReplicaBackendBase][^^^.replica_backend_base.ReplicaBackendBase]

        Example Implementation:
            ```python
            async def fetch_category_async(self, category, *, httpx_client=None, force_refresh=False):
                close_client = httpx_client is None
                if httpx_client is None:
                    httpx_client = httpx.AsyncClient()

                try:
                    response = await httpx_client.get(f"{self.base_url}/{category}")
                    return response.json() if response.status_code == 200 else None
                finally:
                    if close_client:
                        await httpx_client.aclose()
            ```

        See Also:
            - [fetch_category()][(c).fetch_category]: Synchronous variant
            - [fetch_all_categories_async()][(c).fetch_all_categories_async]: Async batch fetching
        """

    @abstractmethod
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch model reference data for all categories.

        Args:
            httpx_client: An optional httpx async client for connection pooling.
            force_refresh: If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]: A mapping of categories to their data.

        Implementation Requirements:
            - Use `asyncio.gather()` for concurrent fetching when possible
            - Share `httpx_client` across fetches for connection pooling
            - Same return format as synchronous version

        Example Implementation:
            ```python
            async def fetch_all_categories_async(self, *, httpx_client=None, force_refresh=False):
                close_client = httpx_client is None
                if httpx_client is None:
                    httpx_client = httpx.AsyncClient()

                try:
                    tasks = [
                        self.fetch_category_async(cat, httpx_client=httpx_client, force_refresh=force_refresh)
                        for cat in MODEL_REFERENCE_CATEGORY
                    ]
                    results = await asyncio.gather(*tasks)
                    return dict(zip(MODEL_REFERENCE_CATEGORY, results, strict=False))
                finally:
                    if close_client:
                        await httpx_client.aclose()
            ```

        See Also:
            - [fetch_all_categories()][(c).fetch_all_categories]: Synchronous variant
            - [fetch_category_async()][(c).fetch_category_async]: Async single category fetch
        """

    @abstractmethod
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if existing cached data for a category needs to be refreshed.

        This method indicates whether cached data has become stale and should be
        re-fetched from the source. It does NOT indicate whether an initial fetch
        is needed for data that has never been loaded.

        Semantic distinction:
        - Returns False when no cached data exists (no initial fetch needed via this method)
        - Returns True when cached data exists but has become stale

        Backends can implement staleness detection based on file modification times,
        database timestamps, ETags, TTL expiration, explicit invalidation, etc.

        Args:
            category: The category to check.

        Returns:
            bool: True if cached data needs refresh due to staleness. False if no
                  cached data exists or if cached data is still fresh.

        Implementation Requirements:
            - Return `False` if no data has been cached yet (initial fetch is different concern)
            - Return `True` if cached data exists but is stale
            - Staleness can be based on: TTL expiration, file mtime changes, ETags, explicit invalidation

        Note:
            [ReplicaBackendBase][^^^.replica_backend_base.ReplicaBackendBase]
            provides a concrete implementation that checks explicit staleness marking,
            TTL expiration, file mtime changes, and custom validation hooks.

        See Also:
            - [mark_stale()][(c).mark_stale]: Explicitly mark a category as stale
            - [ReplicaBackendBase.should_fetch_data()][^^^.replica_backend_base.ReplicaBackendBase.should_fetch_data]:
              Combined initial fetch + refresh check
        """

    def register_invalidation_callback(
        self,
        callback: Callable[[MODEL_REFERENCE_CATEGORY], None],
    ) -> None:
        """Register a callback to be called when a category is invalidated.

        This allows external components (like ModelReferenceManager) to be notified
        when cached data becomes stale and needs to be refreshed.

        Args:
            callback: Function to call with the invalidated category.
        """
        self._invalidation_callbacks.append(callback)
        logger.debug(f"Registered invalidation callback: {callback.__name__}")

    def _notify_invalidation(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Notify all registered callbacks that a category has been invalidated.

        Args:
            category: The category that was invalidated.
        """
        for callback in self._invalidation_callbacks:
            try:
                callback(category)
            except Exception as e:
                logger.error(f"Invalidation callback {callback.__name__} failed for {category}: {e}")

    @abstractmethod
    def _mark_stale_impl(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Backend-specific implementation of marking a category as stale.

        Subclasses must implement this to handle their specific staleness tracking.

        Args:
            category: The category to mark as stale.

        Implementation Requirements:
            - Update backend-specific staleness tracking (e.g., add to `_stale_categories` set)
            - Called by public [mark_stale()][(c).mark_stale] method before notifying callbacks
            - Implementations should override this method, not `mark_stale()`

        Note:
            The public `mark_stale()` method calls this implementation and then automatically
            notifies all registered invalidation callbacks.
        """

    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category's data as stale, requiring refresh on next access.

        This method calls the backend-specific implementation and then notifies
        all registered callbacks.

        Args:
            category: The category to mark as stale.

        Implementation Note:
            The base class provides this public implementation. Subclasses should override
            [_mark_stale_impl()][(c)._mark_stale_impl]
            instead of this method.

        See Also:
            - [_mark_stale_impl()][(c)._mark_stale_impl]: Backend-specific staleness tracking
            - [register_invalidation_callback()][(c).register_invalidation_callback]:
              Register callbacks for invalidation events
        """
        self._mark_stale_impl(category)
        self._notify_invalidation(category)

    @abstractmethod
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's data, if applicable.

        Some backends (like file-based ones) have a physical file path associated
        with each category. Others (like database backends) may return None.

        Args:
            category: The category to get the path for.

        Returns:
            Path | None: The file path, or None if not applicable for this backend.

        Implementation Requirements:
            - Return `Path` object for file-based backends
            - Return `None` for backends without file storage (HTTP-only, database, etc.)

        Example Implementations:
            ```python
            # File-based backend
            def get_category_file_path(self, category):
                return self.base_path / f"{category.value}.json"

            # HTTP-only backend
            def get_category_file_path(self, category):
                return None
            ```

        See Also:
            - [get_all_category_file_paths()][(c).get_all_category_file_paths]: Get all file paths
        """

    @abstractmethod
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories, if applicable.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: Mapping of categories to their file paths.
                Returns None values for categories without file paths.

        Implementation Requirements:
            - Return dictionary with all categories
            - Use `None` values for categories without file paths
            - Typically implemented by iterating over categories and calling `get_category_file_path()`

        Example Implementation:
            ```python
            def get_all_category_file_paths(self):
                return {
                    category: self.get_category_file_path(category)
                    for category in MODEL_REFERENCE_CATEGORY
                }
            ```

        See Also:
            - [get_category_file_path()][(c).get_category_file_path]: Get single category file path
        """

    @abstractmethod
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get raw legacy JSON for a specific category without pydantic validation.

        This method returns cached legacy format JSON data, downloading if needed.
        The cache is populated during initialization, downloads, and on-demand loads.

        Args:
            category: Category to retrieve.
            redownload: If True, redownload before returning and refresh cache.

        Returns:
            dict[str, Any] | None: The raw legacy JSON dict, or None if not found.

        Implementation Requirements:
            - Return legacy format data as dictionary
            - Support caching with optional redownload
            - Return `None` if not available
            - The `redownload` parameter is analogous to `force_refresh` in fetch methods

        See Also:
            - [get_legacy_json_string()][(c).get_legacy_json_string]: Get as string instead of dict
        """

    @abstractmethod
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get raw legacy JSON string for a specific category without pydantic validation.

        This method returns cached legacy format JSON data as a string, downloading if needed.
        The cache is populated during initialization, downloads, and on-demand loads.

        Args:
            category: Category to retrieve.
            redownload: If True, redownload before returning and refresh cache.

        Returns:
            str | None: The raw legacy JSON string, or None if not found.

        Implementation Requirements:
            - Return legacy format data as JSON string
            - Same caching semantics as [get_legacy_json()][(c).get_legacy_json]
            - Return `None` if not available

        See Also:
            - [get_legacy_json()][(c).get_legacy_json]: Get as dict instead of string
        """

    def supports_writes(self) -> bool:
        """Check if this backend supports write operations (v2 format).

        Write operations include update_model() and delete_model().
        Typically only PRIMARY mode backends support writes.

        Returns:
            bool: True if write operations are supported, False otherwise.
        """
        return False

    def supports_legacy_writes(self) -> bool:
        """Check if this backend supports write operations in legacy format.

        Legacy write operations include update_model_legacy() and delete_model_legacy().
        Only available when canonical_format='legacy' in PRIMARY mode.

        Returns:
            bool: True if legacy write operations are supported, False otherwise.
        """
        return False

    def supports_cache_warming(self) -> bool:
        """Check if this backend supports cache warming operations.

        Cache warming pre-populates the cache with data to improve initial request performance.
        Typically only backends with distributed caching (like Redis) support this.

        Returns:
            bool: True if cache warming is supported, False otherwise.
        """
        return False

    def supports_health_checks(self) -> bool:
        """Check if this backend supports health check operations.

        Health checks verify that the backend's external dependencies (Redis, databases, etc.)
        are accessible and functioning correctly.

        Returns:
            bool: True if health checks are supported, False otherwise.
        """
        return False

    def supports_statistics(self) -> bool:
        """Check if this backend supports statistics retrieval.

        Statistics provide insights into backend performance, cache hits/misses, etc.

        Returns:
            bool: True if statistics are supported, False otherwise.
        """
        return False

    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference.

        This is an optional method that write-capable backends can implement.
        Read-only backends should leave the default implementation which raises NotImplementedError.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data as a dictionary.

        Raises:
            NotImplementedError: If the backend does not support write operations.

        Implementation Requirements:
            - Create model if it doesn't exist
            - Update model if it exists
            - Ensure atomic writes (use temp files with rename for file-based backends)
            - Call [mark_stale()][(c).mark_stale] after successful write to invalidate cache
            - Override [supports_writes()][(c).supports_writes] to return `True`

        See Also:
            - [update_model_from_base_model()][(c).update_model_from_base_model]:
              Update from pydantic model (automatically provided)
            - [delete_model()][(c).delete_model]: Delete a model
            - [supports_writes()][(c).supports_writes]: Feature detection method
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support write operations")

    def update_model_from_base_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_model: BaseModel,
    ) -> None:
        """Update or create a model reference from a pydantic BaseModel.

        This is an optional method that write-capable backends can implement.
        Read-only backends should leave the default implementation which raises NotImplementedError.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_model: The model record data as a pydantic BaseModel.

        Raises:
            NotImplementedError: If the backend does not support write operations.

        Implementation Note:
            The base class provides this implementation automatically. It:
            1. Checks [supports_writes()][(c).supports_writes] returns `True`
            2. Converts the pydantic model to dict using `model_dump(exclude_unset=True)`
            3. Calls [update_model()][(c).update_model] with the dictionary

            Backends that support writes typically don't need to override this method.

        See Also:
            - [update_model()][(c).update_model]: Update from dictionary (implement this)
            - [supports_writes()][(c).supports_writes]: Feature detection method
        """
        if not self.supports_writes():
            raise NotImplementedError(f"{self.__class__.__name__} does not support write operations")

        record_dict = record_model.model_dump(exclude_unset=True)
        self.update_model(category, model_name, record_dict)

    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference.

        This is an optional method that write-capable backends can implement.
        Read-only backends should leave the default implementation which raises NotImplementedError.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            NotImplementedError: If the backend does not support write operations.
            KeyError: If the model doesn't exist.

        Implementation Requirements:
            - Raise `KeyError` if model doesn't exist
            - Ensure atomic writes
            - Call [mark_stale()][(c).mark_stale] after successful delete to invalidate cache
            - Override [supports_writes()][(c).supports_writes] to return `True`

        See Also:
            - [update_model()][(c).update_model]: Update or create a model
            - [supports_writes()][(c).supports_writes]: Feature detection method
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support write operations")

    def update_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference in legacy format.

        This is an optional method that legacy-write-capable backends can implement.
        Only available when canonical_format='legacy' in PRIMARY mode.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data in legacy format as a dictionary.

        Raises:
            NotImplementedError: If the backend does not support legacy write operations.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support legacy write operations")

    def update_model_legacy_from_base_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_model: BaseModel,
    ) -> None:
        """Update or create a model reference in legacy format from a pydantic BaseModel.

        This is an optional method that legacy-write-capable backends can implement.
        Only available when canonical_format='legacy' in PRIMARY mode.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_model: The model record data as a pydantic BaseModel.

        Raises:
            NotImplementedError: If the backend does not support legacy write operations.
        """
        if not self.supports_legacy_writes():
            raise NotImplementedError(f"{self.__class__.__name__} does not support legacy write operations")

        record_dict = record_model.model_dump(exclude_unset=True)
        self.update_model_legacy(category, model_name, record_dict)

    def delete_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference from legacy format files.

        This is an optional method that legacy-write-capable backends can implement.
        Only available when canonical_format='legacy' in PRIMARY mode.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            NotImplementedError: If the backend does not support legacy write operations.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support legacy write operations")

    def warm_cache(self) -> None:
        """Pre-populate cache with all categories for faster initial requests.

        This is an optional method that backends with cache warming support can implement.
        Backends without cache warming should leave the default implementation.

        Raises:
            NotImplementedError: If the backend does not support cache warming.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support cache warming")

    async def warm_cache_async(self) -> None:
        """Asynchronously pre-populate cache with all categories for faster initial requests.

        This is an optional method that backends with cache warming support can implement.
        Backends without cache warming should leave the default implementation.

        Raises:
            NotImplementedError: If the backend does not support async cache warming.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support async cache warming")

    def health_check(self) -> bool:
        """Check the health of the backend's external dependencies.

        This is an optional method that backends with health check support can implement.
        Backends without external dependencies should leave the default implementation.

        Returns:
            bool: True if healthy, False otherwise.

        Raises:
            NotImplementedError: If the backend does not support health checks.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support health checks")

    def get_statistics(self) -> dict[str, Any]:
        """Get backend performance and usage statistics.

        This is an optional method that backends with statistics support can implement.
        The structure of returned statistics is backend-specific.

        Returns:
            dict[str, Any]: Backend-specific statistics.

        Raises:
            NotImplementedError: If the backend does not support statistics.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support statistics")

    def get_replicate_mode(self) -> ReplicateMode:
        """Get the replication mode of this backend.

        Returns:
            ReplicateMode: The replicate mode (PRIMARY or REPLICA).
        """
        return self._replicate_mode
