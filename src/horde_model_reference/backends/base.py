"""Abstract base class for model reference backend providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

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

    def __init__(
        self,
        mode: ReplicateMode = ReplicateMode.REPLICA,
    ) -> None:
        """Initialize the backend."""
        super().__init__()

        self._replicate_mode = mode

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
            category (MODEL_REFERENCE_CATEGORY): The category to fetch.
            force_refresh (bool, optional): If True, bypass any backend-level caching
                and fetch fresh data. Defaults to False.

        Returns:
            dict[str, Any] | None: The model reference data as a dictionary mapping
                model names to their attributes, or None if the category cannot be fetched.
        """

    @abstractmethod
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch model reference data for all categories.

        Args:
            force_refresh (bool, optional): If True, bypass any backend-level caching
                and fetch fresh data. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]: A mapping of categories
                to their model reference data. Categories that cannot be fetched have None values.
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
            category (MODEL_REFERENCE_CATEGORY): The category to fetch.
            httpx_client (httpx.AsyncClient | None): An optional httpx async client.
            force_refresh (bool, optional): If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[str, Any] | None: The model reference data, or None if fetch failed.
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
            httpx_client (httpx.AsyncClient | None): An optional httpx async client.
            force_refresh (bool, optional): If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]: A mapping of categories to their data.
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
            category (MODEL_REFERENCE_CATEGORY): The category to check.

        Returns:
            bool: True if cached data needs refresh due to staleness. False if no
                  cached data exists or if cached data is still fresh.
        """

    @abstractmethod
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category's data as stale, requiring refresh on next access.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to mark as stale.
        """

    @abstractmethod
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's data, if applicable.

        Some backends (like file-based ones) have a physical file path associated
        with each category. Others (like database backends) may return None.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to get the path for.

        Returns:
            Path | None: The file path, or None if not applicable for this backend.
        """

    @abstractmethod
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories, if applicable.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: Mapping of categories to their file paths.
                Returns None values for categories without file paths.
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
            category (MODEL_REFERENCE_CATEGORY): Category to retrieve.
            redownload (bool): If True, redownload before returning and refresh cache.

        Returns:
            dict[str, Any] | None: The raw legacy JSON dict, or None if not found.
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
            category (MODEL_REFERENCE_CATEGORY): Category to retrieve.
            redownload (bool): If True, redownload before returning and refresh cache.

        Returns:
            str | None: The raw legacy JSON string, or None if not found.
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
            category (MODEL_REFERENCE_CATEGORY): The category to update.
            model_name (str): The name of the model to update or create.
            record_dict (dict[str, Any]): The model record data as a dictionary.

        Raises:
            NotImplementedError: If the backend does not support write operations.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support write operations")

    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference.

        This is an optional method that write-capable backends can implement.
        Read-only backends should leave the default implementation which raises NotImplementedError.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category containing the model.
            model_name (str): The name of the model to delete.

        Raises:
            NotImplementedError: If the backend does not support write operations.
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
            category (MODEL_REFERENCE_CATEGORY): The category to update.
            model_name (str): The name of the model to update or create.
            record_dict (dict[str, Any]): The model record data in legacy format as a dictionary.

        Raises:
            NotImplementedError: If the backend does not support legacy write operations.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support legacy write operations")

    def delete_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference from legacy format files.

        This is an optional method that legacy-write-capable backends can implement.
        Only available when canonical_format='legacy' in PRIMARY mode.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category containing the model.
            model_name (str): The name of the model to delete.

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
