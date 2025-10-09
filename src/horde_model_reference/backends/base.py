"""Abstract base class for model reference backend providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import aiohttp

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ModelReferenceBackend(ABC):
    """Abstract interface for model reference data providers.

    This interface defines the contract that all backend implementations must fulfill.
    Backends are responsible for fetching raw model reference data from their source
    (GitHub, database, API, etc.) and providing it in a standardized dictionary format.

    The ModelReferenceManager uses backends as pluggable data sources and handles
    caching, TTL management, and conversion to pydantic models.
    """

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
        aiohttp_client_session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a specific category.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to fetch.
            aiohttp_client_session (aiohttp.ClientSession): An existing aiohttp client session.
            force_refresh (bool, optional): If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[str, Any] | None: The model reference data, or None if fetch failed.
        """

    @abstractmethod
    async def fetch_all_categories_async(
        self,
        *,
        aiohttp_client_session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch model reference data for all categories.

        Args:
            aiohttp_client_session (aiohttp.ClientSession): An existing aiohttp client session.
            force_refresh (bool, optional): If True, bypass any backend-level caching. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]: A mapping of categories to their data.
        """

    @abstractmethod
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category's data needs to be refreshed.

        This method is used by the manager to determine if cached data is stale
        and should be refreshed from the backend. Backends can implement this
        based on file modification times, database timestamps, ETags, etc.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to check.

        Returns:
            bool: True if the data needs refresh, False otherwise.
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
