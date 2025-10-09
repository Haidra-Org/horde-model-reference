"""DEPRECATED: Legacy wrapper around LegacyGitHubBackend.

This module provides backwards compatibility for code using LegacyReferenceDownloadManager.
New code should use LegacyGitHubBackend directly or ModelReferenceManager with a backend.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

if TYPE_CHECKING:
    from horde_model_reference.backends import LegacyGitHubBackend


class LegacyReferenceDownloadManager:
    """DEPRECATED: Thin wrapper around LegacyGitHubBackend for backwards compatibility.

    This class is maintained for backwards compatibility only. New code should use
    LegacyGitHubBackend directly or ModelReferenceManager with a pluggable backend.

    All methods forward to the underlying LegacyGitHubBackend instance.
    """

    _backend: LegacyGitHubBackend
    _instance: LegacyReferenceDownloadManager | None = None

    def __new__(
        cls,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = horde_model_reference_settings.cache_ttl_seconds,
        replicate_mode: ReplicateMode = horde_model_reference_settings.replicate_mode,
        retry_max_attempts: int = horde_model_reference_settings.legacy_download_retry_max_attempts,
        retry_backoff_seconds: float = horde_model_reference_settings.legacy_download_retry_backoff_seconds,
    ) -> LegacyReferenceDownloadManager:
        """Return singleton instance (for backwards compatibility).

        Args:
            base_path: Base path for storing files.
            cache_ttl_seconds: Cache TTL in seconds.
            replicate_mode: REPLICA downloads from GitHub, PRIMARY uses local only.
            retry_max_attempts: Max retry attempts for downloads.
            retry_backoff_seconds: Backoff between retries.

        Returns:
            LegacyReferenceDownloadManager: Singleton instance.
        """
        if not cls._instance:
            warnings.warn(
                "LegacyReferenceDownloadManager is deprecated. "
                "Use LegacyGitHubBackend directly or ModelReferenceManager with a backend.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Lazy import to avoid circular dependency
            from horde_model_reference.backends import LegacyGitHubBackend

            cls._instance = super().__new__(cls)
            cls._instance._backend = LegacyGitHubBackend(
                base_path=base_path,
                cache_ttl_seconds=cache_ttl_seconds,
                replicate_mode=replicate_mode,
                retry_max_attempts=retry_max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
            )

        return cls._instance

    # Properties that forward to backend

    @property
    def base_path(self) -> Path:
        """Get base path from backend."""
        return self._backend.base_path

    @property
    def legacy_path(self) -> Path:
        """Get legacy path from backend."""
        return self._backend.legacy_path

    @property
    def _references_paths_cache(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get references paths cache from backend."""
        return self._backend._references_paths_cache

    @property
    def _references_cache(self) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] | None:
        """Get references cache from backend."""
        return self._backend._converted_cache

    @property
    def _categories_needing_redownload(self) -> dict[MODEL_REFERENCE_CATEGORY, bool]:
        """Get categories needing redownload as dict (converts from set).

        This property provides backwards compatibility for code that accesses
        the internal _categories_needing_redownload dict.
        """
        # Convert backend's set to dict for backwards compatibility
        return {cat: (cat in self._backend._stale_categories) for cat in MODEL_REFERENCE_CATEGORY}

    # Method forwards

    def download_legacy_model_reference(
        self,
        *,
        model_category_name: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> Path | None:
        """Download a single legacy model reference file.

        Args:
            model_category_name: Category to download.
            override_existing: If True, overwrite existing files.

        Returns:
            Path | None: Path to downloaded file, or None if failed.
        """
        # Backend's fetch_category does download + conversion, but for legacy compat
        # we just want the download part. Use internal method.
        return self._backend._download_legacy(model_category_name, override_existing=override_existing)

    async def download_legacy_model_reference_async(
        self,
        *,
        aiohttp_client_session: aiohttp.ClientSession,
        model_category_name: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> Path | None:
        """Asynchronously download a single legacy model reference file.

        Args:
            aiohttp_client_session: aiohttp client session.
            model_category_name: Category to download.
            override_existing: If True, overwrite existing files.

        Returns:
            Path | None: Path to downloaded file, or None if failed.
        """
        return await self._backend._download_legacy_async(
            model_category_name,
            aiohttp_client_session,
            override_existing=override_existing,
        )

    def download_all_legacy_model_references(
        self,
        *,
        overwrite_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Download all legacy model reference files.

        Args:
            overwrite_existing: If True, overwrite existing files.

        Returns:
            dict: Mapping of categories to their downloaded file paths.
        """
        result: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = self.download_legacy_model_reference(
                model_category_name=category,
                override_existing=overwrite_existing,
            )

        # Trigger conversion (for backwards compatibility)
        from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references

        convert_all_legacy_model_references()

        return result

    async def download_all_legacy_model_references_async(
        self,
        *,
        aiohttp_client_session: aiohttp.ClientSession,
        overwrite_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Asynchronously download all legacy model reference files.

        Args:
            aiohttp_client_session: aiohttp client session.
            overwrite_existing: If True, overwrite existing files.

        Returns:
            dict: Mapping of categories to their downloaded file paths.
        """
        result: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}

        # Download all in parallel
        import asyncio

        tasks = []
        categories: list[MODEL_REFERENCE_CATEGORY] = list(MODEL_REFERENCE_CATEGORY)
        for category in categories:
            tasks.append(
                self.download_legacy_model_reference_async(
                    aiohttp_client_session=aiohttp_client_session,
                    model_category_name=category,
                    override_existing=overwrite_existing,
                )
            )

        results = await asyncio.gather(*tasks)

        for category, task_result in zip(categories, results, strict=True):
            result[category] = task_result

        # Trigger conversion (for backwards compatibility)
        from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references

        convert_all_legacy_model_references()

        return result

    def get_all_legacy_model_references_paths(
        self,
        *,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get paths to all legacy model reference files.

        Args:
            redownload_all: If True, redownload all files first.

        Returns:
            dict: Mapping of categories to their file paths.
        """
        if redownload_all:
            self.download_all_legacy_model_references(overwrite_existing=True)

        return self._backend._references_paths_cache.copy()

    def get_all_legacy_model_references(
        self,
        *,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Get all legacy model reference data as dictionaries.

        Args:
            redownload_all: If True, redownload all files first.

        Returns:
            dict: Mapping of categories to their legacy reference data.
        """
        if redownload_all:
            self.download_all_legacy_model_references(overwrite_existing=True)

        # For backwards compat, we need to return LEGACY format data, not converted
        # But the backend only caches converted data. So we read legacy files directly.
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        import json

        for category, file_path in self._backend._references_paths_cache.items():
            if file_path and file_path.exists():
                try:
                    with open(file_path) as f:
                        result[category] = json.load(f)
                except Exception:
                    result[category] = None
            else:
                result[category] = None

        return result

    def is_downloaded(self, model_category_name: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category has been downloaded.

        Args:
            model_category_name: Category to check.

        Returns:
            bool: True if downloaded, False otherwise.
        """
        path = self._backend._references_paths_cache.get(model_category_name)
        return path is not None and path.exists()

    def is_all_downloaded(self) -> bool:
        """Check if all categories have been downloaded.

        Returns:
            bool: True if all downloaded, False otherwise.
        """
        return all(self.is_downloaded(cat) for cat in MODEL_REFERENCE_CATEGORY)

    def get_legacy_model_reference_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get raw legacy JSON for a specific category without pydantic validation.

        This method returns the legacy format JSON data from cache or disk, avoiding
        the overhead of pydantic validation. Ideal for API endpoints.

        Args:
            category: Category to retrieve.
            redownload: If True, redownload before returning and invalidate cache.

        Returns:
            dict[str, Any] | None: The raw legacy JSON dict, or None if not found.
        """
        # Delegate to backend which handles disk I/O and caching
        return self._backend.get_legacy_json(category, redownload=redownload)

    def get_legacy_model_reference_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get raw legacy JSON string for a specific category without pydantic validation.

        This method returns the legacy format JSON data as a string from cache or disk, avoiding
        the overhead of pydantic validation. Ideal for API endpoints.

        Args:
            category: Category to retrieve.
            redownload: If True, redownload before returning and invalidate cache.
        Returns:
            str | None: The raw legacy JSON string, or None if not found.
        """
        # Delegate to backend which handles disk I/O and caching
        return self._backend.get_legacy_json_string(category, redownload=redownload)
