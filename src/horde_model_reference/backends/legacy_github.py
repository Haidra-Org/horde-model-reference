"""Legacy GitHub-based backend for model references.

This backend downloads legacy model reference files from GitHub repositories,
converts them to the new format, and provides them to the ModelReferenceManager.
"""

from __future__ import annotations

import asyncio
import json
import time
from asyncio import Lock as AsyncLock
from pathlib import Path
from threading import RLock
from typing import Any

import aiofiles
import aiohttp
import requests
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references
from horde_model_reference.meta_consts import (
    MODEL_REFERENCE_CATEGORY,
    github_image_model_reference_categories,
    github_text_model_reference_categories,
)
from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER_NAME


class LegacyGitHubBackend(ModelReferenceBackend):
    """Backend that fetches legacy model references from GitHub and converts them.

    This backend:
    1. Downloads legacy JSON files from AI-Horde GitHub repositories
    2. Stores them in a local legacy folder
    3. Converts them to the new format using legacy converters
    4. Returns the converted (new format) data

    The backend maintains its own internal cache for performance, but the
    ModelReferenceManager handles the primary caching strategy.
    """

    def __init__(
        self,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = horde_model_reference_settings.cache_ttl_seconds,
        replicate_mode: ReplicateMode = horde_model_reference_settings.replicate_mode,
        retry_max_attempts: int = horde_model_reference_settings.legacy_download_retry_max_attempts,
        retry_backoff_seconds: float = horde_model_reference_settings.legacy_download_retry_backoff_seconds,
    ) -> None:
        """Initialize the Legacy GitHub backend.

        Args:
            base_path (str | Path, optional): Base path for storing model reference files.
            cache_ttl_seconds (int, optional): TTL for internal cache in seconds.
            replicate_mode (ReplicateMode, optional): REPLICA downloads from GitHub, PRIMARY uses local only.
            retry_max_attempts (int, optional): Max download retry attempts.
            retry_backoff_seconds (float, optional): Backoff time between retries.
        """
        self.base_path = Path(base_path)
        self.legacy_path = self.base_path.joinpath(LEGACY_REFERENCE_FOLDER_NAME)
        self._replicate_mode = replicate_mode
        self._cache_ttl_seconds = cache_ttl_seconds
        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

        # Thread safety
        self._lock = RLock()
        self._async_lock = AsyncLock()

        # Internal caches
        self._references_paths_cache: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}
        self._converted_cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] | None = None
        self._legacy_cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        self._legacy_string_cache: dict[MODEL_REFERENCE_CATEGORY, str | None] = {}
        self._cache_timestamp: float = 0.0

        # Tracking
        self._last_known_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._stale_categories: set[MODEL_REFERENCE_CATEGORY] = set()
        self._times_downloaded: dict[MODEL_REFERENCE_CATEGORY, int] = {}

        # Initialize path cache and mtime tracking
        for category in MODEL_REFERENCE_CATEGORY:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.legacy_path,
            )

            if file_path.exists():
                self._references_paths_cache[category] = file_path
                try:
                    self._last_known_mtimes[category] = file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                # Populate legacy cache during initialization
                self._load_legacy_json_from_disk(category, file_path)
            else:
                if self._replicate_mode == ReplicateMode.REPLICA:
                    self._references_paths_cache[category] = None
                    self._legacy_cache[category] = None
                    self._legacy_string_cache[category] = None
                else:
                    raise FileNotFoundError(f"Model reference file not found for {category}.")

            self._times_downloaded[category] = 0

    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch model reference data for a specific category.

        Downloads legacy format from GitHub, converts to new format, and returns it.

        Args:
            category: The category to fetch.
            force_refresh: If True, force download even if file exists.

        Returns:
            dict[str, Any] | None: The converted model reference data (new format).
        """
        with self._lock:
            # Download legacy file if needed
            if force_refresh or category in self._stale_categories:
                self._download_and_convert_single(category, override_existing=force_refresh)

            # Return from cache
            return self._get_cached_category(category)

    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch model reference data for all categories.

        Downloads all legacy files from GitHub, converts them, and returns new format data.

        Args:
            force_refresh: If True, force download all files.

        Returns:
            dict mapping categories to their converted model reference data.
        """
        with self._lock:
            # Download and convert all
            if force_refresh or any(cat in self._stale_categories for cat in MODEL_REFERENCE_CATEGORY):
                self._download_and_convert_all(overwrite_existing=force_refresh)

            # Build and return cache
            return self._build_converted_cache()

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        aiohttp_client_session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a category.

        Args:
            category: The category to fetch.
            aiohttp_client_session: aiohttp client session for downloads.
            force_refresh: If True, force download.

        Returns:
            dict[str, Any] | None: The converted model reference data.
        """
        async with self._async_lock:
            # Download legacy file if needed
            if force_refresh or category in self._stale_categories:
                await self._download_legacy_async(
                    category,
                    aiohttp_client_session,
                    override_existing=force_refresh,
                )
                # Convert happens synchronously after download
                if self._is_cache_expired():
                    convert_all_legacy_model_references()

            # Return from cache
            return self._get_cached_category(category)

    async def fetch_all_categories_async(
        self,
        *,
        aiohttp_client_session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories.

        Args:
            aiohttp_client_session: aiohttp client session.
            force_refresh: If True, force download all.

        Returns:
            dict mapping categories to their data.
        """
        async with self._async_lock:
            # Download all categories in parallel
            tasks = []
            for category in MODEL_REFERENCE_CATEGORY:
                override = force_refresh or category in self._stale_categories
                tasks.append(
                    self._download_legacy_async(
                        category,
                        aiohttp_client_session,
                        override_existing=override,
                    )
                )

            await asyncio.gather(*tasks)

            # Convert all (synchronous)
            if self._is_cache_expired():
                convert_all_legacy_model_references()

            # Return cache
            return self._build_converted_cache()

    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category needs refresh.

        Args:
            category: The category to check.

        Returns:
            bool: True if needs refresh (stale or mtime changed).
        """
        with self._lock:
            # Check if marked stale
            if category in self._stale_categories:
                return True

            # Check mtime
            file_path = self._references_paths_cache.get(category)
            if file_path and file_path.exists():
                try:
                    current_mtime = file_path.stat().st_mtime
                    last_known = self._last_known_mtimes.get(category, 0.0)
                    if current_mtime != last_known:
                        logger.debug(f"Legacy file {file_path.name} mtime changed, needs refresh")
                        return True
                except Exception:
                    pass

            return False

    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category as stale, requiring refresh.

        Args:
            category: The category to mark stale.
        """
        with self._lock:
            logger.debug(f"Marking category {category} as stale")
            self._stale_categories.add(category)

    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's converted data.

        Args:
            category: The category to get path for.

        Returns:
            Path | None: Path to the converted (new format) file, or None if not available.
        """
        # Return path to the CONVERTED file, not the legacy file
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories' converted data.

        Returns:
            dict: Mapping of categories to their converted file paths.
        """
        return horde_model_reference_paths.get_all_model_reference_file_paths(base_path=self.base_path)

    # Internal methods

    def _is_cache_expired(self) -> bool:
        """Check if internal cache has expired."""
        if self._converted_cache is None:
            return True
        return (time.time() - self._cache_timestamp) > self._cache_ttl_seconds

    def _get_cached_category(self, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any] | None:
        """Get a single category from cache."""
        if self._converted_cache is None or self._is_cache_expired():
            self._build_converted_cache()

        return self._converted_cache.get(category) if self._converted_cache else None

    def _load_legacy_json_from_disk(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        file_path: Path,
    ) -> dict[str, Any] | None:
        """Load legacy JSON from disk and populate both dict and string caches.

        Args:
            category: The category to load.
            file_path: Path to the legacy JSON file.

        Returns:
            dict[str, Any] | None: The loaded JSON data, or None on error.
        """
        if not file_path.exists():
            logger.debug(f"Legacy file {file_path} does not exist")
            self._legacy_cache[category] = None
            self._legacy_string_cache[category] = None
            return None

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # Use ujson for faster parsing
            try:
                import ujson

                data: dict[str, Any] = ujson.loads(content)
            except ImportError:
                data = json.loads(content)

            # Store both dict and string in cache
            self._legacy_cache[category] = data
            self._legacy_string_cache[category] = content.decode("utf-8")
            logger.debug(f"Loaded legacy JSON for category {category!r} from {file_path!r}")
            return data
        except Exception:
            logger.exception(f"Failed to load legacy JSON for category {category!r} from {file_path!r}")
            self._legacy_cache[category] = None
            self._legacy_string_cache[category] = None
            return None

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
        """
        with self._lock:
            if redownload:
                # Download will populate the cache
                self._download_legacy(category, override_existing=True)

            # Check if we have cached data
            if category in self._legacy_cache:
                logger.debug(f"Returning cached legacy JSON for category {category!r}")
                return self._legacy_cache[category]

            # Load from disk if not cached (shouldn't happen often after init)
            file_path = self._references_paths_cache.get(category)
            if not file_path:
                self._legacy_cache[category] = None
                return None

            return self._load_legacy_json_from_disk(category, file_path)

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
        """
        with self._lock:
            if redownload:
                # Download will populate the cache
                self._download_legacy(category, override_existing=True)

            # Check if we have cached string data
            if category in self._legacy_string_cache:
                cached_value = self._legacy_string_cache[category]
                if cached_value is not None:
                    logger.debug(f"Returning cached legacy JSON string for category {category!r}")
                    return cached_value

            # Load from disk if not cached (shouldn't happen often after init)
            file_path = self._references_paths_cache.get(category)
            if not file_path:
                self._legacy_string_cache[category] = None
                return None

            # _load_legacy_json_from_disk populates both caches
            self._load_legacy_json_from_disk(category, file_path)
            return self._legacy_string_cache.get(category)

    def _build_converted_cache(self) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Build cache from converted (new format) files on disk."""
        logger.debug("Building converted references cache from disk")

        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            # Read from CONVERTED files, not legacy files
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if file_path and file_path.exists():
                try:
                    with open(file_path) as f:
                        result[category] = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading converted file {file_path}: {e}")
                    result[category] = None
            else:
                result[category] = None

        self._converted_cache = result
        self._cache_timestamp = time.time()

        return result

    def _download_and_convert_single(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> None:
        """Download a single category's legacy file and convert it."""
        self._download_legacy(category, override_existing=override_existing)
        # Conversion happens for all categories at once
        if self._is_cache_expired():
            convert_all_legacy_model_references()
        # Remove from stale set
        self._stale_categories.discard(category)

    def _download_and_convert_all(self, overwrite_existing: bool = False) -> None:
        """Download all legacy files and convert them."""
        for category in MODEL_REFERENCE_CATEGORY:
            override = overwrite_existing or category in self._stale_categories
            self._download_legacy(category, override_existing=override)

        # Convert all at once
        if self._is_cache_expired():
            convert_all_legacy_model_references()

        # Clear stale set
        self._stale_categories.clear()

    def _download_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> Path | None:
        """Download a single legacy file from GitHub (synchronous)."""
        if self._replicate_mode != ReplicateMode.REPLICA:
            logger.debug(f"Replicate mode is not REPLICA, skipping download for {category}")
            return self._references_paths_cache.get(category)

        target_file_path = horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.legacy_path,
        )

        with self._lock:
            if target_file_path.exists() and not override_existing:
                logger.debug(f"Legacy file {target_file_path} already exists, skipping download")
                return target_file_path

            # Get GitHub URL
            target_url: str | None = None
            if category in github_image_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_image_model_github_urls[category]
            elif category in github_text_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_text_model_github_urls[category]
            else:
                logger.debug(f"No known GitHub URL for {category}")
                return None

            # Retry loop
            for attempt in range(1, self.retry_max_attempts + 1):
                if attempt > 1:
                    logger.debug(
                        f"Retrying download of {category} in {self.retry_backoff_seconds}s "
                        f"(attempt {attempt}/{self.retry_max_attempts})"
                    )
                    time.sleep(self.retry_backoff_seconds)

                response = requests.get(target_url, timeout=30)

                if response.status_code != 200:
                    logger.error(f"Failed to download {category}: HTTP {response.status_code}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                try:
                    # Use ujson for faster parsing if available
                    try:
                        import ujson

                        data = ujson.loads(response.content)
                    except ImportError:
                        data = json.loads(response.content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse {category} as JSON")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                # Write file
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file_path, "wb") as f:
                    f.write(response.content)

                # Update tracking
                self._times_downloaded[category] += 1
                if self._times_downloaded[category] > 1:
                    logger.debug(f"Downloaded {category} {self._times_downloaded[category]} times")

                logger.info(f"Downloaded {category} to {target_file_path}")
                self._references_paths_cache[category] = target_file_path

                # Update mtime
                try:
                    self._last_known_mtimes[category] = target_file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                # Populate both dict and string caches with downloaded data
                self._legacy_cache[category] = data
                self._legacy_string_cache[category] = response.content.decode("utf-8")
                logger.debug(f"Populated legacy cache for {category} after download")

                return target_file_path

            return None

    async def _download_legacy_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        aiohttp_client_session: aiohttp.ClientSession,
        override_existing: bool = False,
    ) -> Path | None:
        """Download a single legacy file from GitHub (asynchronous)."""
        if self._replicate_mode != ReplicateMode.REPLICA:
            logger.debug(f"Replicate mode is not REPLICA, skipping download for {category}")
            return self._references_paths_cache.get(category)

        target_file_path = horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.legacy_path,
        )

        if target_file_path.exists() and not override_existing:
            logger.debug(f"Legacy file {target_file_path} already exists, skipping download")
            return target_file_path

        # Get GitHub URL
        target_url: str | None = None
        if category in github_image_model_reference_categories:
            target_url = horde_model_reference_paths.legacy_image_model_github_urls[category]
        elif category in github_text_model_reference_categories:
            target_url = horde_model_reference_paths.legacy_text_model_github_urls[category]
        else:
            logger.debug(f"No known GitHub URL for {category}")
            return None

        # Retry loop
        for attempt in range(1, self.retry_max_attempts + 1):
            if attempt > 1:
                logger.debug(
                    f"Retrying download of {category} in {self.retry_backoff_seconds}s "
                    f"(attempt {attempt}/{self.retry_max_attempts})"
                )
                await asyncio.sleep(self.retry_backoff_seconds)

            async with aiohttp_client_session.get(target_url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {category}: HTTP {response.status}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                content = await response.read()

                try:
                    # Use ujson for faster parsing if available
                    try:
                        import ujson

                        data = ujson.loads(content)
                    except ImportError:
                        data = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse {category} as JSON")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                # Write file
                target_file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    async with aiofiles.open(target_file_path, "wb") as f:
                        await f.write(content)
                except Exception as e:
                    logger.error(f"Failed to write {category}: {e}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                # Update tracking
                self._times_downloaded[category] += 1
                if self._times_downloaded[category] > 1:
                    logger.debug(f"Downloaded {category} {self._times_downloaded[category]} times")

                logger.info(f"Downloaded {category} to {target_file_path}")
                self._references_paths_cache[category] = target_file_path

                # Update mtime
                try:
                    self._last_known_mtimes[category] = target_file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                # Populate both dict and string caches with downloaded data
                self._legacy_cache[category] = data
                self._legacy_string_cache[category] = content.decode("utf-8")
                logger.debug(f"Populated legacy cache for {category} after async download")

                return target_file_path

        return None
