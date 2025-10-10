"""GitHub-based backend for REPLICA mode.

This backend downloads legacy model reference files from GitHub repositories,
converts them to the new format, and provides them to REPLICA clients as a fallback
when the PRIMARY API is unavailable.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, override

import aiofiles
import httpx
import requests
import ujson
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references
from horde_model_reference.meta_consts import (
    MODEL_REFERENCE_CATEGORY,
    github_image_model_reference_categories,
    github_text_model_reference_categories,
)
from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER_NAME


class GitHubBackend(ReplicaBackendBase):
    """Backend that fetches legacy model references from GitHub and converts them.

    This backend is designed for REPLICA mode only. It:
    1. Downloads legacy JSON files from AI-Horde GitHub repositories
    2. Stores them in a local legacy/ folder
    3. Converts them to the new format using legacy converters
    4. Returns the converted (new format) data
    5. Provides a fallback for REPLICA clients when PRIMARY API is down
    6. PRIMARYs can initialize for the first time if needed from GitHub

    This backend is read-only and enforces REPLICA mode.
    """

    def __init__(
        self,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = horde_model_reference_settings.cache_ttl_seconds,
        retry_max_attempts: int = horde_model_reference_settings.legacy_download_retry_max_attempts,
        retry_backoff_seconds: float = horde_model_reference_settings.legacy_download_retry_backoff_seconds,
        replicate_mode: ReplicateMode = ReplicateMode.REPLICA,
    ) -> None:
        """Initialize the GitHub backend for REPLICA mode.

        Args:
            base_path (str | Path, optional): Base path for storing model reference files.
            cache_ttl_seconds (int, optional): TTL for internal cache in seconds.
            retry_max_attempts (int, optional): Max download retry attempts.
            retry_backoff_seconds (float, optional): Backoff time between retries.
            replicate_mode (ReplicateMode, optional): Must be REPLICA. Defaults to REPLICA.

        Raises:
            ValueError: If replicate_mode is not REPLICA.
        """
        super().__init__(mode=replicate_mode, cache_ttl_seconds=cache_ttl_seconds)

        if self._replicate_mode != ReplicateMode.REPLICA:
            logger.warning(
                "GitHubBackend is designed for REPLICA mode only. For PRIMARY mode, use FileSystemBackend or "
                "RedisBackend. You can ignore this warning if you intend to use GitHubBackend for one-time "
                "initialization in PRIMARY mode."
            )

        self.base_path = Path(base_path)
        self.legacy_path = self.base_path.joinpath(LEGACY_REFERENCE_FOLDER_NAME)

        self.retry_max_attempts = retry_max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

        self._references_paths_cache: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}
        self._converted_cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] | None = None
        self._legacy_cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        self._legacy_string_cache: dict[MODEL_REFERENCE_CATEGORY, str | None] = {}
        self._cache_timestamp: float = 0.0

        self._last_known_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._times_downloaded: dict[MODEL_REFERENCE_CATEGORY, int] = {}

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

                self._load_legacy_json_from_disk(category, file_path)
            else:
                if self._replicate_mode == ReplicateMode.REPLICA:
                    self._references_paths_cache[category] = None
                    self._legacy_cache[category] = None
                    self._legacy_string_cache[category] = None
                else:
                    raise FileNotFoundError(f"Model reference file not found for {category}.")

            self._times_downloaded[category] = 0

    @override
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

            if force_refresh or category in self._stale_categories:
                self._download_and_convert_single(category, override_existing=force_refresh)

            return self._get_cached_category(category)

    @override
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

            if (
                force_refresh
                or any(category in self._stale_categories for category in MODEL_REFERENCE_CATEGORY)
                or any(category not in self._references_paths_cache for category in MODEL_REFERENCE_CATEGORY)
                or any(references is None for references in self._references_paths_cache.values())
            ):
                self._download_and_convert_all(overwrite_existing=force_refresh)

            return self._build_converted_cache()

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a category.

        Args:
            category: The category to fetch.
            httpx_client: Optional httpx async client for downloads.
            force_refresh: If True, force download.

        Returns:
            dict[str, Any] | None: The converted model reference data.
        """
        lock = self.async_lock
        if lock is None:
            raise RuntimeError("Async lock is unavailable for GitHubBackend")

        async with lock:

            if force_refresh or category in self._stale_categories:
                await self._download_legacy_async(
                    category,
                    httpx_client,
                    override_existing=force_refresh,
                )

                if self._is_cache_expired():
                    convert_all_legacy_model_references()

            return self._get_cached_category(category)

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories.

        Args:
            httpx_client: Optional httpx async client.
            force_refresh: If True, force download all.

        Returns:
            dict mapping categories to their data.
        """
        lock = self.async_lock
        if lock is None:
            raise RuntimeError("Async lock is unavailable for GitHubBackend")

        async with lock:

            tasks = []
            for category in MODEL_REFERENCE_CATEGORY:
                override = force_refresh or category in self._stale_categories
                tasks.append(
                    self._download_legacy_async(
                        category,
                        httpx_client,
                        override_existing=override,
                    )
                )

            await asyncio.gather(*tasks)

            if self._is_cache_expired():
                convert_all_legacy_model_references()

            return self._build_converted_cache()

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category needs refresh.

        Args:
            category: The category to check.

        Returns:
            bool: True if needs refresh (stale or mtime changed).
        """
        if super().needs_refresh(category):
            return True

        with self._lock:
            file_path = self._references_paths_cache.get(category)
            if file_path and file_path.exists():
                try:
                    current_mtime = file_path.stat().st_mtime
                    last_known = self._last_known_mtimes.get(category, 0.0)
                    if current_mtime != last_known:
                        logger.debug(f"Legacy file {file_path.name} mtime changed, needs refresh")
                        return True
                except Exception:
                    return True

        return False

    @override
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category as stale, requiring refresh.

        Args:
            category: The category to mark stale.
        """
        logger.debug(f"Marking category {category} as stale")
        super().mark_stale(category)

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's converted data.

        Args:
            category: The category to get path for.

        Returns:
            Path | None: Path to the converted (new format) file, or None if not available.
        """
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories' converted data.

        Returns:
            dict: Mapping of categories to their converted file paths.
        """
        return horde_model_reference_paths.get_all_model_reference_file_paths(base_path=self.base_path)

    def _is_cache_expired(self) -> bool:
        """Check if internal cache has expired."""
        if self._converted_cache is None:
            return True
        ttl = self.cache_ttl_seconds
        if ttl is None:
            return False
        return (time.time() - self._cache_timestamp) > ttl

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

            data: dict[str, Any] = ujson.loads(content)

            self._legacy_cache[category] = data
            self._legacy_string_cache[category] = content.decode("utf-8")
            logger.debug(f"Loaded legacy JSON for category {category!r} from {file_path!r}")
            return data
        except Exception:
            logger.exception(f"Failed to load legacy JSON for category {category!r} from {file_path!r}")
            self._legacy_cache[category] = None
            self._legacy_string_cache[category] = None
            return None

    @override
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

                self._download_legacy(category, override_existing=True)

            if category in self._legacy_cache:
                logger.debug(f"Returning cached legacy JSON for category {category!r}")
                return self._legacy_cache[category]

            file_path = self._references_paths_cache.get(category)
            if not file_path:
                self._legacy_cache[category] = None
                return None

            return self._load_legacy_json_from_disk(category, file_path)

    @override
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

                self._download_legacy(category, override_existing=True)

            if category in self._legacy_string_cache:
                cached_value = self._legacy_string_cache[category]
                if cached_value is not None:
                    logger.debug(f"Returning cached legacy JSON string for category {category!r}")
                    return cached_value

            file_path = self._references_paths_cache.get(category)
            if not file_path:
                self._legacy_string_cache[category] = None
                return None

            self._load_legacy_json_from_disk(category, file_path)
            return self._legacy_string_cache.get(category)

    def _build_converted_cache(self) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Build cache from converted (new format) files on disk."""
        logger.debug("Building converted references cache from disk")

        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:

            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if file_path and file_path.exists():
                try:
                    with open(file_path) as f:
                        result[category] = ujson.load(f)
                    self._mark_category_fresh(category)
                except Exception as e:
                    logger.warning(f"Error reading converted file {file_path}: {e}")
                    result[category] = None
                    self._invalidate_category_timestamp(category)
            else:
                result[category] = None
                self._invalidate_category_timestamp(category)

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

        if self._is_cache_expired():
            convert_all_legacy_model_references()

        self._mark_category_fresh(category)

    def _download_and_convert_all(self, overwrite_existing: bool = False) -> None:
        """Download all legacy files and convert them."""
        for category in MODEL_REFERENCE_CATEGORY:
            override = overwrite_existing or category in self._stale_categories
            self._download_legacy(category, override_existing=override)

        if self._is_cache_expired():
            convert_all_legacy_model_references()

        for category in MODEL_REFERENCE_CATEGORY:
            self._mark_category_fresh(category)

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

            target_url: str | None = None
            if category in github_image_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_image_model_github_urls[category]
            elif category in github_text_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_text_model_github_urls[category]
            else:
                logger.debug(f"No known GitHub URL for {category}")
                return None

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

                    try:
                        import ujson

                        data = ujson.loads(response.content)
                    except ImportError:
                        data = ujson.loads(response.content)
                except ujson.JSONDecodeError:
                    logger.error(f"Failed to parse {category} as JSON")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_file_path, "wb") as f:
                    f.write(response.content)

                self._times_downloaded[category] += 1
                if self._times_downloaded[category] > 1:
                    logger.debug(f"Downloaded {category} {self._times_downloaded[category]} times")

                logger.info(f"Downloaded {category} to {target_file_path}")
                self._references_paths_cache[category] = target_file_path

                try:
                    self._last_known_mtimes[category] = target_file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                self._legacy_cache[category] = data
                self._legacy_string_cache[category] = response.content.decode("utf-8")
                logger.debug(f"Populated legacy cache for {category} after download")

                return target_file_path

            return None

    async def _download_legacy_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        httpx_client: httpx.AsyncClient | None = None,
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

        target_url: str | None = None
        if category in github_image_model_reference_categories:
            target_url = horde_model_reference_paths.legacy_image_model_github_urls[category]
        elif category in github_text_model_reference_categories:
            target_url = horde_model_reference_paths.legacy_text_model_github_urls[category]
        else:
            logger.debug(f"No known GitHub URL for {category}")
            return None

        for attempt in range(1, self.retry_max_attempts + 1):
            if attempt > 1:
                logger.debug(
                    f"Retrying download of {category} in {self.retry_backoff_seconds}s "
                    f"(attempt {attempt}/{self.retry_max_attempts})"
                )
                await asyncio.sleep(self.retry_backoff_seconds)

            if httpx_client is not None:
                response = await httpx_client.get(target_url)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(target_url)

            if response.status_code != 200:
                logger.error(f"Failed to download {category}: HTTP {response.status_code}")
                if attempt == self.retry_max_attempts:
                    return None
                continue

            content = response.content

            try:
                data = ujson.loads(content)
            except ujson.JSONDecodeError:
                logger.error(f"Failed to parse {category} as JSON")
                if attempt == self.retry_max_attempts:
                    return None

                target_file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    async with aiofiles.open(target_file_path, "wb") as f:
                        await f.write(content)
                except Exception as e:
                    logger.error(f"Failed to write {category}: {e}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                self._times_downloaded[category] += 1
                if self._times_downloaded[category] > 1:
                    logger.debug(f"Downloaded {category} {self._times_downloaded[category]} times")

                logger.info(f"Downloaded {category} to {target_file_path}")
                self._references_paths_cache[category] = target_file_path

                try:
                    self._last_known_mtimes[category] = target_file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                self._legacy_cache[category] = data
                self._legacy_string_cache[category] = content.decode("utf-8")
                logger.debug(f"Populated legacy cache for {category} after async download")

                return target_file_path

        return None
