"""GitHub-based backend for REPLICA mode.

This backend downloads legacy model reference files from GitHub repositories,
converts them to the new format, and provides them to REPLICA clients as a fallback
when the PRIMARY API is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, cast

import aiofiles
import httpx
import requests
import ujson
from loguru import logger
from typing_extensions import override

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.legacy.convert_all_legacy_dbs import (
    convert_all_legacy_model_references,
    convert_legacy_database_by_category,
)
from horde_model_reference.legacy.text_csv_utils import parse_legacy_text_csv
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
            base_path: Base path for storing model reference files.
            cache_ttl_seconds: TTL for internal cache in seconds.
            retry_max_attempts: Max download retry attempts.
            retry_backoff_seconds: Backoff time between retries.
            replicate_mode: Must be REPLICA. Defaults to REPLICA.

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
        self._times_downloaded: dict[MODEL_REFERENCE_CATEGORY, int] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if file_path.exists():
                self._references_paths_cache[category] = file_path
                self._load_legacy_json_from_disk(category, file_path)
            else:
                if self._replicate_mode == ReplicateMode.REPLICA or (
                    self._replicate_mode == ReplicateMode.PRIMARY
                    and horde_model_reference_settings.github_seed_enabled
                ):
                    self._references_paths_cache[category] = None
                else:
                    raise FileNotFoundError(f"Model reference file not found for {category}.")

            self._times_downloaded[category] = 0

    @override
    def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return the converted (v2) file path for mtime validation.

        Args:
            category: The category to get the file path for.

        Returns:
            Path | None: Path to converted file for mtime validation.
        """
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    @override
    def _get_legacy_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return the legacy file path for mtime validation.

        Args:
            category: The category to get the legacy file path for.

        Returns:
            Path | None: Path to legacy file for mtime validation.
        """
        return self._references_paths_cache.get(category)

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
            # Use helper to determine if we need to fetch
            if force_refresh or self.should_fetch_data(category):
                self._download_and_convert_single(category, overwrite_existing=force_refresh)
                return self._load_converted_from_disk(category)

            # Return cached data
            return self._get_from_cache(category)

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
            if force_refresh:
                self._download_and_convert_all(overwrite_existing=True)

            result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
            for category in MODEL_REFERENCE_CATEGORY:
                # Use helper to determine if we need to fetch
                if force_refresh or self.should_fetch_data(category):
                    self._download_legacy(category, overwrite_existing=force_refresh)
                    convert_legacy_database_by_category(category, self.base_path, self.base_path)
                    result[category] = self._load_converted_from_disk(category)
                else:
                    # Return cached data
                    result[category] = self._get_from_cache(category)

            return result

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
            # Use helper to determine if we need to fetch
            if force_refresh or self.should_fetch_data(category):
                await self._download_legacy_async(
                    category,
                    httpx_client,
                    overwrite_existing=force_refresh,
                )
                convert_legacy_database_by_category(category, self.base_path, self.base_path)
                return self._load_converted_from_disk(category)

            # Return cached data
            return self._get_from_cache(category)

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
            # Download all that need refresh
            tasks = []
            categories_to_download = []
            for category in MODEL_REFERENCE_CATEGORY:
                # Use helper to determine if we need to fetch
                if force_refresh or self.should_fetch_data(category):
                    categories_to_download.append(category)
                    tasks.append(
                        self._download_legacy_async(
                            category,
                            httpx_client,
                            overwrite_existing=force_refresh,
                        )
                    )

            if tasks:
                await asyncio.gather(*tasks)
                convert_all_legacy_model_references()

            # Collect results from cache or disk
            result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
            for category in MODEL_REFERENCE_CATEGORY:
                if category in categories_to_download:
                    result[category] = self._load_converted_from_disk(category)
                else:
                    result[category] = self._get_from_cache(category)

            return result

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category needs refresh.

        Base class handles all validation including mtime checks via hooks.

        Args:
            category: The category to check.

        Returns:
            bool: True if needs refresh (stale or mtime changed).
        """
        return super().needs_refresh(category)

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
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: Mapping of categories to their converted file paths.
        """
        return horde_model_reference_paths.get_all_model_reference_file_paths(base_path=self.base_path)

    def _load_legacy_json_from_disk(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        file_path: Path,
    ) -> dict[str, Any] | None:
        """Load legacy data from disk and populate cache via base class.

        For text_generation category, loads CSV format (models.csv).
        For other categories, loads JSON format.

        Args:
            category: The category to load.
            file_path: Path to the legacy file.

        Returns:
            dict[str, Any] | None: The loaded data, or None on error.
        """
        if not file_path.exists():
            logger.debug(f"Legacy file {file_path} does not exist")
            self._store_legacy_in_cache(category, None, None)
            return None

        try:
            # Handle CSV format for text_generation
            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                data = self._read_legacy_csv_to_dict(file_path)
                # For CSV, we store the dict and generate a JSON string for cache
                content_str = ujson.dumps(data, escape_forward_slashes=False, indent=4)
                self._store_legacy_in_cache(category, data, content_str)
                logger.debug(f"Loaded legacy CSV for category {category!r} from {file_path!r}")
                return data

            # Handle JSON format for other categories
            with open(file_path, "rb") as f:
                content = f.read()

            # Handle empty files (e.g., for categories with no GitHub URL)
            if not content or content.strip() == b"":
                logger.debug(f"Legacy file {file_path} is empty, treating as empty dict")
                self._store_legacy_in_cache(category, {}, "{}")
                return {}

            data = ujson.loads(content)
            content_str = content.decode("utf-8")

            self._store_legacy_in_cache(category, data, content_str)
            logger.debug(f"Loaded legacy JSON for category {category!r} from {file_path!r}")
            return data
        except Exception:
            logger.exception(f"Failed to load legacy data for category {category!r} from {file_path!r}")
            self._store_legacy_in_cache(category, None, None)
            return None

    def _read_legacy_csv_to_dict(self, file_path: Path) -> dict[str, Any]:
        """Read legacy CSV file (models.csv format) and convert to dict format.

        This reads the legacy models.csv format from GitHub and converts it to the
        grouped dict format with backend prefix duplicates (3 entries per CSV row).
        Follows the same logic as scripts/legacy_text/convert.py.

        Each CSV row generates 3 entries:
        1. Base name (e.g., ReadyArt/Broken-Tutu-24B)
        2. Aphrodite prefixed (e.g., aphrodite/ReadyArt/Broken-Tutu-24B)
        3. KoboldCPP prefixed (e.g., koboldcpp/Broken-Tutu-24B) - uses model_name only

        Args:
            file_path: Path to the legacy CSV file.

        Returns:
            dict[str, Any]: Model data with 3 entries per CSV row (base + 2 backend prefixes).

        Raises:
            Exception: If CSV parsing fails.
        """
        data: dict[str, Any] = {}
        parsed_rows, parse_issues = parse_legacy_text_csv(file_path)
        for issue in parse_issues:
            logger.warning(f"Legacy CSV issue for {issue.row_identifier}: {issue.message}")

        for csv_row in parsed_rows:
            name = csv_row.name
            model_name = name.split("/")[1] if "/" in name else name
            tags = set(csv_row.tags)
            if csv_row.style:
                tags.add(csv_row.style)
            tags.add(f"{round(csv_row.parameters_bn, 0):.0f}B")

            settings_dict = dict(csv_row.settings) if csv_row.settings is not None else {}

            display_name = csv_row.display_name
            if not display_name:
                display_name = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

            record: dict[str, Any] = {
                "name": name,
                "model_name": model_name,
                "parameters": csv_row.parameters,
                "description": csv_row.description,
                "version": csv_row.version,
                "style": csv_row.style,
                "nsfw": csv_row.nsfw,
                "baseline": csv_row.baseline,
                "url": csv_row.url,
                "tags": sorted(tags),
                "settings": settings_dict,
                "display_name": display_name,
            }

            record = {k: v for k, v in record.items() if v or v is False}

            # Generate 3 entries per CSV row following convert.py logic
            # 1. Base entry
            data[name] = record.copy()
            data[name]["name"] = name

            # 2. Aphrodite prefixed entry (uses full name)
            aphrodite_key = f"aphrodite/{name}"
            data[aphrodite_key] = record.copy()
            data[aphrodite_key]["name"] = aphrodite_key

            # 3. KoboldCPP prefixed entry (uses model_name only, not full name)
            koboldcpp_key = f"koboldcpp/{model_name}"
            data[koboldcpp_key] = record.copy()
            data[koboldcpp_key]["name"] = koboldcpp_key

        logger.debug(f"Read {len(data)} models from legacy CSV (includes backend prefix duplicates)")
        return data

    def _read_csv_to_dict(self, file_path: Path) -> dict[str, Any]:
        """Read v2 CSV file and convert to dict format (grouped by base name, no backend prefixes).

        This reads the v2 CSV format and returns a dict with one entry per base model.
        No backend prefix duplicates are generated here - that only happens during GitHub sync.

        Args:
            file_path: Path to the CSV file.

        Returns:
            dict[str, Any]: Model data with one entry per base model.

        Raises:
            Exception: If CSV parsing fails.
        """
        data: dict[str, Any] = {}
        parsed_rows, parse_issues = parse_legacy_text_csv(file_path)
        for issue in parse_issues:
            logger.warning(f"CSV issue for {issue.row_identifier}: {issue.message}")

        for csv_row in parsed_rows:
            name = csv_row.name
            model_name = name.split("/")[1] if "/" in name else name
            tags = set(csv_row.tags)
            if csv_row.style:
                tags.add(csv_row.style)
            tags.add(f"{round(csv_row.parameters_bn, 0):.0f}B")

            settings_dict = dict(csv_row.settings) if csv_row.settings is not None else {}

            display_name = csv_row.display_name
            if not display_name:
                display_name = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

            record: dict[str, Any] = {
                "name": name,
                "model_name": model_name,
                "parameters": csv_row.parameters,
                "description": csv_row.description,
                "version": csv_row.version,
                "style": csv_row.style,
                "nsfw": csv_row.nsfw,
                "baseline": csv_row.baseline,
                "url": csv_row.url,
                "tags": sorted(tags),
                "settings": settings_dict,
                "display_name": display_name,
            }

            record = {k: v for k, v in record.items() if v or v is False}
            data[name] = record

        logger.debug(f"Read {len(data)} models from CSV (grouped, no backend prefixes)")
        return data

    def _load_converted_from_disk(
        self,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> dict[str, Any] | None:
        """Load converted (v2 format) data from disk and cache via base class.

        All v2 format files (including text_generation.json) are in JSON format.
        CSV format is only used for legacy files (legacy/models.csv).

        Args:
            category: The category to load.

        Returns:
            dict[str, Any] | None: The loaded data, or None on error.
        """
        file_path = horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

        if not file_path.exists():
            logger.debug(f"Converted file {file_path} does not exist")
            self._store_in_cache(category, None)
            return None

        try:
            # All v2 files are JSON format (including text_generation.json)
            with open(file_path) as f:
                data = cast(dict[str, Any], ujson.load(f))

            self._store_in_cache(category, data)
            logger.debug(f"Loaded converted JSON for category {category!r} from {file_path!r}")
            return data
        except Exception:
            logger.exception(f"Failed to load converted data for category {category!r} from {file_path!r}")
            self._store_in_cache(category, None)
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
            # Use helper to determine if we need to fetch
            if redownload or self.should_fetch_data(category):
                self._download_legacy(category, overwrite_existing=redownload)

            # Try cache first
            legacy_dict, _ = self._get_legacy_from_cache(category)
            if legacy_dict is not None:
                return legacy_dict

            # Load from disk and cache if available
            file_path = self._references_paths_cache.get(category)
            if file_path:
                return self._load_legacy_json_from_disk(category, file_path)

            return None

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
            # Use helper to determine if we need to fetch
            if redownload or self.should_fetch_data(category):
                self._download_legacy(category, overwrite_existing=redownload)

            # Try cache first
            _, legacy_string = self._get_legacy_from_cache(category)
            if legacy_string is not None:
                return legacy_string

            # Load from disk and cache if available
            file_path = self._references_paths_cache.get(category)
            if file_path:
                self._load_legacy_json_from_disk(category, file_path)
                _, legacy_string = self._get_legacy_from_cache(category)
                return legacy_string

            return None

    def _download_and_convert_single(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> None:
        """Download a single legacy file and convert it.

        Args:
            category: The category to download and convert.
            overwrite_existing: If True, overwrite existing files.
        """
        self._download_legacy(category, overwrite_existing=overwrite_existing)

        convert_legacy_database_by_category(category, self.base_path, self.base_path)

    def _download_and_convert_all(self, overwrite_existing: bool = False) -> None:
        """Download all legacy files and convert them."""
        for category in MODEL_REFERENCE_CATEGORY:
            self._download_legacy(category, overwrite_existing=overwrite_existing)

        convert_all_legacy_model_references(self.base_path, self.base_path)

    def _download_allowed(self) -> bool:
        """Return `True` if downloading is allowed based on replicate mode and settings."""
        if self._replicate_mode == ReplicateMode.PRIMARY and horde_model_reference_settings.github_seed_enabled:
            return True
        return self._replicate_mode == ReplicateMode.REPLICA

    def _download_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> Path | None:
        """Download a single legacy file from GitHub (synchronous).

        Args:
            category: The category to download.
            overwrite_existing: If True, overwrite existing file.

        Returns:
            Path | None: Path to the downloaded file, or None on failure.
        """
        if not self._download_allowed():
            return self._references_paths_cache.get(category)

        target_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

        needs_refresh = self.needs_refresh(category)
        if needs_refresh:
            logger.debug(f"Category {category} needs refresh, proceeding to download")
            overwrite_existing = True

        with self._lock:
            if target_file_path.exists() and not overwrite_existing:
                logger.debug(f"Legacy file {target_file_path} already exists, skipping download")
                return target_file_path

            target_url: str | None = None
            if category in github_image_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_image_model_github_urls[category]
            elif category in github_text_model_reference_categories:
                target_url = horde_model_reference_paths.legacy_text_model_github_urls[category]
            else:
                logger.debug(f"No known GitHub URL for {category}, creating empty file")
                target_file_path.parent.mkdir(parents=True, exist_ok=True)
                target_file_path.touch(exist_ok=True)
                return target_file_path

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

                # Handle CSV format for text_generation category
                if category == MODEL_REFERENCE_CATEGORY.text_generation:
                    # Save CSV directly to disk
                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(target_file_path, "wb") as f:
                        f.write(response.content)

                    # Parse CSV to dict for caching
                    try:
                        data = self._read_legacy_csv_to_dict(target_file_path)
                        raw_json_str = ujson.dumps(data, escape_forward_slashes=False, indent=4)
                    except Exception as e:
                        logger.error(f"Failed to parse {category} CSV: {e}")
                        if attempt == self.retry_max_attempts:
                            return None
                        continue
                else:
                    # Handle JSON format for other categories
                    try:
                        data = ujson.loads(response.content)
                    except ujson.JSONDecodeError:
                        logger.error(f"Failed to parse {category} as JSON")
                        if attempt == self.retry_max_attempts:
                            return None
                        continue

                    target_file_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_json_str = response.content.decode("utf-8")
                    with open(target_file_path, "wb") as f:
                        f.write(response.content)

                self._times_downloaded[category] += 1
                if self._times_downloaded[category] > 1:
                    logger.debug(f"Downloaded {category} {self._times_downloaded[category]} times")

                logger.info(f"Downloaded {category} to {target_file_path}")
                self._references_paths_cache[category] = target_file_path

                # Store in base class cache
                self._store_legacy_in_cache(category, data, raw_json_str)
                logger.debug(f"Populated legacy cache for {category} after download")

                return target_file_path

            return None

    async def _download_legacy_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        httpx_client: httpx.AsyncClient | None = None,
        overwrite_existing: bool = False,
    ) -> Path | None:
        """Download a single legacy file from GitHub (asynchronous).

        Args:
            category: The category to download.
            httpx_client: Optional httpx async client for downloads.
            overwrite_existing: If True, overwrite existing file.

        Returns:
            Path | None: Path to the downloaded file, or None on failure.
        """
        if not self._download_allowed():
            logger.debug(f"Replicate mode is not REPLICA, skipping download for {category}")
            return self._references_paths_cache.get(category)

        if httpx_client is None:
            logger.debug("No httpx_client provided, will create a new one for this download")

        target_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

        needs_refresh = self.needs_refresh(category)
        if needs_refresh:
            logger.debug(f"Category {category} needs refresh, proceeding to download")
            overwrite_existing = True

        if target_file_path.exists() and not overwrite_existing:
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
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Handle CSV format for text_generation category
            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                # Save CSV directly to disk
                try:
                    async with aiofiles.open(target_file_path, "wb") as f:
                        await f.write(content)
                except Exception as e:
                    logger.error(f"Failed to write {category} CSV: {e}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

                # Parse CSV to dict for caching
                try:
                    data = self._read_legacy_csv_to_dict(target_file_path)
                    content_str = ujson.dumps(data, escape_forward_slashes=False, indent=4)
                except Exception as e:
                    logger.error(f"Failed to parse {category} CSV: {e}")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue
            else:
                # Handle JSON format for other categories
                try:
                    data = ujson.loads(content)
                    content_str = content.decode("utf-8")
                except ujson.JSONDecodeError:
                    logger.error(f"Failed to parse {category} as JSON")
                    if attempt == self.retry_max_attempts:
                        return None
                    continue

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

            # Store in base class cache
            self._store_legacy_in_cache(category, data, content_str)
            logger.debug(f"Populated legacy cache for {category} after async download")

            return target_file_path

        return None
