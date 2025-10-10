"""HTTP backend for REPLICA mode.

This backend fetches model references from the PRIMARY server's API,
with fallback to GitHub if the PRIMARY is unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, override

import httpx
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_settings
from horde_model_reference.backends.github_backend import GitHubBackend
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class HTTPBackend(ReplicaBackendBase):
    """Backend that fetches model references from PRIMARY API with GitHub fallback.

    This backend is designed for REPLICA mode. It:
    1. Attempts to fetch from PRIMARY server's HTTP API first
    2. Falls back to GitHub if PRIMARY is unavailable
    3. Caches responses locally with TTL
    4. Only works in REPLICA mode

    This provides the best of both worlds:
    - Fast, up-to-date data from PRIMARY when available
    - Resilience via GitHub fallback when PRIMARY is down
    """

    def __init__(
        self,
        *,
        primary_api_url: str,
        github_backend: GitHubBackend,
        cache_ttl_seconds: int = 60,
        timeout_seconds: int = horde_model_reference_settings.primary_api_timeout,
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        enable_github_fallback: bool = horde_model_reference_settings.enable_github_fallback,
    ) -> None:
        """Initialize HTTP backend with GitHub fallback.

        Args:
            primary_api_url: Base URL of PRIMARY server API (e.g., "https://stablehorde.net/api")
            github_backend: GitHub backend to use as fallback
            cache_ttl_seconds: TTL for local cache in seconds
            timeout_seconds: HTTP request timeout in seconds
            retry_max_attempts: Max retry attempts for PRIMARY API
            retry_backoff_seconds: Backoff time between retries
            enable_github_fallback: Whether to fallback to GitHub if PRIMARY fails

        Raises:
            ValueError: If github_backend is not REPLICA mode
        """
        if github_backend.replicate_mode != ReplicateMode.REPLICA:
            raise ValueError("HTTPBackend requires a GitHubBackend in REPLICA mode as fallback")

        super().__init__(mode=ReplicateMode.REPLICA, cache_ttl_seconds=cache_ttl_seconds)

        self._primary_api_url = primary_api_url.rstrip("/")
        self._github_backend = github_backend
        self._timeout_seconds = timeout_seconds
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._enable_github_fallback = enable_github_fallback

        self._cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        self._primary_hits = 0
        self._github_fallbacks = 0

        logger.debug(f"HTTPBackend initialized with PRIMARY at {self._primary_api_url}")

    def _category_api_url(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Get the PRIMARY API URL for a category."""
        return f"{self._primary_api_url}/model_references/v2/{category.value}"

    def _fetch_from_primary(self, category: MODEL_REFERENCE_CATEGORY) -> dict[str, Any] | None:
        """Fetch from PRIMARY API with retries (synchronous)."""
        url = self._category_api_url(category)

        for attempt in range(self._retry_max_attempts):
            if attempt > 0:
                wait_time = self._retry_backoff_seconds * (2 ** (attempt - 1))
                logger.debug(f"Retrying PRIMARY API for {category} in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)

            try:
                response = httpx.get(url, timeout=self._timeout_seconds)

                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    logger.info(f"Fetched {category} from PRIMARY API")
                    self._primary_hits += 1
                    return data

                if response.status_code == 404:
                    logger.debug(f"PRIMARY API returned 404 for {category}")
                    return None

                logger.warning(f"PRIMARY API returned {response.status_code} for {category}")

            except httpx.TimeoutException:
                logger.warning(f"PRIMARY API timeout for {category}")
            except Exception as e:
                logger.warning(f"PRIMARY API error for {category}: {e}")

        logger.warning(f"Failed to fetch {category} from PRIMARY after {self._retry_max_attempts} attempts")
        return None

    async def _fetch_from_primary_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        client: httpx.AsyncClient,
    ) -> dict[str, Any] | None:
        """Fetch from PRIMARY API with retries (asynchronous)."""
        import asyncio

        url = self._category_api_url(category)

        for attempt in range(self._retry_max_attempts):
            if attempt > 0:
                wait_time = self._retry_backoff_seconds * (2 ** (attempt - 1))
                logger.debug(f"Retrying PRIMARY API for {category} in {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)

            try:
                response = await client.get(url, timeout=self._timeout_seconds)

                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    logger.info(f"Fetched {category} from PRIMARY API (async)")
                    self._primary_hits += 1
                    return data

                if response.status_code == 404:
                    logger.debug(f"PRIMARY API returned 404 for {category}")
                    return None

                logger.warning(f"PRIMARY API returned {response.status_code} for {category}")

            except httpx.TimeoutException:
                logger.warning(f"PRIMARY API timeout for {category}")
            except Exception as e:
                logger.warning(f"PRIMARY API error for {category}: {e}")

        logger.warning(f"Failed to fetch {category} from PRIMARY async after {self._retry_max_attempts} attempts")
        return None

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch from PRIMARY API, fallback to GitHub if needed.

        Args:
            category: The category to fetch
            force_refresh: If True, bypass local cache

        Returns:
            Model reference data or None
        """
        with self._lock:

            if not force_refresh and category in self._cache and self.is_cache_valid(category):
                logger.debug(f"Local cache hit for {category}")
                return self._cache[category]

            data = self._fetch_from_primary(category)

            if data is None and self._enable_github_fallback:
                logger.info(f"Falling back to GitHub for {category}")
                self._github_fallbacks += 1
                data = self._github_backend.fetch_category(category, force_refresh=force_refresh)

            if data is not None:
                self._cache[category] = data
                self._mark_category_fresh(category)

            return data

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch all categories from PRIMARY API with GitHub fallback."""
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = self.fetch_category(category, force_refresh=force_refresh)

        return result

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch from PRIMARY API with GitHub fallback.

        Args:
            category: The category to fetch
            httpx_client: Optional httpx AsyncClient to use for requests
            force_refresh: If True, bypass local cache

        Returns:
            Model reference data or None
        """
        if not force_refresh and self.is_cache_valid(category) and category in self._cache:
            logger.debug(f"Local cache hit for {category} (async)")
            return self._cache[category]

        if httpx_client is not None:
            data = await self._fetch_from_primary_async(category, httpx_client)
        else:
            async with httpx.AsyncClient() as client:
                data = await self._fetch_from_primary_async(category, client)

        if data is None and self._enable_github_fallback:
            logger.info(f"Falling back to GitHub for {category} (async)")
            self._github_fallbacks += 1
            data = await self._github_backend.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )

        if data is not None:
            self._cache[category] = data
            self._mark_category_fresh(category)

        return data

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories."""
        import asyncio

        tasks = [
            self.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )
            for category in MODEL_REFERENCE_CATEGORY
        ]

        results = await asyncio.gather(*tasks)

        return dict(zip(MODEL_REFERENCE_CATEGORY, results, strict=False))

    @override
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark category as stale."""
        logger.debug(f"Marking {category} as stale")
        super().mark_stale(category)

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get file path (delegates to GitHub backend)."""
        return self._github_backend.get_category_file_path(category)

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get all file paths (delegates to GitHub backend)."""
        return self._github_backend.get_all_category_file_paths()

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get legacy JSON (delegates to GitHub backend)."""
        return self._github_backend.get_legacy_json(category, redownload=redownload)

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get legacy JSON string (delegates to GitHub backend)."""
        return self._github_backend.get_legacy_json_string(category, redownload=redownload)

    @override
    def get_statistics(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            dict containing:
                - primary_hits: Number of successful PRIMARY API fetches
                - github_fallbacks: Number of times GitHub fallback was used
                - cache_size: Number of categories in local cache
        """
        return {
            "primary_hits": self._primary_hits,
            "github_fallbacks": self._github_fallbacks,
            "cache_size": len(self._cache),
        }
