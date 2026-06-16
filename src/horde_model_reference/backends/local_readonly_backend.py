"""Offline, read-only backend that serves model references straight from local disk.

This backend exists for the worker -> multi-subprocess paradigm. The parent process owns *all*
reference downloading (via a REPLICA backend) and writes the converted v2 JSON to ``base_path``;
every subprocess then constructs a :class:`LocalReadOnlyBackend` that reads those files and is
structurally incapable of reaching the network. This removes the per-subprocess
download/convert pipeline that otherwise runs once in every spawned interpreter.

It is a REPLICA-mode backend with ``supports_writes() == False`` so the manager skips the
write-path machinery (pending queue, group stores, audit). Staleness is purely mtime/TTL based
(inherited from :class:`ReplicaBackendBase`), so a parent refresh that rewrites a category file is
picked up on the next reload without any network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

import httpx
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class LocalReadOnlyBackend(ReplicaBackendBase):
    """Read converted v2 references from ``base_path`` on disk; never download or convert."""

    def __init__(
        self,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = 60,
    ) -> None:
        """Initialize the offline read-only backend.

        Args:
            base_path: Base path the parent process wrote converted reference files to.
            cache_ttl_seconds: TTL for the in-memory cache. mtime changes also invalidate.

        """
        super().__init__(mode=ReplicateMode.REPLICA, cache_ttl_seconds=cache_ttl_seconds)
        self.base_path = Path(base_path)
        logger.debug(f"LocalReadOnlyBackend initialized with base_path={self.base_path}")

    @override
    def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        return horde_model_reference_paths.get_model_reference_file_path(category, base_path=self.base_path)

    @override
    def _get_legacy_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        return horde_model_reference_paths.get_legacy_model_reference_file_path(category, base_path=self.base_path)

    def _read_json_from_disk(self, file_path: Path | None) -> dict[str, Any] | None:
        if not file_path or not file_path.exists():
            return None
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"LocalReadOnlyBackend failed to read {file_path}: {e}")
            return None

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        with self._lock:
            return self._fetch_with_cache(
                category,
                lambda: self._read_json_from_disk(self._get_file_path_for_validation(category)),
                force_refresh=force_refresh,
            )

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {
            category: self.fetch_category(category, force_refresh=force_refresh)
            for category in MODEL_REFERENCE_CATEGORY
        }

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        # Reads are local disk; there is no benefit to async file I/O for these small JSON files.
        return self.fetch_category(category, force_refresh=force_refresh)

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        return {
            category: await self.fetch_category_async(category, force_refresh=force_refresh)
            for category in MODEL_REFERENCE_CATEGORY
        }

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        return self._get_file_path_for_validation(category)

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        return {category: self.get_category_file_path(category) for category in MODEL_REFERENCE_CATEGORY}

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        # Offline: redownload is a no-op; serve whatever the parent left on disk.
        return self._read_json_from_disk(self._get_legacy_file_path_for_validation(category))

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        file_path = self._get_legacy_file_path_for_validation(category)
        if not file_path or not file_path.exists():
            return None
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error(f"LocalReadOnlyBackend failed to read legacy {file_path}: {e}")
            return None
