"""FileSystem backend for PRIMARY mode.

This backend reads and writes model reference JSON files directly on the local filesystem.
It is the source of truth for PRIMARY mode instances and never interacts with GitHub.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from threading import RLock
from typing import Any, override

import httpx
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class FileSystemBackend(ModelReferenceBackend):
    """Backend that reads/writes model references directly on the local filesystem."""

    def __init__(
        self,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = 60,
        replicate_mode: ReplicateMode = ReplicateMode.PRIMARY,
    ) -> None:
        """Initialize the FileSystem backend.

        Args:
            base_path (str | Path, optional): Base path for model reference files.
            cache_ttl_seconds (int, optional): TTL for internal cache in seconds.
            replicate_mode (ReplicateMode, optional): Must be PRIMARY.

        Raises:
            ValueError: If replicate_mode is not PRIMARY.
        """
        if replicate_mode != ReplicateMode.PRIMARY:
            raise ValueError(
                "FileSystemBackend can only be used in PRIMARY mode. "
                "For REPLICA mode, use GitHubBackend or HTTPBackend."
            )
        super().__init__(mode=replicate_mode)

        self.base_path = Path(base_path)
        self._cache_ttl_seconds = cache_ttl_seconds

        self._lock = RLock()

        self._cache: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        self._cache_timestamp: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._last_known_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._stale_categories: set[MODEL_REFERENCE_CATEGORY] = set()

        for category in MODEL_REFERENCE_CATEGORY:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if file_path and file_path.exists():
                try:
                    self._last_known_mtimes[category] = file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

        logger.debug(f"FileSystemBackend initialized with base_path={self.base_path}")

    def _is_cache_expired(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if cache for a category has expired."""
        if category not in self._cache_timestamp:
            return True
        return (time.time() - self._cache_timestamp[category]) > self._cache_ttl_seconds

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch model reference data for a specific category from filesystem.

        Args:
            category: The category to fetch.
            force_refresh: If True, bypass cache and read from disk.

        Returns:
            dict[str, Any] | None: The model reference data, or None if file doesn't exist.
        """
        with self._lock:
            if (
                not force_refresh
                and not self._is_cache_expired(category)
                and category not in self._stale_categories
                and category in self._cache
            ):
                logger.debug(f"Cache hit for {category}")
                return self._cache[category]

            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path or not file_path.exists():
                logger.debug(f"File not found for {category}: {file_path}")
                self._cache[category] = None
                self._cache_timestamp[category] = time.time()
                return None

            try:
                with open(file_path, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)

                self._cache[category] = data
                self._cache_timestamp[category] = time.time()
                self._stale_categories.discard(category)

                try:
                    self._last_known_mtimes[category] = file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

                logger.debug(f"Loaded {category} from {file_path}")
                return data

            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return None

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch model reference data for all categories.

        Args:
            force_refresh: If True, bypass cache for all categories.

        Returns:
            dict mapping categories to their model reference data.
        """
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = self.fetch_category(category, force_refresh=force_refresh)

        return result

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a category.

        Note: File I/O is still synchronous as async file I/O doesn't provide
        significant benefits for small JSON files.

        Args:
            category: The category to fetch.
            httpx_client: Optional httpx async client for downloads.
            force_refresh: If True, bypass cache.

        Returns:
            dict[str, Any] | None: The model reference data.
        """
        return self.fetch_category(category, force_refresh=force_refresh)

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories (delegates to sync method)."""
        return self.fetch_all_categories(force_refresh=force_refresh)

    @override
    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category needs refresh.

        Args:
            category: The category to check.

        Returns:
            bool: True if needs refresh (stale or mtime changed).
        """
        with self._lock:
            if category in self._stale_categories:
                return True

            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if file_path and file_path.exists():
                try:
                    current_mtime = file_path.stat().st_mtime
                    last_known = self._last_known_mtimes.get(category, 0.0)
                    if current_mtime != last_known:
                        logger.debug(f"File {file_path.name} mtime changed, needs refresh")
                        return True
                except Exception:
                    pass

            return False

    @override
    def mark_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category as stale, requiring refresh on next access.

        Args:
            category: The category to mark stale.
        """
        with self._lock:
            logger.debug(f"Marking category {category} as stale")
            self._stale_categories.add(category)

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's data.

        Args:
            category: The category to get path for.

        Returns:
            Path | None: Path to the JSON file, or None if not configured.
        """
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories.

        Returns:
            dict: Mapping of categories to their file paths.
        """
        return horde_model_reference_paths.get_all_model_reference_file_paths(base_path=self.base_path)

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get legacy format JSON from legacy/ folder.

        Args:
            category: Category to retrieve.
            redownload: Ignored (files are already local).

        Returns:
            dict[str, Any] | None: The legacy format JSON data, or None if file doesn't exist.
        """
        legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

        if not legacy_file_path.exists():
            logger.debug(f"Legacy file not found for {category}: {legacy_file_path}")
            return None

        try:
            with open(legacy_file_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            logger.debug(f"Loaded legacy JSON for {category} from {legacy_file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to read legacy file {legacy_file_path}: {e}")
            return None

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get legacy format JSON string from legacy/ folder.

        Args:
            category: Category to retrieve.
            redownload: Ignored (files are already local).

        Returns:
            str | None: The legacy format JSON string, or None if file doesn't exist.
        """
        legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

        if not legacy_file_path.exists():
            logger.debug(f"Legacy file not found for {category}: {legacy_file_path}")
            return None

        try:
            with open(legacy_file_path, encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Loaded legacy JSON string for {category} from {legacy_file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read legacy file {legacy_file_path}: {e}")
            return None

    @override
    def supports_writes(self) -> bool:
        """Check if backend supports writes (always True for PRIMARY filesystem).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference.

        This method modifies the JSON file on disk atomically.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data as a dictionary.

        Raises:
            FileNotFoundError: If the category file path is not configured.
        """
        with self._lock:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path:
                raise FileNotFoundError(f"No file path configured for category {category}")

            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        existing_data: dict[str, Any] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    raise
            else:
                existing_data = {}
                file_path.parent.mkdir(parents=True, exist_ok=True)

            existing_data[model_name] = record_dict

            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if file_path.exists():
                    backup_path = file_path.with_suffix(".bak")
                    file_path.replace(backup_path)
                    temp_path.replace(file_path)
                    with contextlib.suppress(Exception):
                        backup_path.unlink()
                else:
                    temp_path.replace(file_path)

                logger.info(f"Updated model {model_name} in category {category} at {file_path}")

                self._stale_categories.add(category)

                try:
                    self._last_known_mtimes[category] = file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to update model {model_name} in {category}: {e}")
                raise

    @override
    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference.

        This method removes the model from the JSON file on disk atomically.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            FileNotFoundError: If the category file doesn't exist.
            KeyError: If the model doesn't exist in the category.
        """
        with self._lock:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path or not file_path.exists():
                raise FileNotFoundError(f"Category file not found: {file_path}")

            try:
                with open(file_path, encoding="utf-8") as f:
                    existing_data: dict[str, Any] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                raise

            if model_name not in existing_data:
                raise KeyError(f"Model {model_name} not found in category {category}")

            del existing_data[model_name]

            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                backup_path = file_path.with_suffix(".bak")
                file_path.replace(backup_path)
                temp_path.replace(file_path)

                with contextlib.suppress(Exception):
                    backup_path.unlink()

                logger.info(f"Deleted model {model_name} from category {category} at {file_path}")

                self._stale_categories.add(category)

                try:
                    self._last_known_mtimes[category] = file_path.stat().st_mtime
                except Exception:
                    self._last_known_mtimes[category] = 0.0

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to delete model {model_name} from {category}: {e}")
                raise

    @override
    def supports_legacy_writes(self) -> bool:
        """Check if backend supports legacy format writes.

        Returns True only when canonical_format='legacy' in settings.

        Returns:
            bool: True if legacy writes are supported.
        """
        from horde_model_reference import horde_model_reference_settings

        return horde_model_reference_settings.canonical_format == "legacy"

    @override
    def update_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference in legacy format.

        This method modifies the legacy format JSON file on disk atomically.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data in legacy format as a dictionary.

        Raises:
            FileNotFoundError: If the legacy category file path is not configured.
            RuntimeError: If canonical_format is not set to 'legacy'.
        """
        from horde_model_reference import horde_model_reference_settings

        if not self.supports_legacy_writes():
            raise RuntimeError(
                "Legacy writes are only supported when canonical_format='legacy'. "
                f"Current setting: canonical_format='{horde_model_reference_settings.canonical_format}'"
            )

        with self._lock:
            legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not legacy_file_path:
                raise FileNotFoundError(f"No legacy file path configured for category {category}")

            if legacy_file_path.exists():
                try:
                    with open(legacy_file_path, encoding="utf-8") as f:
                        existing_data: dict[str, Any] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read {legacy_file_path}: {e}")
                    raise
            else:
                existing_data = {}
                legacy_file_path.parent.mkdir(parents=True, exist_ok=True)

            existing_data[model_name] = record_dict

            temp_path = legacy_file_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if legacy_file_path.exists():
                    backup_path = legacy_file_path.with_suffix(".bak")
                    legacy_file_path.replace(backup_path)
                    temp_path.replace(legacy_file_path)
                    with contextlib.suppress(Exception):
                        backup_path.unlink()
                else:
                    temp_path.replace(legacy_file_path)

                logger.info(f"Updated legacy model {model_name} in category {category} at {legacy_file_path}")

                self._stale_categories.add(category)

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to update legacy model {model_name} in {category}: {e}")
                raise

    @override
    def delete_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference from legacy format files.

        This method removes the model from the legacy format JSON file on disk atomically.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            FileNotFoundError: If the legacy category file doesn't exist.
            KeyError: If the model doesn't exist in the category.
            RuntimeError: If canonical_format is not set to 'legacy'.
        """
        from horde_model_reference import horde_model_reference_settings

        if not self.supports_legacy_writes():
            raise RuntimeError(
                "Legacy writes are only supported when canonical_format='legacy'. "
                f"Current setting: canonical_format='{horde_model_reference_settings.canonical_format}'"
            )

        with self._lock:
            legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not legacy_file_path or not legacy_file_path.exists():
                raise FileNotFoundError(f"Legacy category file not found: {legacy_file_path}")

            try:
                with open(legacy_file_path, encoding="utf-8") as f:
                    existing_data: dict[str, Any] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {legacy_file_path}: {e}")
                raise

            if model_name not in existing_data:
                raise KeyError(f"Model {model_name} not found in legacy category {category}")

            del existing_data[model_name]

            temp_path = legacy_file_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                backup_path = legacy_file_path.with_suffix(".bak")
                legacy_file_path.replace(backup_path)
                temp_path.replace(legacy_file_path)

                with contextlib.suppress(Exception):
                    backup_path.unlink()

                logger.info(f"Deleted legacy model {model_name} from category {category} at {legacy_file_path}")

                self._stale_categories.add(category)

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to delete legacy model {model_name} from {category}: {e}")
                raise
