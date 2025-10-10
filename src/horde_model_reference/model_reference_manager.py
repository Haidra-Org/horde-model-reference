from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.backends.github_backend import GitHubBackend
from horde_model_reference.backends.http_backend import HTTPBackend
from horde_model_reference.backends.redis_backend import RedisBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import MODEL_RECORD_TYPE_LOOKUP, GenericModelRecord


class ModelReferenceManager:
    """Singleton class for downloading and reading model reference files.

    This class is responsible for managing the lifecycle of model reference files,
    including downloading, caching, and providing access to the model references.

    Uses a pluggable backend architecture to support different data sources (GitHub, database, etc.).

    Settings on initialization (base_path, backend, lazy_mode, etc) are only set on the first instantiation
    (e.g. `ModelReferenceManager(base_path=...)`). Subsequent instantiations will return the same instance.

    Retrieve all model references with `get_all_model_references_unsafe()`.
    """

    backend: ModelReferenceBackend
    """The backend provider for model reference data."""
    _cached_file_json: dict[MODEL_REFERENCE_CATEGORY, dict[Any, Any] | None]
    _cached_system_file_modified_times: dict[MODEL_REFERENCE_CATEGORY, float]

    _instance: ModelReferenceManager | None = None
    _lazy_mode: bool = True
    _replicate_mode: ReplicateMode = ReplicateMode.REPLICA

    _lock: RLock = RLock()

    @staticmethod
    def _create_backend(
        base_path: str | Path,
        replicate_mode: ReplicateMode,
    ) -> ModelReferenceBackend:
        """Create the appropriate backend based on mode and settings.

        Args:
            base_path (str | Path): Base path for model reference files.
            replicate_mode (ReplicateMode): The replication mode.

        Returns:
            ModelReferenceBackend: The configured backend instance.
        """
        if replicate_mode == ReplicateMode.PRIMARY:
            logger.debug("Creating backend for PRIMARY mode")

            filesystem_backend = FileSystemBackend(
                base_path=base_path,
                cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
                replicate_mode=ReplicateMode.PRIMARY,
            )

            if horde_model_reference_settings.github_seed_enabled:
                logger.info("GitHub seeding enabled for PRIMARY mode")

                all_paths = filesystem_backend.get_all_category_file_paths()
                missing_categories = [cat for cat, path in all_paths.items() if path is None or not path.exists()]

                if missing_categories:
                    logger.info(f"Missing categories detected: {missing_categories}. Seeding from GitHub...")

                    github_backend = GitHubBackend(
                        base_path=base_path,
                        replicate_mode=ReplicateMode.PRIMARY,
                    )

                    github_backend.fetch_all_categories(force_refresh=True)
                    logger.info("GitHub seeding completed")
                else:
                    logger.debug("All files exist, skipping GitHub seeding")

            if horde_model_reference_settings.redis is not None:
                logger.info("Wrapping FileSystemBackend with RedisBackend for distributed caching")
                return RedisBackend(
                    file_backend=filesystem_backend,
                    redis_settings=horde_model_reference_settings.redis,
                    cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
                )

            return filesystem_backend

        logger.debug("Creating backend for REPLICA mode")

        github_backend = GitHubBackend(
            base_path=base_path,
            replicate_mode=ReplicateMode.REPLICA,
        )

        if horde_model_reference_settings.primary_api_url:
            logger.info(f"Using HTTPBackend with PRIMARY API: {horde_model_reference_settings.primary_api_url}")
            return HTTPBackend(
                primary_api_url=horde_model_reference_settings.primary_api_url,
                github_backend=github_backend,
                cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
                timeout_seconds=horde_model_reference_settings.primary_api_timeout,
                enable_github_fallback=horde_model_reference_settings.enable_github_fallback,
            )

        logger.info("Using GitHubBackend only (no PRIMARY API configured)")
        return github_backend

    def __new__(
        cls,
        *,
        backend: ModelReferenceBackend | None = None,
        lazy_mode: bool = True,
        base_path: str | Path = horde_model_reference_paths.base_path,
        replicate_mode: ReplicateMode = horde_model_reference_settings.replicate_mode,
    ) -> ModelReferenceManager:
        """Create a new instance of ModelReferenceManager.

        Uses the singleton pattern to ensure only one instance exists to avoid multiple downloads and conversions.
        Subsequent instantiations will return the same instance, and an attempt to re-instantiate with different
        settings will raise an exception.

        Args:
            backend (ModelReferenceBackend | None, optional): The backend to use for fetching model references.
                If None, automatically selects the appropriate backend based on replicate_mode and settings:
                - PRIMARY mode: FileSystemBackend (optionally wrapped with RedisBackend if configured)
                - REPLICA mode: HTTPBackend (if PRIMARY API URL configured) or GitHubBackend (fallback)
                Defaults to None.
            lazy_mode (bool, optional): Whether to use lazy mode. In lazy mode, references are only downloaded
                when needed. Defaults to True.
            base_path (str | Path, optional): The base path to use for storing model reference files.
                Only used if backend is None. Defaults to horde_model_reference_paths.base_path.
            replicate_mode (ReplicateMode, optional): The replicate mode to use.
                - PRIMARY: Local filesystem is source of truth
                - REPLICA: Fetch from PRIMARY API or GitHub
                Only used if backend is None. Defaults to horde_model_reference_settings.replicate_mode.

        Returns:
            ModelReferenceManager: The singleton instance of ModelReferenceManager.

        Raises:
            RuntimeError: If an attempt is made to re-instantiate with different settings.

        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)

                if backend is None:
                    backend = cls._create_backend(base_path=base_path, replicate_mode=replicate_mode)

                backend_mode = backend.replicate_mode
                if backend_mode != replicate_mode:
                    raise RuntimeError(
                        "Backend replicate_mode does not match requested ModelReferenceManager configuration. "
                        f"Backend mode: {backend_mode}, requested mode: {replicate_mode}."
                    )

                cls._instance.backend = backend
                cls._instance._replicate_mode = replicate_mode
                cls._instance._lazy_mode = lazy_mode
                cls._instance._cached_file_json = {}
                cls._instance._cached_system_file_modified_times = {}

                for category, file_path in cls._instance.backend.get_all_category_file_paths().items():
                    if file_path and file_path.exists():
                        try:
                            cls._instance._cached_system_file_modified_times[category] = file_path.stat().st_mtime
                        except Exception:
                            cls._instance._cached_system_file_modified_times[category] = 0.0

                if not lazy_mode:
                    cls._instance._fetch_from_backend_if_needed(force_refresh=False)
            else:
                if backend is not None and backend is not cls._instance.backend:
                    raise RuntimeError(
                        "ModelReferenceManager is a singleton and has already been instantiated "
                        "with a different backend."
                    )
                if replicate_mode != cls._instance._replicate_mode or lazy_mode != cls._instance._lazy_mode:
                    raise RuntimeError(
                        "ModelReferenceManager is a singleton and has already been instantiated with different "
                        f"settings.\nExisting settings: "
                        f"replicate_mode={cls._instance._replicate_mode}, lazy_mode={cls._instance._lazy_mode}.\n"
                        f"New settings: "
                        f"replicate_mode={replicate_mode}, lazy_mode={lazy_mode}.",
                    )

        return cls._instance

    def _invalidate_cache(self, category: MODEL_REFERENCE_CATEGORY | None = None) -> None:
        """Invalidate the cached model references.

        Args:
            category (MODEL_REFERENCE_CATEGORY | None): If provided, only invalidate the specific category.
                If None, invalidate the entire cache.
        """
        with self._lock:
            if category is None:
                logger.debug("Invalidating entire cached model references.")
                self._cached_file_json = {}
            else:
                logger.debug(f"Invalidating cached model references for category: {category}.")
                self._cached_file_json.pop(category, None)

    def _mark_backend_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark a category as stale in the backend.

        When a model reference file changes, mark it in the backend so it knows
        to refresh the data on next access.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category that needs refresh.
        """
        with self._lock:
            logger.debug(f"Marking category {category} as stale in backend.")
            self.backend.mark_stale(category)

    def _check_file_modified_times(self) -> None:
        """Check model reference file mtimes and mark categories that need refresh due to external changes.

        This method detects when model reference files have been modified externally (e.g., by another process
        or manual edit) and marks them for refresh in the backend.

        When a category's mtime changes, only that category is invalidated from the cache.
        """
        with self._lock:
            for category, file_path in self.backend.get_all_category_file_paths().items():
                if file_path is None or not file_path.exists():
                    continue

                try:
                    current_mtime = file_path.stat().st_mtime
                    cached_modified_time = self._cached_system_file_modified_times.get(category)

                    if cached_modified_time is None:
                        self._cached_system_file_modified_times[category] = current_mtime
                    elif current_mtime != cached_modified_time:
                        logger.info(
                            f"Model reference file for {category} mtime changed "
                            f"(was {cached_modified_time}, now {current_mtime}), marking for refresh."
                        )
                        self._cached_system_file_modified_times[category] = current_mtime

                        self._invalidate_cache(category=category)

                        self._mark_backend_stale(category)
                except Exception as e:
                    logger.warning(f"Error checking mtime for {category}: {e}")

    def _fetch_from_backend_if_needed(
        self,
        force_refresh: bool,
    ) -> None:
        """Fetch references from backend if needed.

        Args:
            force_refresh (bool): Whether to force refresh all categories.
        """
        if self._replicate_mode == ReplicateMode.REPLICA:
            self.backend.fetch_all_categories(force_refresh=force_refresh)
        else:
            logger.debug(f"Not fetching from backend due to replicate mode: {self._replicate_mode}")

    async def _fetch_from_backend_if_needed_async(
        self,
        force_refresh: bool,
        httpx_client: httpx.AsyncClient | None,
    ) -> None:
        """Asynchronously fetch references from backend if needed.

        Args:
            force_refresh (bool): Whether to force refresh all categories.
            httpx_client (httpx.AsyncClient | None): An optional httpx async client to use.
        """
        if self._replicate_mode == ReplicateMode.REPLICA:
            await self.backend.fetch_all_categories_async(
                force_refresh=force_refresh,
                httpx_client=httpx_client,
            )
        else:
            logger.debug(f"Not fetching from backend due to replicate mode: {self._replicate_mode}")

    @staticmethod
    def _file_json_dict_to_model_reference(
        category: MODEL_REFERENCE_CATEGORY,
        file_json_dict: dict[str, Any] | None,
        safe_mode: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return a model reference object from a JSON dictionary, or None if conversion failed.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The target model reference category to convert.
            file_json_dict (dict): The dict object representing the model reference.
            safe_mode (bool, optional): Whether to raise exceptions on failure. If False, exceptions are caught
                and None is returned. Defaults to False.

        Returns:
            dict[str, GenericModelRecord] | None: The dict representing the model reference,
                or None if conversion failed.
        """
        if file_json_dict is None:
            logger.warning(f"File dict json is None for {category}.")
            return None

        try:
            record_type = MODEL_RECORD_TYPE_LOOKUP.get(category, GenericModelRecord)
            model_reference: dict[str, GenericModelRecord] = {}
            for model_value in file_json_dict.values():
                model_instance = record_type.model_validate(model_value)
                model_reference[model_instance.name] = model_instance

            return model_reference

        except Exception as e:
            if not safe_mode:
                logger.exception(f"Failed to convert file dict JSON to model reference for {category}: {e}")
                return None
            raise e

    @staticmethod
    def model_reference_to_json_dict(
        model_reference: dict[str, GenericModelRecord],
        safe_mode: bool = False,
    ) -> dict[str, Any] | None:
        """Return a JSON dictionary from a model reference object, or None if conversion failed.

        Args:
            model_reference (KNOWN_MODEL_REFERENCE_INSTANCES): The model reference object.
            safe_mode (bool, optional): Whether to raise exceptions on failure. If False, exceptions are caught
                and None is returned. Use `model_reference_to_json_dict_safe()` for the better type hinting if you
                intend to use this. Defaults to False.

        Returns:
            dict | None: The dict representing the model reference, or None if conversion failed.
        """
        if model_reference is None:
            raise ValueError("model_reference cannot be None")

        try:
            return {
                name: record.model_dump(
                    exclude_unset=True,
                )
                for name, record in model_reference.items()
            }
        except Exception as e:
            if not safe_mode:
                logger.exception(f"Failed to convert model reference to JSON: {e}")
                return None

            raise e

    @staticmethod
    def model_reference_to_json_dict_safe(
        model_reference: dict[str, GenericModelRecord],
    ) -> dict[str, Any]:
        """Return a JSON dictionary from a model reference object.

        Raises an exception if conversion fails.

        Args:
            model_reference (KNOWN_MODEL_REFERENCE_INSTANCES): The model reference object.

        Returns:
            dict: The dict representing the model reference.
        """
        json_dict_safe = ModelReferenceManager.model_reference_to_json_dict(model_reference, safe_mode=True)

        if json_dict_safe is None:
            raise RuntimeError("Conversion to JSON dict failed in safe mode, but no exception was raised.")

        return json_dict_safe

    def _get_all_cached_model_references(
        self,
        safe_mode: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]:
        """Get all cached model references.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        return_dict = {}
        with self._lock:
            for category, file_json in self._cached_file_json.items():
                model_reference = self._file_json_dict_to_model_reference(category, file_json, safe_mode=safe_mode)
                return_dict[category] = model_reference

        logger.debug(f"Returning {len(return_dict)} cached model references.")
        return return_dict

    def get_all_model_references_unsafe(
        self,
        override_existing: bool = False,
        *,
        safe_mode: bool = False,
    ) -> dict[
        MODEL_REFERENCE_CATEGORY,
        dict[str, GenericModelRecord] | None,
    ]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        Note that values may be None if the model reference file could not be found or parsed.

        Args:
            override_existing (bool, optional): Whether to force a redownload of all model reference files.
                Defaults to False.
            safe_mode (bool, optional): Whether to raise exceptions on failure. If False, exceptions are caught
                and None is returned for that category. Defaults to False. Use `get_all_model_references()`
                for the better type hinting if you intend to use this.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        with self._lock:
            self._check_file_modified_times()

            all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None]
            all_files = self.backend.get_all_category_file_paths()

            all_categories_cached = all(cat in self._cached_file_json for cat in all_files)

            needs_backend_refresh = override_existing or any(
                self.backend.needs_refresh(cat) for cat in MODEL_REFERENCE_CATEGORY
            )

            if not override_existing and all_categories_cached and not needs_backend_refresh:
                logger.debug("Using fully cached model references.")
                return self._get_all_cached_model_references(safe_mode=safe_mode)

            if needs_backend_refresh:
                self._fetch_from_backend_if_needed(force_refresh=override_existing)

            categories_to_load = []
            for category in all_files:
                if override_existing or category not in self._cached_file_json or self.backend.needs_refresh(category):
                    categories_to_load.append(category)

            if categories_to_load:
                logger.debug(f"Loading {len(categories_to_load)} model reference categories: {categories_to_load}")

            for category in categories_to_load:
                file_path = all_files[category]
                if file_path is None:
                    self._cached_file_json[category] = None
                    continue

                if not file_path.exists():
                    logger.warning(
                        f"Model reference file for {category} does not exist at {file_path}.",
                    )
                    self._cached_file_json[category] = None
                    continue

                with open(file_path) as f:
                    file_contents = f.read()
                try:
                    file_json = json.loads(file_contents)
                    self._cached_file_json[category] = file_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {category} from {file_path}: {e}")
                    self._cached_file_json[category] = None

            return self._get_all_cached_model_references(safe_mode=safe_mode)

    def get_all_model_references(
        self,
        override_existing: bool = False,
    ) -> dict[
        MODEL_REFERENCE_CATEGORY,
        dict[str, GenericModelRecord],
    ]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        If a model reference file could not be found or parsed, an exception is raised. If you want to allow
        missing model references, use `get_all_model_references_unsafe()` instead.

        Args:
            override_existing (bool, optional): Whether to force a redownload of all model reference files.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        all_references = self.get_all_model_references_unsafe(override_existing=override_existing)
        safe_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {}
        missing_references = []
        for category, reference in all_references.items():
            if reference is None:
                missing_references.append(category)
            else:
                safe_references[category] = reference
        if missing_references:
            raise ValueError(f"Missing model references for categories: {missing_references}")
        return safe_references

    def get_raw_model_reference_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> dict[str, Any] | None:
        """Return the raw JSON dict for a specific category without pydantic validation.

        This method returns the cached JSON data directly, avoiding the overhead of creating
        and serializing pydantic models. Ideal for API endpoints that need fast JSON responses.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to retrieve.
            override_existing (bool, optional): Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, Any] | None: The raw JSON dict for the category, or None if not found.
        """
        with self._lock:
            self._check_file_modified_times()

            all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None]
            all_files = self.backend.get_all_category_file_paths()

            needs_backend_refresh = override_existing or self.backend.needs_refresh(category)

            if not override_existing and category in self._cached_file_json and not needs_backend_refresh:
                logger.debug(f"Returning cached raw JSON for category: {category}")
                return self._cached_file_json[category]

            if needs_backend_refresh:
                self._fetch_from_backend_if_needed(force_refresh=override_existing)

            if override_existing or category not in self._cached_file_json or self.backend.needs_refresh(category):
                file_path = all_files.get(category)
                if file_path is None:
                    logger.warning(f"No file path for category: {category}")
                    self._cached_file_json[category] = None
                    return None

                if not file_path.exists():
                    logger.warning(f"Model reference file for {category} does not exist at {file_path}.")
                    self._cached_file_json[category] = None
                    return None

                with open(file_path) as f:
                    file_contents = f.read()
                try:
                    file_json = json.loads(file_contents)
                    self._cached_file_json[category] = file_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {category} from {file_path}: {e}")
                    self._cached_file_json[category] = None
                    return None

            return self._cached_file_json.get(category)

    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record: GenericModelRecord,
    ) -> None:
        """Update or create a model reference.

        This method is only available in PRIMARY mode with a backend that supports writes.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to update.
            model_name (str): The name of the model to update or create.
            record (GenericModelRecord): The model record object to save.

        Raises:
            RuntimeError: If backend doesn't support writes (not in PRIMARY mode).
            ValueError: If model_name doesn't match record.name.
        """
        if not self.backend.supports_writes():
            raise RuntimeError(
                "Cannot update model: backend does not support writes. "
                "Only PRIMARY mode instances with FileSystemBackend support write operations."
            )

        if model_name != record.name:
            raise ValueError(f"Model name mismatch: {model_name} != {record.name}")

        record_dict = record.model_dump(exclude_unset=True)

        self.backend.update_model(category, model_name, record_dict)

        with self._lock:
            self._invalidate_cache(category)

        logger.info(f"Updated model {model_name} in category {category}")

    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference.

        This method is only available in PRIMARY mode with a backend that supports writes.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category containing the model.
            model_name (str): The name of the model to delete.

        Raises:
            RuntimeError: If backend doesn't support writes (not in PRIMARY mode).
            KeyError: If model doesn't exist in the category.
        """
        if not self.backend.supports_writes():
            raise RuntimeError(
                "Cannot delete model: backend does not support writes. "
                "Only PRIMARY mode instances with FileSystemBackend support write operations."
            )

        self.backend.delete_model(category, model_name)

        with self._lock:
            self._invalidate_cache(category)

        logger.info(f"Deleted model {model_name} from category {category}")
