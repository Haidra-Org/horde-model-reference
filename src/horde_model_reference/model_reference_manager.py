from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

import aiohttp
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import MODEL_RECORD_TYPE_LOOKUP, GenericModelRecord


class ModelReferenceManager:
    """Singleton class for downloading and reading model reference files.

    This class is responsible for managing the lifecycle of model reference files,
    including downloading, caching, and providing access to the model references.

    Uses a pluggable backend architecture to support different data sources (GitHub, database, etc.).

    Settings on initialization (base_path, backend, lazy_mode, etc) are only set on the first instantiation
    (e.g. `ModelReferenceManager(base_path=...)`). Subsequent instantiations will return the same instance.

    Retrieve all model references with `get_all_model_references()`.
    """

    backend: ModelReferenceBackend
    """The backend provider for model reference data."""
    _cached_file_json: dict[MODEL_REFERENCE_CATEGORY, dict[Any, Any] | None]
    _cached_system_file_modified_times: dict[MODEL_REFERENCE_CATEGORY, float]

    _instance: ModelReferenceManager | None = None
    _lazy_mode: bool = True
    _replicate_mode: ReplicateMode = ReplicateMode.REPLICA

    _lock: RLock = RLock()

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
                If None, defaults to LegacyGitHubBackend. Defaults to None.
            lazy_mode (bool, optional): Whether to use lazy mode. In lazy mode, references are only downloaded
                when needed. Defaults to True.
            base_path (str | Path, optional): The base path to use for storing model reference files.
                Only used if backend is None. Defaults to horde_model_reference_paths.base_path.
            replicate_mode (ReplicateMode, optional): The replicate mode to use. If not REPLICA,
                references will not be downloaded, and only local files will be used.
                Only used if backend is None. Defaults to horde_model_reference_settings.replicate_mode.

        Returns:
            ModelReferenceManager: The singleton instance of ModelReferenceManager.

        Raises:
            RuntimeError: If an attempt is made to re-instantiate with different settings.

        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)

                # Create backend if not provided
                if backend is None:
                    from horde_model_reference.backends import LegacyGitHubBackend

                    backend = LegacyGitHubBackend(
                        base_path=base_path,
                        replicate_mode=replicate_mode,
                    )

                cls._instance.backend = backend
                cls._instance._replicate_mode = replicate_mode
                cls._instance._lazy_mode = lazy_mode
                cls._instance._cached_file_json = {}
                cls._instance._cached_system_file_modified_times = {}

                # Initialize mtime tracking for existing files
                for category, file_path in cls._instance.backend.get_all_category_file_paths().items():
                    if file_path and file_path.exists():
                        try:
                            cls._instance._cached_system_file_modified_times[category] = file_path.stat().st_mtime
                        except Exception:
                            cls._instance._cached_system_file_modified_times[category] = 0.0

                if not lazy_mode:
                    cls._instance._fetch_from_backend_if_needed(force_refresh=False)
            else:
                # Check if settings match for existing instance
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
                        # Invalidate only this specific category from cache
                        self._invalidate_cache(category=category)
                        # Mark stale in backend
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
            # Fetch all categories from backend
            self.backend.fetch_all_categories(force_refresh=force_refresh)
        else:
            logger.debug(f"Not fetching from backend due to replicate mode: {self._replicate_mode}")

    async def _fetch_from_backend_if_needed_async(
        self,
        force_refresh: bool,
        aiohttp_client_session: aiohttp.ClientSession,
    ) -> None:
        """Asynchronously fetch references from backend if needed.

        Args:
            force_refresh (bool): Whether to force refresh all categories.
            aiohttp_client_session (aiohttp.ClientSession): An existing aiohttp client session to use.
        """
        if self._replicate_mode == ReplicateMode.REPLICA:
            # Fetch all categories from backend asynchronously
            await self.backend.fetch_all_categories_async(
                force_refresh=force_refresh,
                aiohttp_client_session=aiohttp_client_session,
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

    def get_all_model_references(
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
                and None is returned for that category. Defaults to False. Use `get_all_model_reference_safe()`
                for the better type hinting if you intend to use this.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        with self._lock:
            self._check_file_modified_times()

            # Check if we have a complete cache and nothing needs refresh
            all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = self.backend.get_all_category_file_paths()
            all_categories_cached = all(cat in self._cached_file_json for cat in all_files)

            # Check if any categories need refresh from backend
            needs_backend_refresh = override_existing or any(
                self.backend.needs_refresh(cat) for cat in MODEL_REFERENCE_CATEGORY
            )

            if not override_existing and all_categories_cached and not needs_backend_refresh:
                logger.debug("Using fully cached model references.")
                return self._get_all_cached_model_references(safe_mode=safe_mode)

            # Fetch from backend if needed
            if needs_backend_refresh:
                self._fetch_from_backend_if_needed(force_refresh=override_existing)

            # Determine which categories need to be loaded/reloaded from disk
            categories_to_load = []
            for category in all_files:
                if override_existing or category not in self._cached_file_json or self.backend.needs_refresh(category):
                    categories_to_load.append(category)

            if categories_to_load:
                logger.debug(f"Loading {len(categories_to_load)} model reference categories: {categories_to_load}")

            # Load the categories that need loading
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

    def get_all_model_reference_safe(
        self,
        override_existing: bool = False,
    ) -> dict[
        MODEL_REFERENCE_CATEGORY,
        dict[str, GenericModelRecord],
    ]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        If a model reference file could not be found or parsed, an exception is raised. If you want to allow
        missing model references, use `get_all_model_references()` instead.

        Args:
            override_existing (bool, optional): Whether to force a redownload of all model reference files.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        all_references = self.get_all_model_references(override_existing=override_existing)
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

            # Check if we have this category cached and if it needs refresh
            all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = self.backend.get_all_category_file_paths()

            # Check if category needs refresh from backend
            needs_backend_refresh = override_existing or self.backend.needs_refresh(category)

            if not override_existing and category in self._cached_file_json and not needs_backend_refresh:
                logger.debug(f"Returning cached raw JSON for category: {category}")
                return self._cached_file_json[category]

            # Fetch from backend if needed
            if needs_backend_refresh:
                self._fetch_from_backend_if_needed(force_refresh=override_existing)

            # Load from disk if needed
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
