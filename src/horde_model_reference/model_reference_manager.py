from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

import httpx
from loguru import logger

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends import (
    FileSystemBackend,
    GitHubBackend,
    HTTPBackend,
    ModelReferenceBackend,
    RedisBackend,
)
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
    _cached_records: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]
    """Cache of pydantic model records by category."""

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
            base_path: Base path for model reference files.
            replicate_mode: The replication mode.

        Returns:
            ModelReferenceBackend: The configured backend instance.
        """
        logger.debug(f"Creating backend with replicate_mode={replicate_mode}, base_path={base_path}")
        if replicate_mode == ReplicateMode.PRIMARY:
            logger.debug("Creating backend for PRIMARY mode")

            # Check if GitHub seeding will be needed
            github_seeding_will_occur = False
            if horde_model_reference_settings.github_seed_enabled:
                # Quick check to see if any categories are missing
                # (we'll do proper check after backend creation)
                github_seeding_will_occur = True

            filesystem_backend = FileSystemBackend(
                base_path=base_path,
                cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
                replicate_mode=ReplicateMode.PRIMARY,
                skip_startup_metadata_population=github_seeding_will_occur,
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

                    # Populate metadata after seeding
                    logger.info("Populating metadata after GitHub seeding")
                    filesystem_backend.ensure_all_metadata_populated()
                else:
                    logger.debug("All files exist, skipping GitHub seeding")
                    # Files exist but seeding was skipped, so run metadata population
                    logger.info("Running metadata population check (seeding was skipped)")
                    filesystem_backend.ensure_all_metadata_populated()

            if horde_model_reference_settings.redis.use_redis:
                logger.info("Wrapping FileSystemBackend with RedisBackend for distributed caching")
                return RedisBackend(
                    file_backend=filesystem_backend,
                    redis_settings=horde_model_reference_settings.redis,
                    cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
                )

            logger.info("Using FileSystemBackend for single-worker PRIMARY deployment")
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
            backend: The backend to use for fetching model references.
                If None, automatically selects the appropriate backend based on replicate_mode and settings:
                - PRIMARY mode: FileSystemBackend (optionally wrapped with RedisBackend if configured)
                - REPLICA mode: HTTPBackend (if PRIMARY API URL configured) or GitHubBackend (fallback)
                Defaults to None.
            lazy_mode: Whether to use lazy mode. In lazy mode, references are only downloaded
                when needed. Defaults to True.
            base_path: The base path to use for storing model reference files.
                Only used if backend is None. Defaults to horde_model_reference_paths.base_path.
            replicate_mode: The replicate mode to use.
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
                cls._instance._cached_records = {}

                # Register invalidation callback so backend can notify us when cache is stale
                cls._instance.backend.register_invalidation_callback(cls._instance._on_backend_invalidated)

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

    def _on_backend_invalidated(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """On callback invoked by backend when a category's cache is invalidated.

        This ensures the pydantic model cache stays in sync with backend invalidations.

        Args:
            category: The category that was invalidated.
        """
        logger.debug(f"Backend invalidated category {category}, clearing pydantic cache")
        self._invalidate_cache(category)

    def _invalidate_cache(self, category: MODEL_REFERENCE_CATEGORY | None = None) -> None:
        """Invalidate the cached pydantic model references.

        Args:
            category: If provided, only invalidate the specific category.
                If None, invalidate the entire cache.
        """
        with self._lock:
            if category is None:
                logger.debug("Invalidating entire cached pydantic records.")
                self._cached_records = {}
            else:
                logger.debug(f"Invalidating cached pydantic records for category: {category}.")
                self._cached_records.pop(category, None)

    def _fetch_from_backend_if_needed(
        self,
        force_refresh: bool,
    ) -> None:
        """Fetch references from backend if needed.

        Args:
            force_refresh: Whether to force refresh all categories.
        """
        self.backend.fetch_all_categories(force_refresh=force_refresh)

    async def _fetch_from_backend_if_needed_async(
        self,
        force_refresh: bool,
        httpx_client: httpx.AsyncClient | None,
    ) -> None:
        """Asynchronously fetch references from backend if needed.

        Args:
            force_refresh: Whether to force refresh all categories.
            httpx_client: An optional httpx async client to use.
        """
        await self.backend.fetch_all_categories_async(
            force_refresh=force_refresh,
            httpx_client=httpx_client,
        )

    @staticmethod
    def _file_json_dict_to_model_reference(
        category: MODEL_REFERENCE_CATEGORY,
        file_json_dict: dict[str, Any] | None,
        safe_mode: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return a model reference object from a JSON dictionary, or None if conversion failed.

        Args:
            category: The target model reference category to convert.
            file_json_dict: The dict object representing the model reference.
            safe_mode: Whether to raise exceptions on failure. If False, exceptions are caught
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
            model_reference: The model reference object.
            safe_mode: Whether to raise exceptions on failure. If False, exceptions are caught
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
            model_reference: The model reference object.

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
        """Get all cached pydantic model references.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: A mapping of model reference
                categories to their corresponding pydantic model objects.
        """
        with self._lock:
            logger.debug(f"Returning {len(self._cached_records)} cached pydantic model references.")
            return dict(self._cached_records)

    def get_all_model_references_unsafe(
        self,
        overwrite_existing: bool = False,
        *,
        safe_mode: bool = False,
    ) -> dict[
        MODEL_REFERENCE_CATEGORY,
        dict[str, GenericModelRecord] | None,
    ]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        Note that values may be None if the model reference file could not be found or parsed.

        Args:
            overwrite_existing: Whether to force a redownload of all model reference files.
                Defaults to False.
            safe_mode: Whether to raise exceptions on failure. If False, exceptions are caught
                and None is returned for that category. Defaults to False. Use `get_all_model_references()`
                for the better type hinting if you intend to use this.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        with self._lock:
            # Check if all categories are cached and don't need refresh
            all_categories_cached = all(cat in self._cached_records for cat in MODEL_REFERENCE_CATEGORY)

            needs_backend_refresh = overwrite_existing or any(
                self.backend.needs_refresh(cat) for cat in MODEL_REFERENCE_CATEGORY
            )

            if not overwrite_existing and all_categories_cached and not needs_backend_refresh:
                logger.debug("Using fully cached pydantic model references.")
                return self._get_all_cached_model_references(safe_mode=safe_mode)

            logger.debug("Fetching model references from backend as needed.")
            self._fetch_from_backend_if_needed(force_refresh=overwrite_existing)

            # Determine which categories need to be loaded/refreshed
            categories_to_load = []
            for category in MODEL_REFERENCE_CATEGORY:
                # Retry if:
                # 1. Force overwrite requested
                # 2. Category not in cache yet
                # 3. Cached value is None (failed previous attempt)
                # 4. Backend says it needs refresh (stale/mtime changed)
                cached_value = self._cached_records.get(category)
                if (
                    overwrite_existing
                    or category not in self._cached_records
                    or cached_value is None
                    or self.backend.needs_refresh(category)
                ):
                    categories_to_load.append(category)

            if categories_to_load:
                logger.debug(f"Loading {len(categories_to_load)} model reference categories: {categories_to_load}")

            # Fetch from backend and convert to pydantic models
            for category in categories_to_load:
                file_json: dict[str, Any] | None
                file_json = self.backend.fetch_category(category, force_refresh=overwrite_existing)

                model_reference: dict[str, GenericModelRecord] | None
                model_reference = self._file_json_dict_to_model_reference(category, file_json, safe_mode=safe_mode)

                # Cache pydantic models (not raw JSON)
                self._cached_records[category] = model_reference

            return self._get_all_cached_model_references(safe_mode=safe_mode)

    def get_all_model_references(
        self,
        overwrite_existing: bool = False,
    ) -> dict[
        MODEL_REFERENCE_CATEGORY,
        dict[str, GenericModelRecord],
    ]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        If a model reference file could not be found or parsed, an exception is raised. If you want to allow
        missing model references, use `get_all_model_references_unsafe()` instead.

        Args:
            overwrite_existing: Whether to force a redownload of all model reference files.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        all_references = self.get_all_model_references_unsafe(overwrite_existing=overwrite_existing)
        safe_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {}
        missing_references = []
        for category, reference in all_references.items():
            if reference is None:
                missing_references.append(category)
                safe_references[category] = {}
            else:
                safe_references[category] = reference
        if missing_references:
            logger.error(f"Missing model references for categories: {missing_references}")

        return safe_references

    def get_model_reference_unsafe(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return the model reference object for a specific category.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, GenericModelRecord] | None: The model reference object for the category,
                or None if not found.
        """
        all_references = self.get_all_model_references_unsafe(overwrite_existing=overwrite_existing)
        return all_references.get(category)

    def get_model_reference(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> dict[str, GenericModelRecord]:
        """Return the model reference object for a specific category.

        Raises an exception if the model reference could not be found or parsed.
        If you want to allow missing model references, use `get_model_reference_unsafe()` instead.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, GenericModelRecord]: The model reference object for the category.

        """
        model_reference = self.get_model_reference_unsafe(
            category,
            overwrite_existing=overwrite_existing,
        )
        if model_reference is None:
            raise RuntimeError(f"Model reference for category {category} not found or could not be parsed.")

        return model_reference

    def get_model_names_unsafe(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> list[str] | None:
        """Return a list of model names for a specific category.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            list[str] | None: The list of model names for the category, or None if not found.
        """
        model_reference = self.get_model_reference_unsafe(
            category,
            overwrite_existing=overwrite_existing,
        )
        if model_reference is None:
            return None

        return list(model_reference.keys())

    def get_model_names(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> list[str]:
        """Return a list of model names for a specific category.

        Raises an exception if the model reference could not be found or parsed.
        If you want to allow missing model references, use `get_model_names_unsafe()` instead.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            list[str]: The list of model names for the category.
        """
        model_reference = self.get_model_reference(
            category,
            overwrite_existing=overwrite_existing,
        )
        if model_reference is None:
            raise RuntimeError(f"Model reference for category {category} not found or could not be parsed.")

        return list(model_reference.keys())

    def get_model_unsafe(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        overwrite_existing: bool = False,
    ) -> GenericModelRecord | None:
        """Return a specific model from a category.

        Args:
            category: The category to retrieve.
            model_name: The name of the model within the category.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            GenericModelRecord | None: The model record, or None if not found.
        """
        model_reference = self.get_model_reference_unsafe(
            category,
            overwrite_existing=overwrite_existing,
        )
        if model_reference is None:
            return None

        return model_reference.get(model_name)

    def get_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        overwrite_existing: bool = False,
    ) -> GenericModelRecord:
        """Return a specific model from a category.

        Raises an exception if the model could not be found or parsed.
        If you want to allow missing models, use `get_model_unsafe()` instead.

        Args:
            category: The category to retrieve.
            model_name: The name of the model within the category.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            GenericModelRecord: The model record.
        """
        model_reference = self.get_model_reference(
            category,
            overwrite_existing=overwrite_existing,
        )

        model_record = model_reference.get(model_name)
        if model_record is None:
            raise RuntimeError(f"Model {model_name} not found in category {category}.")

        return model_record

    def get_raw_model_reference_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> dict[str, Any] | None:
        """Return the raw JSON dict for a specific category without pydantic validation.

        This method delegates to the backend to fetch the raw JSON data directly,
        avoiding the overhead of creating pydantic models. Ideal for API endpoints
        that need fast JSON responses.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, Any] | None: The raw JSON dict for the category, or None if not found.
        """
        return self.backend.fetch_category(category, force_refresh=overwrite_existing)

    def get_raw_model_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        overwrite_existing: bool = False,
    ) -> dict[str, Any] | None:
        """Return the raw JSON dict for a specific model in a category without pydantic validation.

        This method delegates to the backend to fetch the raw JSON data directly,
        avoiding the overhead of creating pydantic models. Ideal for API endpoints
        that need fast JSON responses.

        Args:
            category: The category to retrieve.
            model_name: The name of the model within the category.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, Any] | None: The raw JSON dict for the model, or None if not found.
        """
        category_json = self.backend.fetch_category(category, force_refresh=overwrite_existing)

        if category_json is None:
            return None

        return category_json.get(model_name)
