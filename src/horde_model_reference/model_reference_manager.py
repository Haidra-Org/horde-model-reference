from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Generator, Iterable
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, overload

import httpx
from loguru import logger
from strenum import StrEnum

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.audit import AuditTrailWriter
from horde_model_reference.backends import (
    FileSystemBackend,
    GitHubBackend,
    HTTPBackend,
    ModelReferenceBackend,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY, categories_managed_elsewhere
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    AudioGenerationModelRecord,
    BlipModelRecord,
    ClipModelRecord,
    CodeformerModelRecord,
    ControlNetModelRecord,
    EsrganModelRecord,
    GenericModelRecord,
    GfpganModelRecord,
    ImageGenerationModelRecord,
    MiscellaneousModelRecord,
    SafetyCheckerModelRecord,
    TextGenerationModelRecord,
    VideoGenerationModelRecord,
)
from horde_model_reference.query import (
    ControlNetFieldName,
    ControlNetQuery,
    GenericFieldName,
    ImageGenerationQuery,
    ImageGenFieldName,
    ModelQuery,
    TextGenFieldName,
    TextModelQuery,
    build_controlnet_query,
    build_cross_category_query,
    build_image_query,
    build_query,
    build_text_query,
)

if TYPE_CHECKING:
    from horde_model_reference.integrations.data_merger import PopularModelResult
    from horde_model_reference.integrations.horde_api_models import HordeModelType
    from horde_model_reference.pending_queue import PendingQueueService


class PrefetchStrategy(StrEnum):
    """Controls when and how the manager fetches model references."""

    LAZY = "lazy"
    """Defer backend fetches until first access (legacy lazy_mode=True behavior)."""

    SYNC = "sync"
    """Immediately fetch all categories on the calling thread during initialization."""

    DEFERRED = "deferred"
    """Expose a handle the caller can trigger later (sync or async) without blocking init."""

    ASYNC = "async"
    """Automatically schedule a background async warm-up when an event loop is available."""

    NONE = "none"
    """Skip all automatic warm-up; callers must invoke caching helpers manually."""


TModelRecord = TypeVar("TModelRecord", bound=GenericModelRecord)


class ModelReferenceManager:
    """Singleton class for downloading and reading model reference files.

    This class is responsible for managing the lifecycle of model reference files,
    including downloading, caching, and providing access to the model references.

    Uses a pluggable backend architecture to support different data sources (GitHub, database, etc.).

        Settings on initialization (base_path, backend, prefetch_strategy, etc) are only set on the first instantiation
    (e.g. `ModelReferenceManager(base_path=...)`). Subsequent instantiations will return the same instance.

    Retrieve all model references with `get_all_model_references_or_none()`.
    """

    backend: ModelReferenceBackend
    """The backend provider for model reference data."""
    _cached_records: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]
    """Cache of pydantic model records by category."""

    _instance: ModelReferenceManager | None = None
    _replicate_mode: ReplicateMode = ReplicateMode.REPLICA
    _prefetch_strategy: PrefetchStrategy = PrefetchStrategy.SYNC
    _deferred_prefetch_handle: DeferredPrefetchHandle | None = None
    _async_prefetch_task: asyncio.Task[None] | None = None
    _audit_writer: AuditTrailWriter | None = None
    _pending_queue_service: PendingQueueService | None = None

    _lock: RLock = RLock()

    @classmethod
    def get_instance(cls) -> ModelReferenceManager:
        """Get the singleton instance of ModelReferenceManager.

        Returns:
            ModelReferenceManager: The singleton instance.

        Raises:
            RuntimeError: If the instance has not been created yet.
        """
        with cls._lock:
            if cls._instance is None:
                raise RuntimeError("ModelReferenceManager instance has not been created yet.")
            return cls._instance

    @classmethod
    def has_instance(cls) -> bool:
        """Check if the singleton instance has been created.

        Returns:
            bool: True if the instance exists, False otherwise.
        """
        with cls._lock:
            return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance so a fresh one can be created.

        Intended for testing and development only. Production code should not
        call this — the singleton is designed to live for the process lifetime.
        """
        with cls._lock:
            instance = cls._instance
            if instance is None:
                return

            if instance._deferred_prefetch_handle is not None:
                instance._deferred_prefetch_handle = None

            if instance._async_prefetch_task is not None and not instance._async_prefetch_task.done():
                instance._async_prefetch_task.cancel()
                instance._async_prefetch_task = None

            cls._instance = None

    @staticmethod
    def _create_backend(
        base_path: str | Path,
        replicate_mode: ReplicateMode,
        audit_writer: AuditTrailWriter | None,
    ) -> ModelReferenceBackend:
        """Create the appropriate backend based on mode and settings.

        Args:
            base_path: Base path for model reference files.
            replicate_mode: The replication mode.
            audit_writer: Optional audit writer used by write-capable backends.

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
                audit_writer=audit_writer,
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
                from horde_model_reference.backends.redis_backend import RedisBackend

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
        base_path: str | Path = horde_model_reference_paths.base_path,
        replicate_mode: ReplicateMode = horde_model_reference_settings.replicate_mode,
        prefetch_strategy: PrefetchStrategy = PrefetchStrategy.LAZY,
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
            base_path: The base path to use for storing model reference files.
                Only used if backend is None. Defaults to horde_model_reference_paths.base_path.
            replicate_mode: The replicate mode to use.
                - PRIMARY: Local filesystem is source of truth
                - REPLICA: Fetch from PRIMARY API or GitHub
                Only used if backend is None. Defaults to horde_model_reference_settings.replicate_mode.
            prefetch_strategy: Controls whether initial cache warm-up is skipped (LAZY/NONE),
                performed synchronously, deferred, or executed via background async task.
                Defaults to PrefetchStrategy.LAZY.

        Returns:
            ModelReferenceManager: The singleton instance of ModelReferenceManager.

        Raises:
            RuntimeError: If an attempt is made to re-instantiate with different settings.

        """
        if not isinstance(prefetch_strategy, PrefetchStrategy):
            try:
                prefetch_strategy = PrefetchStrategy(prefetch_strategy)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError(
                    f"prefetch_strategy must be one of: {', '.join(strategy.value for strategy in PrefetchStrategy)}"
                ) from exc

        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)

                audit_writer: AuditTrailWriter | None = None
                if horde_model_reference_settings.audit.enabled:
                    audit_writer = AuditTrailWriter(
                        root_path=horde_model_reference_paths.audit_path,
                        max_file_size_bytes=horde_model_reference_settings.audit.max_segment_bytes,
                    )

                if backend is None:
                    backend = cls._create_backend(
                        base_path=base_path,
                        replicate_mode=replicate_mode,
                        audit_writer=audit_writer,
                    )

                backend_mode = backend.replicate_mode
                if backend_mode != replicate_mode:
                    raise RuntimeError(
                        "Backend replicate_mode does not match requested ModelReferenceManager configuration. "
                        f"Backend mode: {backend_mode}, requested mode: {replicate_mode}."
                    )

                cls._instance.backend = backend
                cls._instance._replicate_mode = replicate_mode
                if backend.supports_writes():
                    cls._instance._audit_writer = audit_writer
                    cls._instance._pending_queue_service = cls._build_pending_queue_service(
                        audit_writer=audit_writer,
                    )
                else:
                    cls._instance._audit_writer = None
                    cls._instance._pending_queue_service = None
                cls._instance._cached_records = {}
                cls._instance._deferred_prefetch_handle = None
                cls._instance._async_prefetch_task = None

                # Register invalidation callback so backend can notify us when cache is stale
                cls._instance.backend.register_invalidation_callback(cls._instance._on_backend_invalidated)

                cls._instance._apply_prefetch_strategy(strategy=prefetch_strategy)
            else:
                if backend is not None and backend is not cls._instance.backend:
                    raise RuntimeError(
                        "ModelReferenceManager is a singleton and has already been instantiated "
                        "with a different backend."
                    )
                if replicate_mode != cls._instance._replicate_mode:
                    raise RuntimeError(
                        "ModelReferenceManager is a singleton and has already been instantiated with different "
                        "settings.\nExisting settings: "
                        f"replicate_mode={cls._instance._replicate_mode}.\n"
                        "New settings: "
                        f"replicate_mode={replicate_mode}."
                    )
                if prefetch_strategy != cls._instance._prefetch_strategy:
                    raise RuntimeError(
                        "ModelReferenceManager is a singleton and has already been instantiated with different "
                        "settings."
                        f"\nExisting prefetch_strategy={cls._instance._prefetch_strategy.value};"
                        f" new prefetch_strategy={prefetch_strategy.value}."
                    )

        return cls._instance

    def _apply_prefetch_strategy(self, *, strategy: PrefetchStrategy) -> None:
        """Apply the configured prefetch strategy once the backend is available."""
        self._prefetch_strategy = strategy
        self._deferred_prefetch_handle = None
        self._async_prefetch_task = None

        if strategy in (PrefetchStrategy.LAZY, PrefetchStrategy.NONE):
            logger.debug(f"prefetch skipped because strategy={strategy.value}")
            return

        if strategy is PrefetchStrategy.SYNC:
            self._fetch_from_backend_if_needed(force_refresh=False)
            return

        if strategy is PrefetchStrategy.DEFERRED:
            self._deferred_prefetch_handle = DeferredPrefetchHandle(manager=self, force_refresh=False)
            logger.info(
                "Deferred prefetch handle created; call run_sync/run_async to warm caches without blocking",
            )
            return

        if strategy is PrefetchStrategy.ASYNC:
            self._schedule_async_prefetch(force_refresh=False)
            return

        raise ValueError(f"Unsupported prefetch strategy: {strategy}")

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
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch references from backend if needed.

        Args:
            force_refresh: Whether to force refresh all categories.
        """
        return self.backend.fetch_all_categories(force_refresh=force_refresh)

    async def _fetch_from_backend_if_needed_async(
        self,
        force_refresh: bool,
        httpx_client: httpx.AsyncClient | None,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch references from backend if needed.

        Args:
            force_refresh: Whether to force refresh all categories.
            httpx_client: An optional httpx async client to use.
        """
        return await self.backend.fetch_all_categories_async(
            force_refresh=force_refresh,
            httpx_client=httpx_client,
        )

    @staticmethod
    def _build_pending_queue_service(
        *,
        audit_writer: AuditTrailWriter | None,
    ) -> PendingQueueService | None:
        """Create the pending queue service when enabled."""
        if not horde_model_reference_settings.pending_queue.enabled:
            return None

        from horde_model_reference.pending_queue.service import PendingQueueService
        from horde_model_reference.pending_queue.store import PendingQueueStore

        store = PendingQueueStore(root_path=horde_model_reference_paths.pending_queue_path)
        return PendingQueueService(store=store, audit_writer=audit_writer)

    @property
    def prefetch_strategy(self) -> PrefetchStrategy:
        """Return the prefetch strategy originally configured for this manager."""
        return self._prefetch_strategy

    @property
    def pending_queue_service(self) -> PendingQueueService | None:
        """Return the pending queue service when queueing is enabled."""
        return self._pending_queue_service

    @property
    def deferred_prefetch_handle(self) -> DeferredPrefetchHandle | None:
        """Handle that callers can use to trigger a deferred eager fetch."""
        return self._deferred_prefetch_handle

    def create_deferred_prefetch_handle(
        self,
        *,
        force_refresh: bool = False,
    ) -> DeferredPrefetchHandle:
        """Create a deferred prefetch handle tied to this manager.

        Args:
            force_refresh: Whether the handle should bypass backend caches.

        Returns:
            DeferredPrefetchHandle: Handle that can execute the warm-up later.
        """
        handle = DeferredPrefetchHandle(manager=self, force_refresh=force_refresh)
        self._deferred_prefetch_handle = handle
        return handle

    def _schedule_async_prefetch(self, *, force_refresh: bool) -> None:
        """Schedule an async cache warm-up when an event loop is available."""
        handle = self.create_deferred_prefetch_handle(force_refresh=force_refresh)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "PrefetchStrategy.ASYNC requested but no running event loop detected; "
                "exposing deferred handle for manual execution instead.",
            )
            self._async_prefetch_task = None
            return

        logger.info("Scheduling asynchronous prefetch warm-up task")
        task = loop.create_task(handle.run_async())
        self._async_prefetch_task = task

        def _log_completion(completed: asyncio.Task[None]) -> None:
            try:
                completed.result()
            except Exception as exc:  # pragma: no cover - best-effort logging
                logger.error("Deferred async prefetch failed: %s", exc)

        task.add_done_callback(_log_completion)

    async def warm_cache_async(
        self,
        *,
        force_refresh: bool = False,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Warm cached pydantic records using backend async APIs.

        Args:
            force_refresh: Whether to bypass backend caches while warming.
            httpx_client: Optional shared async client for HTTP backends.
        """
        await self.get_all_model_references_or_none_async(
            overwrite_existing=force_refresh,
            httpx_client=httpx_client,
        )

    async def ensure_ready_async(
        self,
        *,
        overwrite_existing: bool = False,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Ensure cached references exist by delegating to ``warm_cache_async``.

        Args:
            overwrite_existing: Whether to bypass backend caches while warming.
            httpx_client: Optional shared async client for HTTP backends.
        """
        await self.warm_cache_async(force_refresh=overwrite_existing, httpx_client=httpx_client)

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

        if category in categories_managed_elsewhere:
            logger.info(f"Skipping conversion for category: {category} (managed elsewhere)")
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

    def _evaluate_cache_state(
        self,
        *,
        overwrite_existing: bool,
        safe_mode: bool,
    ) -> tuple[
        bool,
        dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None],
        list[MODEL_REFERENCE_CATEGORY],
    ]:
        """Return whether cached data can be reused plus categories needing refresh."""
        with self._lock:
            refresh_map = {category: self.backend.needs_refresh(category) for category in MODEL_REFERENCE_CATEGORY}
            all_categories_cached = all(cat in self._cached_records for cat in MODEL_REFERENCE_CATEGORY)
            needs_backend_refresh = overwrite_existing or any(refresh_map.values())

            if not overwrite_existing and all_categories_cached and not needs_backend_refresh:
                logger.debug("Using fully cached pydantic model references.")
                return True, self._get_all_cached_model_references(safe_mode=safe_mode), []

            categories_to_load: list[MODEL_REFERENCE_CATEGORY] = []
            for category in MODEL_REFERENCE_CATEGORY:
                cached_value = self._cached_records.get(category)
                if (
                    overwrite_existing
                    or category not in self._cached_records
                    or cached_value is None
                    or refresh_map[category]
                ):
                    categories_to_load.append(category)

            return False, {}, categories_to_load

    def _load_categories_from_payload(
        self,
        *,
        categories_to_load: Iterable[MODEL_REFERENCE_CATEGORY],
        payload: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] | None,
        overwrite_existing: bool,
        safe_mode: bool,
    ) -> None:
        """Convert backend payload into cached pydantic models for selected categories."""
        normalized_payload = payload or {}
        prepared_payload: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}
        missing_payload: list[MODEL_REFERENCE_CATEGORY] = []

        for category in categories_to_load:
            if category in normalized_payload:
                prepared_payload[category] = normalized_payload[category]
            else:
                missing_payload.append(category)

        if missing_payload:
            logger.debug(
                "Backend payload missing %d categories; falling back to per-category fetch: %s",
                len(missing_payload),
                missing_payload,
            )
            for category in missing_payload:
                prepared_payload[category] = self.backend.fetch_category(
                    category,
                    force_refresh=overwrite_existing,
                )

        with self._lock:
            for category, file_json in prepared_payload.items():
                model_reference = self._file_json_dict_to_model_reference(
                    category,
                    file_json,
                    safe_mode=safe_mode,
                )
                self._cached_records[category] = model_reference

    def get_all_model_references_or_none(
        self,
        overwrite_existing: bool = False,
        *,
        safe_mode: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]:
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
        use_cache, cached_result, categories_to_load = self._evaluate_cache_state(
            overwrite_existing=overwrite_existing,
            safe_mode=safe_mode,
        )

        if use_cache:
            return cached_result

        logger.debug("Fetching model references from backend as needed.")
        backend_payload = self._fetch_from_backend_if_needed(force_refresh=overwrite_existing)

        if categories_to_load:
            logger.debug("Loading %d model reference categories: %s", len(categories_to_load), categories_to_load)
            self._load_categories_from_payload(
                categories_to_load=categories_to_load,
                payload=backend_payload,
                overwrite_existing=overwrite_existing,
                safe_mode=safe_mode,
            )

        return self._get_all_cached_model_references(safe_mode=safe_mode)

    def _build_safe_reference_view(
        self,
        all_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None],
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]:
        """Convert a possibly sparse reference view into a safe mapping with logging.

        Args:
            all_references: Mapping of categories to model reference dicts or None.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: Mapping where
            missing categories map to empty dicts.
        """
        safe_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {}
        missing_references: list[MODEL_REFERENCE_CATEGORY] = []
        for category, reference in all_references.items():
            if reference is None:
                missing_references.append(category)
                safe_references[category] = {}
            else:
                safe_references[category] = reference

        if missing_references:
            logger.error(f"Missing model references for categories: {missing_references}")

        return safe_references

    def get_all_model_references(
        self,
        overwrite_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]:
        """Return a mapping of all model reference categories to their corresponding model reference objects.

        If a model reference file could not be found or parsed, an exception is raised. If you want to allow
        missing model references, use `get_all_model_references_or_none()` instead.

        Args:
            overwrite_existing: Whether to force a redownload of all model reference files.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        all_references = self.get_all_model_references_or_none(overwrite_existing=overwrite_existing)
        return self._build_safe_reference_view(all_references)

    async def get_all_model_references_or_none_async(
        self,
        overwrite_existing: bool = False,
        *,
        safe_mode: bool = False,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]:
        """Return model references asynchronously without enforcing presence.

        Args:
            overwrite_existing: Whether to force backend refresh.
            safe_mode: Whether to propagate conversion errors.
            httpx_client: Optional shared async client for HTTP backends.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]: Possibly
            sparse mapping keyed by category.
        """
        use_cache, cached_result, categories_to_load = self._evaluate_cache_state(
            overwrite_existing=overwrite_existing,
            safe_mode=safe_mode,
        )

        if use_cache:
            return cached_result

        logger.debug("Asynchronously fetching model references from backend as needed.")
        backend_payload = await self._fetch_from_backend_if_needed_async(
            force_refresh=overwrite_existing,
            httpx_client=httpx_client,
        )

        if categories_to_load:
            logger.debug("Loading %d model reference categories via async payload", len(categories_to_load))
            self._load_categories_from_payload(
                categories_to_load=categories_to_load,
                payload=backend_payload,
                overwrite_existing=overwrite_existing,
                safe_mode=safe_mode,
            )

        return self._get_all_cached_model_references(safe_mode=safe_mode)

    async def get_all_model_references_async(
        self,
        overwrite_existing: bool = False,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]:
        """Return all model references asynchronously, raising on missing categories.

        Args:
            overwrite_existing: Whether to force backend refresh.
            httpx_client: Optional shared async client for HTTP backends.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]: Mapping with
            empty dicts substituted for missing categories.
        """
        all_references = await self.get_all_model_references_or_none_async(
            overwrite_existing=overwrite_existing,
            httpx_client=httpx_client,
        )
        return self._build_safe_reference_view(all_references)

    def get_model_reference_or_none(
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
        all_references = self.get_all_model_references_or_none(overwrite_existing=overwrite_existing)
        return all_references.get(category)

    async def get_model_reference_or_none_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> dict[str, GenericModelRecord] | None:
        """Return a single category's references asynchronously without strict enforcement.

        Args:
            category: Target category to load.
            overwrite_existing: Whether to force backend refresh.
            httpx_client: Optional shared async client for HTTP backends.

        Returns:
            dict[str, GenericModelRecord] | None: Mapping of model names or None.
        """
        all_references = await self.get_all_model_references_or_none_async(
            overwrite_existing=overwrite_existing,
            httpx_client=httpx_client,
        )
        return all_references.get(category)

    def get_model_reference(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
    ) -> dict[str, GenericModelRecord]:
        """Return the model reference object for a specific category.

        Raises an exception if the model reference could not be found or parsed.
        If you want to allow missing model references, use `get_model_reference_or_none()` instead.

        Args:
            category: The category to retrieve.
            overwrite_existing: Whether to force a redownload. Defaults to False.

        Returns:
            dict[str, GenericModelRecord]: The model reference object for the category.

        """
        model_reference = self.get_model_reference_or_none(
            category,
            overwrite_existing=overwrite_existing,
        )
        if model_reference is None:
            raise RuntimeError(f"Model reference for category {category} not found or could not be parsed.")

        return model_reference

    async def get_model_reference_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        overwrite_existing: bool = False,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> dict[str, GenericModelRecord]:
        """Return a single category's references asynchronously, raising if missing.

        Args:
            category: Target category to load.
            overwrite_existing: Whether to force backend refresh.
            httpx_client: Optional shared async client for HTTP backends.

        Returns:
            dict[str, GenericModelRecord]: Mapping of model names for the category.

        Raises:
            RuntimeError: If the category is missing or could not be parsed.
        """
        model_reference = await self.get_model_reference_or_none_async(
            category,
            overwrite_existing=overwrite_existing,
            httpx_client=httpx_client,
        )
        if model_reference is None:
            raise RuntimeError(f"Model reference for category {category} not found or could not be parsed.")

        return model_reference

    def get_model_names_or_none(
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
        model_reference = self.get_model_reference_or_none(
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
        If you want to allow missing model references, use `get_model_names_or_none()` instead.

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

    def get_model_or_none(
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
        model_reference = self.get_model_reference_or_none(
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
        If you want to allow missing models, use `get_model_or_none()` instead.

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

    def _get_typed_models(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        record_type: type[TModelRecord],
    ) -> dict[str, TModelRecord]:
        """Return a typed mapping for the requested category."""
        model_reference = self.get_model_reference(category)

        if len(model_reference) == 0:
            return {}

        typed_reference: dict[str, TModelRecord] = {}
        for name, record in model_reference.items():
            if not isinstance(record, record_type):
                raise RuntimeError(
                    f"Some records in {category.value} category are not {record_type.__name__} instances."
                )
            typed_reference[name] = record

        return typed_reference

    @property
    def blip_models(self) -> dict[str, BlipModelRecord]:
        """Return a mapping of BLIP model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.blip,
            record_type=BlipModelRecord,
        )

    @property
    def clip_models(self) -> dict[str, ClipModelRecord]:
        """Return a mapping of CLIP model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.clip,
            record_type=ClipModelRecord,
        )

    @property
    def codeformer_models(self) -> dict[str, CodeformerModelRecord]:
        """Return a mapping of CodeFormer model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.codeformer,
            record_type=CodeformerModelRecord,
        )

    @property
    def controlnet_models(self) -> dict[str, ControlNetModelRecord]:
        """Return a mapping of ControlNet model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.controlnet,
            record_type=ControlNetModelRecord,
        )

    @property
    def esrgan_models(self) -> dict[str, EsrganModelRecord]:
        """Return a mapping of ESRGAN model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.esrgan,
            record_type=EsrganModelRecord,
        )

    @property
    def gfpgan_models(self) -> dict[str, GfpganModelRecord]:
        """Return a mapping of GFPGAN model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.gfpgan,
            record_type=GfpganModelRecord,
        )

    @property
    def safety_checker_models(self) -> dict[str, SafetyCheckerModelRecord]:
        """Return a mapping of safety checker model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.safety_checker,
            record_type=SafetyCheckerModelRecord,
        )

    @property
    def image_generation_models(self) -> dict[str, ImageGenerationModelRecord]:
        """Return a mapping of image generation model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.image_generation,
            record_type=ImageGenerationModelRecord,
        )

    @property
    def text_generation_models(self) -> dict[str, TextGenerationModelRecord]:
        """Return a mapping of text generation model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.text_generation,
            record_type=TextGenerationModelRecord,
        )

    @property
    def audio_generation_models(self) -> dict[str, AudioGenerationModelRecord]:
        """Return a mapping of audio generation model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.audio_generation,
            record_type=AudioGenerationModelRecord,
        )

    @property
    def video_generation_models(self) -> dict[str, VideoGenerationModelRecord]:
        """Return a mapping of video generation model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.video_generation,
            record_type=VideoGenerationModelRecord,
        )

    @property
    def miscellaneous_models(self) -> dict[str, MiscellaneousModelRecord]:
        """Return a mapping of miscellaneous model names to their records."""
        return self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.miscellaneous,
            record_type=MiscellaneousModelRecord,
        )

    @overload
    def query(self, category: Literal["image_generation"]) -> ImageGenerationQuery: ...

    @overload
    def query(self, category: Literal["text_generation"]) -> TextModelQuery: ...

    @overload
    def query(self, category: Literal["controlnet"]) -> ControlNetQuery: ...

    @overload
    def query(
        self,
        category: str,
    ) -> ModelQuery[
        GenericModelRecord, GenericFieldName | ImageGenFieldName | TextGenFieldName | ControlNetFieldName
    ]: ...

    def query(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
    ) -> (
        ImageGenerationQuery
        | TextModelQuery
        | ControlNetQuery
        | ModelQuery[GenericModelRecord, GenericFieldName | ImageGenFieldName | TextGenFieldName | ControlNetFieldName]
    ):
        """Return a query builder for a single category.

        When called with a literal category string, the return type is
        narrowed to the corresponding typed query builder (e.g.
        ``ImageGenerationQuery`` for ``"image_generation"``).

        Args:
            category: The model reference category to query.

        Returns:
            A ``ModelQuery`` (or typed subclass) ready for chaining filters.
        """
        if isinstance(category, str):
            category = MODEL_REFERENCE_CATEGORY(category)

        if category == MODEL_REFERENCE_CATEGORY.image_generation:
            return self.query_image_generation()
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            return self.query_text_generation()
        if category == MODEL_REFERENCE_CATEGORY.controlnet:
            return self.query_controlnet()

        records = self.get_model_reference(category)
        record_type = MODEL_RECORD_TYPE_LOOKUP.get(category, GenericModelRecord)
        return build_query(records, record_type)

    def query_all(
        self,
    ) -> ModelQuery[GenericModelRecord, GenericFieldName | ImageGenFieldName | TextGenFieldName | ControlNetFieldName]:
        """Return a query builder spanning all categories.

        Returns:
            A ``ModelQuery[GenericModelRecord]`` over every cached record.
        """
        all_refs = self.get_all_model_references()
        return build_cross_category_query(all_refs)

    def query_image_generation(self) -> ImageGenerationQuery:
        """Return a typed query builder for image generation models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.image_generation,
            record_type=ImageGenerationModelRecord,
        )
        return build_image_query(records)

    def query_text_generation(self) -> TextModelQuery:
        """Return a typed query builder for text generation models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.text_generation,
            record_type=TextGenerationModelRecord,
        )
        return build_text_query(records)

    def query_controlnet(self) -> ControlNetQuery:
        """Return a typed query builder for ControlNet models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.controlnet,
            record_type=ControlNetModelRecord,
        )
        return build_controlnet_query(records)

    def query_blip(self) -> ModelQuery[BlipModelRecord, GenericFieldName]:
        """Return a typed query builder for BLIP models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.blip,
            record_type=BlipModelRecord,
        )
        return build_query(records, BlipModelRecord)

    def query_clip(self) -> ModelQuery[ClipModelRecord, GenericFieldName]:
        """Return a typed query builder for CLIP models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.clip,
            record_type=ClipModelRecord,
        )
        return build_query(records, ClipModelRecord)

    def query_codeformer(self) -> ModelQuery[CodeformerModelRecord, GenericFieldName]:
        """Return a typed query builder for CodeFormer models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.codeformer,
            record_type=CodeformerModelRecord,
        )
        return build_query(records, CodeformerModelRecord)

    def query_esrgan(self) -> ModelQuery[EsrganModelRecord, GenericFieldName]:
        """Return a typed query builder for ESRGAN models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.esrgan,
            record_type=EsrganModelRecord,
        )
        return build_query(records, EsrganModelRecord)

    def query_gfpgan(self) -> ModelQuery[GfpganModelRecord, GenericFieldName]:
        """Return a typed query builder for GFPGAN models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.gfpgan,
            record_type=GfpganModelRecord,
        )
        return build_query(records, GfpganModelRecord)

    def query_safety_checker(self) -> ModelQuery[SafetyCheckerModelRecord, GenericFieldName]:
        """Return a typed query builder for safety checker models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.safety_checker,
            record_type=SafetyCheckerModelRecord,
        )
        return build_query(records, SafetyCheckerModelRecord)

    def query_audio_generation(self) -> ModelQuery[AudioGenerationModelRecord, GenericFieldName]:
        """Return a typed query builder for audio generation models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.audio_generation,
            record_type=AudioGenerationModelRecord,
        )
        return build_query(records, AudioGenerationModelRecord)

    def query_video_generation(self) -> ModelQuery[VideoGenerationModelRecord, GenericFieldName]:
        """Return a typed query builder for video generation models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.video_generation,
            record_type=VideoGenerationModelRecord,
        )
        return build_query(records, VideoGenerationModelRecord)

    def query_miscellaneous(self) -> ModelQuery[MiscellaneousModelRecord, GenericFieldName]:
        """Return a typed query builder for miscellaneous models."""
        records = self._get_typed_models(
            MODEL_REFERENCE_CATEGORY.miscellaneous,
            record_type=MiscellaneousModelRecord,
        )
        return build_query(records, MiscellaneousModelRecord)

    _CATEGORY_TO_HORDE_TYPE: ClassVar[dict[MODEL_REFERENCE_CATEGORY, HordeModelType]] = {
        MODEL_REFERENCE_CATEGORY.image_generation: "image",
        MODEL_REFERENCE_CATEGORY.text_generation: "text",
    }

    async def get_popular_models(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        limit: int = 10,
        sort_by: Literal["worker_count", "usage_day", "usage_month", "usage_total"] = "worker_count",
        include_workers: bool = False,
    ) -> list[PopularModelResult]:
        """Return models ranked by live Horde popularity metrics.

        Requires the Horde public API to be reachable. Only ``image_generation``
        and ``text_generation`` categories have Horde API data; other categories
        return an empty list.

        Args:
            category: Model category to rank.
            limit: Maximum number of results.
            sort_by: Metric to rank by.
            include_workers: Whether to fetch per-worker details (slower).

        Returns:
            A list of ``PopularModelResult`` sorted by the chosen metric.
        """
        from horde_model_reference.integrations.data_merger import (
            CombinedModelStatistics,
            PopularModelResult,
            merge_category_with_horde_data,
        )
        from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration

        horde_type: HordeModelType | None = self._CATEGORY_TO_HORDE_TYPE.get(category)
        if horde_type is None:
            return []

        model_reference = self.get_model_reference_or_none(category)
        if model_reference is None:
            return []

        horde_api = HordeAPIIntegration()
        indexed_status, indexed_stats, indexed_workers = await horde_api.get_combined_data_indexed(
            model_type=horde_type,
            include_workers=include_workers,
        )

        merged = merge_category_with_horde_data(
            model_names=model_reference.keys(),
            horde_status=indexed_status,
            horde_stats=indexed_stats,
            workers=indexed_workers,
        )

        def _sort_key(item: tuple[str, object]) -> float:
            _name, stats = item
            if not isinstance(stats, CombinedModelStatistics):
                return 0.0
            if sort_by == "worker_count":
                return float(stats.worker_count)
            if stats.usage_stats is None:
                return 0.0
            if sort_by == "usage_day":
                return float(stats.usage_stats.day)
            if sort_by == "usage_month":
                return float(stats.usage_stats.month)
            return float(stats.usage_stats.total)

        ranked = sorted(merged.items(), key=_sort_key, reverse=True)[:limit]

        results: list[PopularModelResult] = []
        for name, stats in ranked:
            record = model_reference.get(name)
            if record is None:
                continue
            results.append(
                PopularModelResult(
                    name=name,
                    record=record.model_dump(mode="json", exclude_none=True),
                    stats=stats,
                )
            )

        return results


class DeferredPrefetchHandle(Awaitable[None]):
    """Encapsulates a deferred eager fetch for a `ModelReferenceManager`."""

    def __init__(
        self,
        *,
        manager: ModelReferenceManager,
        force_refresh: bool,
    ) -> None:
        """Store the manager reference and desired refresh semantics."""
        self._manager = manager
        self._force_refresh = force_refresh

    @property
    def force_refresh(self) -> bool:
        """Whether this handle forces a backend refresh when executed."""
        return self._force_refresh

    def run_sync(self) -> None:
        """Execute the deferred fetch synchronously on the current thread."""
        self._manager._fetch_from_backend_if_needed(force_refresh=self._force_refresh)

    async def run_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Execute the deferred fetch asynchronously using the backend's async API."""
        await self._manager._fetch_from_backend_if_needed_async(
            force_refresh=self._force_refresh,
            httpx_client=httpx_client,
        )

    def __await__(self) -> Generator[Any]:
        """Allow awaiting the handle directly as sugar for run_async()."""
        return self.run_async().__await__()
