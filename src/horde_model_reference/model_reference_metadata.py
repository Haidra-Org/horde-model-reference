"""Backend metadata tracking system for model reference operations.

This module provides centralized metadata tracking for both legacy (v1) and v2 format operations.
All metadata updates go through MetadataManager methods to enable observability hooks.
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from strenum import StrEnum

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class OperationType(StrEnum):
    """Type of CRUD operation performed on model references."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class CategoryMetadata(BaseModel):
    """Metadata tracking for a single model reference category.

    Tracks operation counts, timestamps, and health metrics for either legacy or v2 format operations.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        use_attribute_docstrings=True,
    )

    category: MODEL_REFERENCE_CATEGORY
    """The category of the model reference."""
    last_updated: int
    """Unix timestamp when metadata was last updated."""
    last_operation_type: OperationType | None = None
    """Type of the last operation performed."""
    last_operation_model: str | None = None
    """Name of the model affected by the last operation."""

    total_creates: int = Field(default=0, ge=0)
    """Total number of create operations."""
    total_updates: int = Field(default=0, ge=0)
    """Total number of update operations."""
    total_deletes: int = Field(default=0, ge=0)
    """Total number of delete operations."""
    total_models: int = Field(default=0, ge=0)
    """Current total number of models in category."""

    initialization_time: int
    """Unix timestamp when metadata was first created."""
    last_successful_operation: int
    """Unix timestamp of the last successful operation."""
    error_count: int = 0
    """Total number of errors encountered."""

    metadata_schema_version: str = "1.0.0"
    """Version of the metadata schema."""
    backend_type: str
    """Type of backend performing operations."""


class GenericModelRecordMetadata(BaseModel):
    """Metadata for a generic model record.

    This is imported from model_reference_records but defined here as a forward reference
    to avoid circular imports.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    schema_version: str = Field(default="1.0.0")
    created_at: int | None = None
    updated_at: int | None = None
    created_by: str | None = None
    updated_by: str | None = None


class ModelMetadataHandlerProtocol(Protocol):
    """Protocol for custom per-model metadata transformation handlers.

    Implementations can be injected into ModelMetadataManager to provide
    custom metadata transformation logic for specific use cases.
    """

    def transform_metadata(
        self,
        metadata: GenericModelRecordMetadata,
        context: dict[str, Any],
    ) -> GenericModelRecordMetadata:
        """Transform metadata with custom logic.

        Args:
            metadata: The metadata to transform.
            context: Additional context (e.g., model_name, category, operation_type).

        Returns:
            GenericModelRecordMetadata: Transformed metadata.
        """
        ...


class ModelMetadataManager:
    """Manages per-model metadata operations with strong typing.

    This class provides a typed interface for working with model metadata,
    eliminating the need for raw dictionary manipulation and magic strings.
    Supports dependency injection of custom handlers for specialized behavior.
    """

    def __init__(self, custom_handler: ModelMetadataHandlerProtocol | None = None) -> None:
        """Initialize the model metadata manager.

        Args:
            custom_handler: Optional custom handler for metadata transformation.
        """
        self._custom_handler = custom_handler

    def get_metadata(self, record_dict: dict[str, Any]) -> GenericModelRecordMetadata:
        """Get metadata from a model record dictionary.

        Args:
            record_dict: Model record as dictionary.

        Returns:
            GenericModelRecordMetadata: The metadata object, or a new empty one if not present.
        """
        metadata_dict = record_dict.get("metadata", {})
        return GenericModelRecordMetadata(**metadata_dict)

    def set_metadata(self, record_dict: dict[str, Any], metadata: GenericModelRecordMetadata) -> None:
        """Set metadata on a model record dictionary.

        Args:
            record_dict: Model record as dictionary.
            metadata: The metadata to set.
        """
        record_dict["metadata"] = metadata.model_dump(exclude_unset=True, mode="json")

    def update_metadata(
        self,
        record_dict: dict[str, Any],
        **updates: Any,  # noqa: ANN401
    ) -> GenericModelRecordMetadata:
        """Update specific metadata fields on a model record.

        Args:
            record_dict: Model record as dictionary.
            **updates: Field names and values to update.

        Returns:
            GenericModelRecordMetadata: The updated metadata object.
        """
        metadata = self.get_metadata(record_dict)
        for field, value in updates.items():
            if hasattr(metadata, field):
                setattr(metadata, field, value)
        self.set_metadata(record_dict, metadata)
        return metadata

    def preserve_creation_fields(
        self,
        existing_record: dict[str, Any],
        new_record: dict[str, Any],
    ) -> None:
        """Preserve creation metadata fields from existing record when updating.

        Args:
            existing_record: The existing model record with metadata to preserve.
            new_record: The new model record to update with preserved metadata.
        """
        existing_metadata = self.get_metadata(existing_record)
        new_metadata = self.get_metadata(new_record)

        if existing_metadata.created_at is not None:
            new_metadata.created_at = existing_metadata.created_at
        if existing_metadata.created_by is not None:
            new_metadata.created_by = existing_metadata.created_by

        self.set_metadata(new_record, new_metadata)

    def set_creation_timestamp(
        self,
        record_dict: dict[str, Any],
        timestamp: int | None = None,
    ) -> None:
        """Set creation timestamp on a model record.

        Args:
            record_dict: Model record as dictionary.
            timestamp: Unix timestamp to use, or None to use current time.
        """
        if timestamp is None:
            timestamp = int(time.time())
        self.update_metadata(record_dict, created_at=timestamp)

    def set_update_timestamp(
        self,
        record_dict: dict[str, Any],
        timestamp: int | None = None,
    ) -> None:
        """Set update timestamp on a model record.

        Args:
            record_dict: Model record as dictionary.
            timestamp: Unix timestamp to use, or None to use current time.
        """
        if timestamp is None:
            timestamp = int(time.time())
        self.update_metadata(record_dict, updated_at=timestamp)

    def ensure_metadata_populated(
        self,
        record_dict: dict[str, Any],
        timestamp: int | None = None,
    ) -> bool:
        """Ensure model record has all required metadata fields populated.

        Populates missing created_at and updated_at timestamps if they are None or missing.

        Args:
            record_dict: Model record as dictionary.
            timestamp: Unix timestamp to use for missing fields, or None to use current time.

        Returns:
            bool: True if any metadata fields were populated, False if all were already present.
        """
        if timestamp is None:
            timestamp = int(time.time())

        metadata = self.get_metadata(record_dict)
        updated = False

        if metadata.created_at is None:
            metadata.created_at = timestamp
            updated = True

        if metadata.updated_at is None:
            metadata.updated_at = timestamp
            updated = True

        if updated:
            self.set_metadata(record_dict, metadata)

        return updated

    def apply_custom_handler(
        self,
        record_dict: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Apply custom handler transformation if one is configured.

        Args:
            record_dict: Model record as dictionary.
            context: Additional context for the transformation.
        """
        if self._custom_handler is not None:
            metadata = self.get_metadata(record_dict)
            transformed = self._custom_handler.transform_metadata(metadata, context)
            self.set_metadata(record_dict, transformed)


class MetadataManager:
    """Centralized manager for backend metadata tracking.

    Handles metadata for both legacy (v1) and v2 format operations separately.
    Also provides access to per-model metadata operations via ModelMetadataManager.
    All metadata updates should go through this class to enable observability hooks.

    Thread-safe with RLock protection.
    """

    def __init__(
        self,
        base_path: Path,
        model_metadata_manager: ModelMetadataManager | None = None,
    ) -> None:
        """Initialize metadata manager.

        Args:
            base_path: Base path for horde model reference data.
            model_metadata_manager: Optional custom ModelMetadataManager for dependency injection.
                If None, a default instance will be created.
        """
        self._base_path = base_path
        self._lock = RLock()

        # Per-model metadata manager (supports dependency injection)
        self._model_metadata_manager = model_metadata_manager or ModelMetadataManager()

        # Separate caching for legacy and v2 metadata
        self._legacy_cache: dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata] = {}
        self._v2_cache: dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata] = {}

        # Cache timestamps for TTL
        self._legacy_cache_timestamps: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._v2_cache_timestamps: dict[MODEL_REFERENCE_CATEGORY, float] = {}

        # File modification times for cache invalidation
        self._legacy_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}
        self._v2_mtimes: dict[MODEL_REFERENCE_CATEGORY, float] = {}

        # Staleness tracking
        self._stale_legacy: set[MODEL_REFERENCE_CATEGORY] = set()
        self._stale_v2: set[MODEL_REFERENCE_CATEGORY] = set()

        # Cache TTL (default 60 seconds)
        self._cache_ttl_seconds: int = 60

        # Ensure meta directories exist
        self._ensure_meta_directories()

    @property
    def model_metadata(self) -> ModelMetadataManager:
        """Access point for per-model metadata operations.

        Returns:
            ModelMetadataManager: The model metadata manager instance.
        """
        return self._model_metadata_manager

    def _ensure_meta_directories(self) -> None:
        """Ensure meta/legacy and meta/v2 directories exist."""
        legacy_meta_path = self._base_path / "meta" / "legacy"
        v2_meta_path = self._base_path / "meta" / "v2"

        legacy_meta_path.mkdir(parents=True, exist_ok=True)
        v2_meta_path.mkdir(parents=True, exist_ok=True)

    def _get_legacy_metadata_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path:
        """Get file path for legacy metadata.

        Args:
            category: Model reference category

        Returns:
            Path to legacy metadata JSON file
        """
        return self._base_path / "meta" / "legacy" / f"{category.value}_metadata.json"

    def _get_v2_metadata_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path:
        """Get file path for v2 metadata.

        Args:
            category: Model reference category

        Returns:
            Path to v2 metadata JSON file
        """
        return self._base_path / "meta" / "v2" / f"{category.value}_metadata.json"

    def _read_metadata_file(self, file_path: Path) -> CategoryMetadata | None:
        """Read metadata from file.

        Args:
            file_path: Path to metadata file

        Returns:
            CategoryMetadata if file exists and is valid, None otherwise
        """
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)
            return CategoryMetadata(**data)
        except Exception as e:
            logger.error(f"Failed to read metadata from {file_path}: {e}")
            return None

    def _write_metadata_file(self, file_path: Path, metadata: CategoryMetadata) -> None:
        """Write metadata to file atomically.

        Uses temp file + rename pattern for atomic writes with backup/rollback on error.

        Args:
            file_path: Path to metadata file
            metadata: Metadata to write
        """
        temp_path = file_path.with_suffix(f".tmp.{time.time()}")
        backup_path = file_path.with_suffix(".bak")

        try:
            # Write to temp file
            with open(temp_path, "w") as f:
                json.dump(metadata.model_dump(mode="json"), f, indent=2)
                os.fsync(f.fileno())

            # Atomic rename with backup
            if file_path.exists():
                if backup_path.exists():
                    backup_path.unlink()
                file_path.replace(backup_path)

            temp_path.replace(file_path)

            # Clean up backup
            if backup_path.exists():
                backup_path.unlink()

        except Exception as e:
            logger.error(f"Failed to write metadata to {file_path}: {e}")
            # Rollback if needed
            if backup_path.exists() and not file_path.exists():
                backup_path.replace(file_path)
            raise
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def _is_cache_valid(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        cache_timestamps: dict[MODEL_REFERENCE_CATEGORY, float],
        mtimes: dict[MODEL_REFERENCE_CATEGORY, float],
        stale_set: set[MODEL_REFERENCE_CATEGORY],
        metadata_path_fn: Callable[[MODEL_REFERENCE_CATEGORY], Path],
    ) -> bool:
        """Check if cached metadata is valid.

        Args:
            category: Category to check
            cache_timestamps: Dict of cache timestamps
            mtimes: Dict of file modification times
            stale_set: Set of stale categories
            metadata_path_fn: Function to get metadata file path

        Returns:
            True if cache is valid, False otherwise
        """
        # Check explicit staleness
        if category in stale_set:
            return False

        # Check if cached
        if category not in cache_timestamps:
            return False

        # Check TTL
        cache_age = time.time() - cache_timestamps[category]
        if cache_age > self._cache_ttl_seconds:
            return False

        # Check file modification time
        metadata_path = metadata_path_fn(category)
        if metadata_path.exists():
            current_mtime = metadata_path.stat().st_mtime
            if category not in mtimes or mtimes[category] != current_mtime:
                return False

        return True

    def initialize_legacy_metadata(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        backend_type: str,
    ) -> CategoryMetadata:
        """Initialize new legacy metadata for a category.

        Args:
            category: Model reference category
            backend_type: Type of backend (FileSystemBackend/RedisBackend)

        Returns:
            Newly created CategoryMetadata
        """
        current_time = int(time.time())
        return CategoryMetadata(
            category=category,
            last_updated=current_time,
            initialization_time=current_time,
            last_successful_operation=current_time,
            backend_type=backend_type,
        )

    def initialize_v2_metadata(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        backend_type: str,
    ) -> CategoryMetadata:
        """Initialize new v2 metadata for a category.

        Args:
            category: Model reference category
            backend_type: Type of backend (FileSystemBackend/RedisBackend)

        Returns:
            Newly created CategoryMetadata
        """
        current_time = int(time.time())
        return CategoryMetadata(
            category=category,
            last_updated=current_time,
            initialization_time=current_time,
            last_successful_operation=current_time,
            backend_type=backend_type,
        )

    def get_or_initialize_v2_metadata(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Get v2 metadata for a category, initializing if it doesn't exist.

        This method is safe to call during initialization/seeding when metadata may not exist yet.
        It will create and persist new metadata if the file doesn't exist.

        Args:
            category: Model reference category
            backend_type: Type of backend (FileSystemBackend/RedisBackend)

        Returns:
            CategoryMetadata: The existing or newly created v2 metadata
        """
        with self._lock:
            metadata_path = self._get_v2_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                # Initialize new metadata
                metadata = self.initialize_v2_metadata(category, backend_type)
                self._write_metadata_file(metadata_path, metadata)
                logger.debug(f"Initialized v2 metadata for {category.value}")

                # Update cache
                self._v2_cache[category] = metadata
                self._v2_cache_timestamps[category] = time.time()
                self._v2_mtimes[category] = metadata_path.stat().st_mtime
                self._stale_v2.discard(category)
            else:
                # Update cache if valid check hasn't already cached it
                if not self._is_cache_valid(
                    category,
                    self._v2_cache_timestamps,
                    self._v2_mtimes,
                    self._stale_v2,
                    self._get_v2_metadata_path,
                ):
                    self._v2_cache[category] = metadata
                    self._v2_cache_timestamps[category] = time.time()
                    self._v2_mtimes[category] = metadata_path.stat().st_mtime
                    self._stale_v2.discard(category)

            return metadata

    def get_or_initialize_legacy_metadata(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Get legacy metadata for a category, initializing if it doesn't exist.

        This method is safe to call during initialization/seeding when metadata may not exist yet.
        It will create and persist new metadata if the file doesn't exist.

        Args:
            category: Model reference category
            backend_type: Type of backend (FileSystemBackend/RedisBackend)

        Returns:
            CategoryMetadata: The existing or newly created legacy metadata
        """
        with self._lock:
            metadata_path = self._get_legacy_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                # Initialize new metadata
                metadata = self.initialize_legacy_metadata(category, backend_type)
                self._write_metadata_file(metadata_path, metadata)
                logger.debug(f"Initialized legacy metadata for {category.value}")

                # Update cache
                self._legacy_cache[category] = metadata
                self._legacy_cache_timestamps[category] = time.time()
                self._legacy_mtimes[category] = metadata_path.stat().st_mtime
                self._stale_legacy.discard(category)
            else:
                # Update cache if valid check hasn't already cached it
                if not self._is_cache_valid(
                    category,
                    self._legacy_cache_timestamps,
                    self._legacy_mtimes,
                    self._stale_legacy,
                    self._get_legacy_metadata_path,
                ):
                    self._legacy_cache[category] = metadata
                    self._legacy_cache_timestamps[category] = time.time()
                    self._legacy_mtimes[category] = metadata_path.stat().st_mtime
                    self._stale_legacy.discard(category)

            return metadata

    def record_legacy_operation(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        operation: OperationType,
        model_name: str,
        success: bool = True,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Record a legacy format operation (observability hook point).

        This is the centralized method for tracking all v1/legacy operations.

        Args:
            category: Model reference category
            operation: Type of operation (create/update/delete)
            model_name: Name of model affected
            success: Whether operation was successful
            backend_type: Type of backend performing operation

        Returns:
            Updated CategoryMetadata
        """
        with self._lock:
            # Load existing metadata or initialize
            metadata_path = self._get_legacy_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                metadata = self.initialize_legacy_metadata(category, backend_type)

            # Update metadata
            current_time = int(time.time())
            metadata.last_updated = current_time
            metadata.last_operation_type = operation
            metadata.last_operation_model = model_name

            if success:
                metadata.last_successful_operation = current_time

                # Increment operation counters
                if operation == OperationType.CREATE:
                    metadata.total_creates += 1
                elif operation == OperationType.UPDATE:
                    metadata.total_updates += 1
                elif operation == OperationType.DELETE:
                    metadata.total_deletes += 1

            # Write to disk
            self._write_metadata_file(metadata_path, metadata)

            # Update cache
            self._legacy_cache[category] = metadata
            self._legacy_cache_timestamps[category] = time.time()
            self._legacy_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_legacy.discard(category)

            logger.debug(
                f"Recorded legacy {operation.value} operation for {category.value}/{model_name} "
                f"(creates={metadata.total_creates}, updates={metadata.total_updates}, "
                f"deletes={metadata.total_deletes})"
            )

            return metadata

    def record_v2_operation(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        operation: OperationType,
        model_name: str,
        success: bool = True,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Record a v2 format operation (observability hook point).

        This is the centralized method for tracking all v2 operations.

        Args:
            category: Model reference category
            operation: Type of operation (create/update/delete)
            model_name: Name of model affected
            success: Whether operation was successful
            backend_type: Type of backend performing operation

        Returns:
            Updated CategoryMetadata
        """
        with self._lock:
            # Load existing metadata or initialize
            metadata_path = self._get_v2_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                metadata = self.initialize_v2_metadata(category, backend_type)

            # Update metadata
            current_time = int(time.time())
            metadata.last_updated = current_time
            metadata.last_operation_type = operation
            metadata.last_operation_model = model_name

            if success:
                metadata.last_successful_operation = current_time

                # Increment operation counters
                if operation == OperationType.CREATE:
                    metadata.total_creates += 1
                elif operation == OperationType.UPDATE:
                    metadata.total_updates += 1
                elif operation == OperationType.DELETE:
                    metadata.total_deletes += 1

            # Write to disk
            self._write_metadata_file(metadata_path, metadata)

            # Update cache
            self._v2_cache[category] = metadata
            self._v2_cache_timestamps[category] = time.time()
            self._v2_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_v2.discard(category)

            logger.debug(
                f"Recorded v2 {operation.value} operation for {category.value}/{model_name} "
                f"(creates={metadata.total_creates}, updates={metadata.total_updates}, "
                f"deletes={metadata.total_deletes})"
            )

            return metadata

    def record_legacy_error(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        error_info: str,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Record a legacy format operation error (observability hook point).

        Args:
            category: Model reference category
            error_info: Error information
            backend_type: Type of backend

        Returns:
            Updated CategoryMetadata
        """
        with self._lock:
            metadata_path = self._get_legacy_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                metadata = self.initialize_legacy_metadata(category, backend_type)

            metadata.error_count += 1
            metadata.last_updated = int(time.time())

            self._write_metadata_file(metadata_path, metadata)

            # Update cache
            self._legacy_cache[category] = metadata
            self._legacy_cache_timestamps[category] = time.time()
            self._legacy_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_legacy.discard(category)

            logger.warning(f"Recorded legacy error for {category.value}: {error_info}")

            return metadata

    def record_v2_error(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        error_info: str,
        backend_type: str = "FileSystemBackend",
    ) -> CategoryMetadata:
        """Record a v2 format operation error (observability hook point).

        Args:
            category: Model reference category
            error_info: Error information
            backend_type: Type of backend

        Returns:
            Updated CategoryMetadata
        """
        with self._lock:
            metadata_path = self._get_v2_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                metadata = self.initialize_v2_metadata(category, backend_type)

            metadata.error_count += 1
            metadata.last_updated = int(time.time())

            self._write_metadata_file(metadata_path, metadata)

            # Update cache
            self._v2_cache[category] = metadata
            self._v2_cache_timestamps[category] = time.time()
            self._v2_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_v2.discard(category)

            logger.warning(f"Recorded v2 error for {category.value}: {error_info}")

            return metadata

    def get_legacy_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get legacy metadata for a category.

        Uses caching with TTL and mtime tracking.

        Args:
            category: Model reference category

        Returns:
            CategoryMetadata: The legacy metadata

        Raises:
            RuntimeError: If legacy metadata does not exist on disk
        """
        with self._lock:
            # Check cache validity
            if self._is_cache_valid(
                category,
                self._legacy_cache_timestamps,
                self._legacy_mtimes,
                self._stale_legacy,
                self._get_legacy_metadata_path,
            ):
                cached_metadata = self._legacy_cache.get(category)

                if cached_metadata is not None:
                    return cached_metadata

                raise RuntimeError("Inconsistent cache state: timestamp exists but no cached metadata")

            # Load from disk
            metadata_path = self._get_legacy_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                raise RuntimeError(f"Legacy metadata for category {category.value} does not exist on disk")

            # Update cache
            self._legacy_cache[category] = metadata
            self._legacy_cache_timestamps[category] = time.time()
            self._legacy_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_legacy.discard(category)

            return metadata

    def get_v2_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get v2 metadata for a category.

        Uses caching with TTL and mtime tracking.

        Args:
            category: Model reference category

        Returns:
            CategoryMetadata: The v2 metadata

        Raises:
            RuntimeError: If v2 metadata does not exist on disk
        """
        with self._lock:
            # Check cache validity
            if self._is_cache_valid(
                category,
                self._v2_cache_timestamps,
                self._v2_mtimes,
                self._stale_v2,
                self._get_v2_metadata_path,
            ):
                cached_metadata = self._v2_cache.get(category)

                if cached_metadata is not None:
                    return cached_metadata

                raise RuntimeError("Inconsistent cache state: timestamp exists but no cached metadata")

            # Load from disk
            metadata_path = self._get_v2_metadata_path(category)
            metadata = self._read_metadata_file(metadata_path)

            if metadata is None:
                raise RuntimeError(f"V2 metadata for category {category.value} does not exist on disk")

            # Update cache
            self._v2_cache[category] = metadata
            self._v2_cache_timestamps[category] = time.time()
            self._v2_mtimes[category] = metadata_path.stat().st_mtime
            self._stale_v2.discard(category)

            return metadata

    def get_all_legacy_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get all legacy metadata.

        Returns:
            Dict mapping categories to their legacy metadata
        """
        result = {}
        for category in MODEL_REFERENCE_CATEGORY:
            metadata = self.get_legacy_metadata(category)
            if metadata is not None:
                result[category] = metadata
        return result

    def get_all_v2_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get all v2 metadata.

        Returns:
            Dict mapping categories to their v2 metadata
        """
        result = {}
        for category in MODEL_REFERENCE_CATEGORY:
            metadata = self.get_v2_metadata(category)
            if metadata is not None:
                result[category] = metadata
        return result

    def mark_legacy_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark legacy metadata cache as stale.

        Args:
            category: Category to mark stale
        """
        with self._lock:
            self._stale_legacy.add(category)

    def mark_v2_stale(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """Mark v2 metadata cache as stale.

        Args:
            category: Category to mark stale
        """
        with self._lock:
            self._stale_v2.add(category)
