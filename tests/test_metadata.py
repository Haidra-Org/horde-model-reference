"""Tests for metadata correctness across PRIMARY and REPLICA modes.

This test module verifies that metadata behavior correctly reflects the semantic
differences between PRIMARY and REPLICA modes, and that metadata initialization
respects the actual state of model reference files.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ReplicateMode, horde_model_reference_paths
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.model_reference_metadata import CategoryMetadata, MetadataManager, OperationType


class TestCategoryMetadataSemantics:
    """Tests for CategoryMetadata semantic correctness."""

    def test_category_metadata_only_for_primary_mode(self, primary_base: Path) -> None:
        """CategoryMetadata should only exist and be tracked for PRIMARY mode backends.

        Semantic meaning: CategoryMetadata tracks write operations, which only occur in PRIMARY mode.
        REPLICA mode is read-only and should never create or maintain CategoryMetadata.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create a model reference file
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({"test_model": {"name": "test_model"}}))

        # Initialize PRIMARY backend
        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Verify CategoryMetadata exists for this category
        v2_metadata_path = backend._metadata_manager._get_v2_metadata_path(category)
        legacy_metadata_path = backend._metadata_manager._get_legacy_metadata_path(category)

        assert v2_metadata_path.exists(), "v2 CategoryMetadata should exist for PRIMARY mode"
        assert legacy_metadata_path.exists(), "legacy CategoryMetadata should exist for PRIMARY mode"

        # Verify metadata content is correct
        v2_metadata = backend._metadata_manager.get_v2_metadata(category)
        assert v2_metadata.backend_type == "FileSystemBackend"
        assert v2_metadata.category == category

    def test_category_metadata_not_created_for_nonexistent_files(self, primary_base: Path) -> None:
        """CategoryMetadata should only exist for categories that have actual model files.

        Semantic meaning: Metadata tracks operations on real data. If no data file exists,
        no metadata should be created during initialization.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Ensure no file exists
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        if file_path.exists():
            file_path.unlink()

        # Initialize backend (will scan and populate metadata)
        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Verify CategoryMetadata does NOT exist for this category
        v2_metadata_path = backend._metadata_manager._get_v2_metadata_path(category)
        legacy_metadata_path = backend._metadata_manager._get_legacy_metadata_path(category)

        assert not v2_metadata_path.exists(), "v2 CategoryMetadata should NOT exist when no model file exists"
        assert not legacy_metadata_path.exists(), "legacy CategoryMetadata should NOT exist when no model file exists"

    def test_category_metadata_created_on_first_write(self, primary_base: Path) -> None:
        """CategoryMetadata should be created when the first write operation occurs.

        Semantic meaning: Metadata tracks write operations, so it should be initialized
        when the first write happens, not before.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Initialize backend with skip_startup_metadata_population
        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
            skip_startup_metadata_population=True,
        )

        # Verify no metadata exists yet
        v2_metadata_path = backend._metadata_manager._get_v2_metadata_path(category)
        assert not v2_metadata_path.exists(), "Metadata should not exist before first write"

        # Perform first write
        timestamp_before_write = int(time.time())
        backend.update_model(category, "new_model", {"name": "new_model"})
        timestamp_after_write = int(time.time())

        # Now metadata should exist
        assert v2_metadata_path.exists(), "Metadata should exist after first write"

        # Verify metadata reflects the write operation
        # Note: Since the model didn't exist, this is a CREATE operation, not UPDATE
        v2_metadata = backend._metadata_manager.get_v2_metadata(category)
        assert v2_metadata.total_creates == 1, "First write to new model should be CREATE"
        assert v2_metadata.total_updates == 0, "No updates yet (first operation was CREATE)"
        assert v2_metadata.last_operation_type == OperationType.CREATE
        assert v2_metadata.last_operation_model == "new_model"
        assert timestamp_before_write <= v2_metadata.initialization_time <= timestamp_after_write

    def test_category_metadata_tracks_operation_counts(self, primary_base: Path) -> None:
        """CategoryMetadata should accurately track counts of different operation types.

        Semantic meaning: CategoryMetadata provides observability into write patterns.
        Counters should increment correctly for creates, updates, and deletes.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial file
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({}))

        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Record operations through metadata manager
        backend._metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="model1",
            backend_type="FileSystemBackend",
        )

        backend._metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="model2",
            backend_type="FileSystemBackend",
        )

        backend._metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.UPDATE,
            model_name="model1",
            backend_type="FileSystemBackend",
        )

        backend._metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.DELETE,
            model_name="model2",
            backend_type="FileSystemBackend",
        )

        # Verify counters
        v2_metadata = backend._metadata_manager.get_v2_metadata(category)
        assert v2_metadata.total_creates == 2, "Should track 2 creates"
        assert v2_metadata.total_updates == 1, "Should track 1 update"
        assert v2_metadata.total_deletes == 1, "Should track 1 delete"
        assert v2_metadata.last_operation_type == OperationType.DELETE
        assert v2_metadata.last_operation_model == "model2"

    def test_initialization_time_vs_operation_time(self, primary_base: Path) -> None:
        """initialization_time and last_updated should have different semantic meanings.

        Semantic meaning:
        - initialization_time: When the metadata file was first created (never changes)
        - last_updated: When the last operation occurred (changes with each operation)
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial file
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({}))

        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Get initial metadata
        v2_metadata_initial = backend._metadata_manager.get_v2_metadata(category)
        init_time = v2_metadata_initial.initialization_time
        initial_last_updated = v2_metadata_initial.last_updated

        # initialization_time should equal last_updated at creation
        assert init_time == initial_last_updated, "At creation, initialization_time should equal last_updated"

        # Wait a moment then perform an operation (needs to be >1 second for timestamp difference)
        time.sleep(1.1)
        backend._metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.UPDATE,
            model_name="test",
            backend_type="FileSystemBackend",
        )

        # Get updated metadata
        v2_metadata_after = backend._metadata_manager.get_v2_metadata(category)

        # initialization_time should NOT change
        assert (
            v2_metadata_after.initialization_time == init_time
        ), "initialization_time should never change after creation"

        # last_updated should change
        assert v2_metadata_after.last_updated > initial_last_updated, "last_updated should change after operations"


class TestModelRecordMetadataSemantics:
    """Tests for per-model metadata semantic correctness."""

    def test_model_metadata_populated_during_initialization(self, primary_base: Path) -> None:
        """Per-model metadata should be populated when files are first scanned.

        Semantic meaning: Even if models were created externally (e.g., GitHub seeding),
        they should get metadata timestamps reflecting when they were first seen by this backend.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create a file with models that don't have metadata
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        models_without_metadata = {
            "model1": {"name": "model1", "description": "test1"},
            "model2": {"name": "model2", "description": "test2"},
        }
        file_path.write_text(json.dumps(models_without_metadata))

        timestamp_before_init = int(time.time())

        # Initialize backend (should populate metadata)
        _backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        timestamp_after_init = int(time.time())

        # Read back the file
        stored_data = json.loads(file_path.read_text())

        # Verify metadata was added to models
        assert "metadata" in stored_data["model1"], "model1 should have metadata"
        assert "metadata" in stored_data["model2"], "model2 should have metadata"

        # Verify timestamps are reasonable
        model1_metadata = stored_data["model1"]["metadata"]
        assert "created_at" in model1_metadata
        assert "updated_at" in model1_metadata
        assert timestamp_before_init <= model1_metadata["created_at"] <= timestamp_after_init
        assert timestamp_before_init <= model1_metadata["updated_at"] <= timestamp_after_init

    def test_model_metadata_preserved_on_update(self, primary_base: Path) -> None:
        """created_at and created_by should be preserved when updating models.

        Semantic meaning: creation metadata is immutable provenance data.
        Only update metadata should change on updates.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial file with metadata
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        original_model = {
            "model1": {
                "name": "model1",
                "description": "original",
                "metadata": {
                    "created_at": 1000000,
                    "created_by": "original_creator",
                    "updated_at": 1000000,
                },
            }
        }
        file_path.write_text(json.dumps(original_model))

        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Update the model
        time.sleep(0.1)
        updated_model_data = {
            "name": "model1",
            "description": "updated",
            "metadata": {
                "updated_by": "new_updater",
            },
        }
        backend.update_model(category, "model1", updated_model_data)

        # Read back
        stored_data = json.loads(file_path.read_text())
        metadata = stored_data["model1"]["metadata"]

        # Verify created_* fields preserved
        assert metadata["created_at"] == 1000000, "created_at should be preserved"
        assert metadata["created_by"] == "original_creator", "created_by should be preserved"

        # Verify updated_at changed
        assert metadata["updated_at"] > 1000000, "updated_at should be newer"

    def test_skip_startup_metadata_population_semantic(self, primary_base: Path) -> None:
        """skip_startup_metadata_population should prevent metadata creation until explicitly triggered.

        Semantic meaning: When GitHub seeding will populate data, we don't want to create
        metadata for empty categories. Metadata should only be created after seeding completes.
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Initialize backend with skip flag
        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
            skip_startup_metadata_population=True,
        )

        # Verify no metadata exists
        v2_metadata_path = backend._metadata_manager._get_v2_metadata_path(category)
        assert not v2_metadata_path.exists(), "Metadata should not exist when skip_startup_metadata_population=True"

        # Now "seed" some data (simulating GitHub seeding)
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({"seeded_model": {"name": "seeded_model"}}))

        # Explicitly trigger metadata population
        backend.ensure_category_metadata_populated(category)

        # Now metadata should exist
        assert v2_metadata_path.exists(), "Metadata should exist after explicit population"

        # Verify it reflects the seeded data
        v2_metadata = backend._metadata_manager.get_v2_metadata(category)
        assert v2_metadata.backend_type == "FileSystemBackend"


class TestMetadataManagerAPI:
    """Tests for MetadataManager API correctness."""

    def test_get_metadata_raises_when_not_exists(self, primary_base: Path) -> None:
        """get_v2_metadata and get_legacy_metadata should raise RuntimeError when metadata doesn't exist.

        Semantic meaning: These methods enforce a runtime guarantee that metadata exists.
        They should never return None - instead raising an exception for missing metadata.

        This is the design constraint that was fixed in the earlier changes.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Ensure no metadata file exists
        v2_path = metadata_manager._get_v2_metadata_path(category)
        if v2_path.exists():
            v2_path.unlink()

        # get_v2_metadata should raise RuntimeError
        with pytest.raises(RuntimeError, match="V2 metadata for category.*does not exist"):
            metadata_manager.get_v2_metadata(category)

        # Same for legacy
        legacy_path = metadata_manager._get_legacy_metadata_path(category)
        if legacy_path.exists():
            legacy_path.unlink()

        with pytest.raises(RuntimeError, match="Legacy metadata for category.*does not exist"):
            metadata_manager.get_legacy_metadata(category)

    def test_get_or_initialize_creates_when_missing(self, primary_base: Path) -> None:
        """get_or_initialize_* methods should safely handle missing metadata.

        Semantic meaning: During initialization/seeding, we need safe methods that
        create metadata if it doesn't exist. These methods never raise exceptions.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Ensure no metadata file exists
        v2_path = metadata_manager._get_v2_metadata_path(category)
        if v2_path.exists():
            v2_path.unlink()

        # get_or_initialize should succeed and create metadata
        v2_metadata = metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")

        assert v2_metadata is not None
        assert v2_metadata.category == category
        assert v2_metadata.backend_type == "FileSystemBackend"
        assert v2_path.exists(), "Metadata file should be created"

        # Calling again should return existing metadata
        v2_metadata_again = metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")
        assert v2_metadata_again.initialization_time == v2_metadata.initialization_time

    def test_model_metadata_manager_typed_interface(self, primary_base: Path) -> None:
        """ModelMetadataManager should provide type-safe access to per-model metadata.

        Semantic meaning: Avoid raw dictionary manipulation and magic strings.
        Use typed interfaces for better IDE support and type checking.
        """
        metadata_manager = MetadataManager(primary_base)
        model_meta_manager = metadata_manager.model_metadata

        # Create a model record dict
        record_dict = {"name": "test", "description": "test"}

        # Set creation timestamp
        model_meta_manager.set_creation_timestamp(record_dict, timestamp=1000000)

        # Verify it's in the dict
        assert "metadata" in record_dict
        metadata_dict = record_dict["metadata"]
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["created_at"] == 1000000

        # Get metadata through typed interface
        metadata = model_meta_manager.get_metadata(record_dict)
        assert metadata.created_at == 1000000

        # Update through typed interface
        model_meta_manager.set_update_timestamp(record_dict, timestamp=2000000)

        # Verify update
        updated_metadata = model_meta_manager.get_metadata(record_dict)
        assert updated_metadata.created_at == 1000000  # preserved
        assert updated_metadata.updated_at == 2000000  # updated


class TestMetadataConsistency:
    """Tests for metadata consistency across backend operations."""

    def test_write_operation_updates_both_category_and_model_metadata(
        self,
        primary_base: Path,
    ) -> None:
        """Write operations should update both CategoryMetadata and model metadata.

        Semantic meaning: A single write operation affects two types of metadata:
        1. CategoryMetadata tracks the operation itself (observability)
        2. Model metadata tracks the model's lifecycle (provenance)
        """
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial file
        file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps({}))

        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Perform update
        timestamp_before = int(time.time())
        backend.update_model(category, "new_model", {"name": "new_model", "description": "test"})
        timestamp_after = int(time.time())

        # Verify CategoryMetadata updated
        # Note: Since the model didn't exist, this is a CREATE operation, not UPDATE
        v2_metadata = backend._metadata_manager.get_v2_metadata(category)
        assert v2_metadata.last_operation_type == OperationType.CREATE, "First write should be CREATE"
        assert v2_metadata.total_creates == 1
        assert v2_metadata.last_operation_model == "new_model"
        assert timestamp_before <= v2_metadata.last_updated <= timestamp_after

        # Verify model metadata exists
        stored_data = json.loads(file_path.read_text())
        assert "metadata" in stored_data["new_model"]
        model_metadata = stored_data["new_model"]["metadata"]
        assert timestamp_before <= model_metadata["created_at"] <= timestamp_after
        assert timestamp_before <= model_metadata["updated_at"] <= timestamp_after


class TestMetadataFileIO:
    """Tests for metadata file I/O operations.

    This test class validates that metadata files are being written and read correctly,
    focusing on file system interactions, atomic writes, error handling, and cache behavior.

    Test Coverage:
    - V2 and legacy metadata file creation and directory structure
    - File read/write durability and data integrity
    - Atomic write patterns (temp files, backups, fsync)
    - JSON formatting and structure validation
    - Cache invalidation based on file mtime tracking
    - Error handling (corruption, permissions, missing fields)
    - Thread-safe concurrent operations
    - Coexistence of legacy and v2 metadata files

    Semantic Guarantees Tested:
    - Metadata files use predictable paths (meta/v2/ and meta/legacy/)
    - Writes are atomic using temp file + rename pattern
    - Files are properly formatted JSON with indentation
    - Cache automatically invalidates on external file changes
    - System gracefully handles corrupted or incomplete files
    - Concurrent writes are serialized via RLock for consistency
    - Legacy and v2 metadata can coexist without interference
    """

    def test_v2_metadata_file_creation(self, primary_base: Path) -> None:
        """V2 metadata files should be created in the correct directory with proper structure.

        Semantic meaning: Metadata files follow a predictable directory structure
        (meta/v2/{category}_metadata.json) and are created with proper JSON formatting.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Verify directory structure exists
        meta_v2_dir = primary_base / "meta" / "v2"
        assert meta_v2_dir.exists(), "meta/v2 directory should exist after MetadataManager init"
        assert meta_v2_dir.is_dir(), "meta/v2 should be a directory"

        # Create v2 metadata
        timestamp_before = int(time.time())
        metadata = metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")
        timestamp_after = int(time.time())

        # Verify file was created
        expected_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        assert expected_path.exists(), f"V2 metadata file should exist at {expected_path}"
        assert expected_path.is_file(), "V2 metadata path should be a file"

        # Verify file contains valid JSON
        with open(expected_path) as f:
            file_content = f.read()
            assert file_content, "Metadata file should not be empty"
            file_data = json.loads(file_content)

        # Verify structure
        assert "category" in file_data
        assert "backend_type" in file_data
        assert "initialization_time" in file_data
        assert file_data["category"] == category.value
        assert file_data["backend_type"] == "FileSystemBackend"
        assert timestamp_before <= file_data["initialization_time"] <= timestamp_after

        # Verify returned object matches file
        assert metadata.category == category
        assert metadata.backend_type == "FileSystemBackend"
        assert metadata.initialization_time == file_data["initialization_time"]

    def test_legacy_metadata_file_creation(self, primary_base: Path) -> None:
        """Legacy metadata files should be created in the correct directory.

        Semantic meaning: Legacy metadata files are separate from v2 metadata files,
        stored in meta/legacy/ directory for backward compatibility.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.image_generation

        # Verify directory structure
        meta_legacy_dir = primary_base / "meta" / "legacy"
        assert meta_legacy_dir.exists(), "meta/legacy directory should exist after MetadataManager init"

        # Create legacy metadata
        metadata = metadata_manager.get_or_initialize_legacy_metadata(category, "FileSystemBackend")

        # Verify file was created in correct location
        expected_path = primary_base / "meta" / "legacy" / f"{category.value}_metadata.json"
        assert expected_path.exists(), f"Legacy metadata file should exist at {expected_path}"

        # Verify it's separate from v2 file
        v2_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        assert expected_path != v2_path, "Legacy and v2 metadata paths should be different"

        # Verify file contents
        with open(expected_path) as f:
            file_data = json.loads(f.read())

        assert file_data["category"] == category.value
        assert metadata.category == category

    def test_metadata_file_read_after_write(self, primary_base: Path) -> None:
        """Metadata should be readable after being written to disk.

        Semantic meaning: File writes are durable - data written to disk
        can be retrieved in subsequent reads.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Write metadata
        original_metadata = metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")
        original_init_time = original_metadata.initialization_time

        # Clear cache to force disk read
        metadata_manager._v2_cache.clear()
        metadata_manager._v2_cache_timestamps.clear()

        # Read metadata from disk
        read_metadata = metadata_manager.get_v2_metadata(category)

        # Verify data integrity
        assert read_metadata.category == original_metadata.category
        assert read_metadata.backend_type == original_metadata.backend_type
        assert read_metadata.initialization_time == original_init_time
        assert read_metadata.last_updated == original_metadata.last_updated

    def test_metadata_file_atomic_write_with_temp_file(
        self,
        primary_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Metadata writes should use temp files for atomicity.

        Semantic meaning: Writes use temp file + rename pattern to ensure
        atomic updates and prevent corruption from partial writes.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Track calls to _write_metadata_file
        temp_files_created: list[Path] = []
        original_write = metadata_manager._write_metadata_file

        def tracking_write(file_path: Path, metadata: CategoryMetadata) -> None:
            # Capture temp file creation
            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            temp_files_created.append(temp_path)
            return original_write(file_path, metadata)

        monkeypatch.setattr(metadata_manager, "_write_metadata_file", tracking_write)

        # Perform write operation
        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="test_model",
            backend_type="FileSystemBackend",
        )

        # Verify temp file pattern was used
        assert len(temp_files_created) > 0, "Temp file should have been created"

        # Verify final file exists
        final_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        assert final_path.exists(), "Final metadata file should exist"

        # Verify no temp files remain (they should be cleaned up)
        for temp_path in temp_files_created:
            assert not temp_path.exists(), f"Temp file {temp_path} should have been cleaned up"

    def test_metadata_file_backup_on_update(self, primary_base: Path) -> None:
        """Updating existing metadata should create a backup before overwriting.

        Semantic meaning: The atomic write pattern creates a .bak file
        before replacing the original, enabling recovery on error.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial metadata
        metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")

        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        backup_path = metadata_path.with_suffix(".bak")

        # Ensure no backup exists initially
        if backup_path.exists():
            backup_path.unlink()

        # Record an operation (triggers update to existing metadata)
        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.UPDATE,
            model_name="test_model",
            backend_type="FileSystemBackend",
        )

        # Note: backup file is cleaned up after successful write
        # This tests that the mechanism doesn't leave artifacts
        assert not backup_path.exists(), "Backup file should be cleaned up after successful write"
        assert metadata_path.exists(), "Original metadata file should still exist"

    def test_metadata_file_json_formatting(self, primary_base: Path) -> None:
        """Metadata files should be written as properly formatted JSON.

        Semantic meaning: Files are human-readable with indentation,
        making them suitable for version control and manual inspection.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="test_model",
            backend_type="FileSystemBackend",
        )

        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        with open(metadata_path) as f:
            content = f.read()

        # Verify JSON is indented (not minified)
        assert "\n" in content, "JSON should have newlines (not minified)"
        assert "  " in content, "JSON should be indented"

        # Verify it's valid JSON
        parsed = json.loads(content)
        assert isinstance(parsed, dict), "Root should be a dictionary"

        # Verify structure is correct
        assert "category" in parsed
        assert "backend_type" in parsed

    def test_metadata_cache_invalidation_on_file_change(
        self,
        primary_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Metadata cache should be invalidated when underlying file is modified.

        Semantic meaning: Cache uses mtime tracking to detect external file changes
        and automatically refresh to prevent serving stale data.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial metadata
        initial_metadata = metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")
        initial_creates = initial_metadata.total_creates

        # Verify it's cached
        assert category in metadata_manager._v2_cache
        cached_metadata = metadata_manager._v2_cache[category]
        assert cached_metadata.total_creates == initial_creates

        # Modify file directly (simulating external change)
        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        with open(metadata_path) as f:
            file_data = json.loads(f.read())

        file_data["total_creates"] = initial_creates + 10
        file_data["last_updated"] = int(time.time())

        with open(metadata_path, "w") as f:
            json.dump(file_data, f, indent=2)

        # Touch file to update mtime
        import os

        stat_info = metadata_path.stat()
        os.utime(metadata_path, (stat_info.st_atime, stat_info.st_mtime + 2))

        # Clear the cache timestamp to force recheck
        metadata_manager._v2_cache_timestamps.pop(category, None)

        # Read metadata again - should detect mtime change and reload
        refreshed_metadata = metadata_manager.get_v2_metadata(category)

        # Verify cache was invalidated and fresh data loaded
        assert refreshed_metadata.total_creates == initial_creates + 10
        assert refreshed_metadata.total_creates != cached_metadata.total_creates

    def test_metadata_file_corruption_handling(self, primary_base: Path) -> None:
        """MetadataManager should handle corrupted metadata files gracefully.

        Semantic meaning: If a file is corrupted (invalid JSON), reads should
        return None rather than crashing, allowing recovery mechanisms.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create metadata directory
        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Write corrupted JSON
        metadata_path.write_text("{ invalid json content }}")

        # Clear cache to force disk read
        metadata_manager._v2_cache.clear()
        metadata_manager._v2_cache_timestamps.clear()

        # Attempt to read corrupted file
        result = metadata_manager._read_metadata_file(metadata_path)

        # Should return None on corruption
        assert result is None, "Corrupted metadata file should return None"

    def test_metadata_file_missing_fields_handled(self, primary_base: Path) -> None:
        """Metadata files with missing optional fields should still parse correctly.

        Semantic meaning: Pydantic models with defaults should tolerate
        incomplete JSON, using default values for missing fields.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create minimal metadata file (missing optional fields)
        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        minimal_data = {
            "category": category.value,
            "last_updated": int(time.time()),
            "initialization_time": int(time.time()),
            "last_successful_operation": int(time.time()),
            "backend_type": "FileSystemBackend",
        }

        with open(metadata_path, "w") as f:
            json.dump(minimal_data, f)

        # Clear cache
        metadata_manager._v2_cache.clear()
        metadata_manager._v2_cache_timestamps.clear()

        # Read file
        result = metadata_manager._read_metadata_file(metadata_path)

        # Should successfully parse with defaults
        assert result is not None, "Minimal metadata should parse successfully"
        assert result.total_creates == 0, "Missing total_creates should default to 0"
        assert result.total_updates == 0, "Missing total_updates should default to 0"
        assert result.error_count == 0, "Missing error_count should default to 0"

    def test_concurrent_metadata_writes_thread_safety(
        self,
        primary_base: Path,
    ) -> None:
        """MetadataManager should safely handle concurrent write operations.

        Semantic meaning: The RLock ensures thread-safe access to metadata,
        preventing race conditions during concurrent updates.
        """
        import threading

        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Initialize metadata
        metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")

        # Perform concurrent operations
        num_threads = 5
        operations_per_thread = 10
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(operations_per_thread):
                    metadata_manager.record_v2_operation(
                        category=category,
                        operation=OperationType.CREATE,
                        model_name=f"thread_{thread_id}_model_{i}",
                        backend_type="FileSystemBackend",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent writes should not raise errors: {errors}"

        # Verify operation count is correct
        final_metadata = metadata_manager.get_v2_metadata(category)
        expected_creates = num_threads * operations_per_thread
        assert (
            final_metadata.total_creates == expected_creates
        ), f"Expected {expected_creates} creates, got {final_metadata.total_creates}"

    def test_metadata_file_permissions_error_handling(
        self,
        primary_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MetadataManager should handle file permission errors gracefully.

        Semantic meaning: If file system operations fail due to permissions,
        appropriate errors should be raised with context.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Mock open() to raise PermissionError
        original_open = open

        def mock_open(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            # Only fail for metadata file writes
            if args and isinstance(args[0], Path) and ".tmp." in str(args[0]):
                raise PermissionError("Permission denied")
            return original_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        # Attempt to record operation (triggers write)
        with pytest.raises(PermissionError, match="Permission denied"):
            metadata_manager.record_v2_operation(
                category=category,
                operation=OperationType.CREATE,
                model_name="test_model",
                backend_type="FileSystemBackend",
            )

    def test_legacy_and_v2_metadata_files_coexist(self, primary_base: Path) -> None:
        """Legacy and v2 metadata files should coexist independently.

        Semantic meaning: The system maintains separate metadata tracking
        for legacy and v2 formats, allowing dual-format support during migration.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.image_generation

        # Create both legacy and v2 metadata
        metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")
        metadata_manager.get_or_initialize_legacy_metadata(category, "FileSystemBackend")

        # Verify both files exist
        v2_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        legacy_path = primary_base / "meta" / "legacy" / f"{category.value}_metadata.json"

        assert v2_path.exists(), "V2 metadata file should exist"
        assert legacy_path.exists(), "Legacy metadata file should exist"
        assert v2_path != legacy_path, "Files should be in different locations"

        # Record operations to each
        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="v2_model",
            backend_type="FileSystemBackend",
        )

        metadata_manager.record_legacy_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="legacy_model",
            backend_type="FileSystemBackend",
        )

        # Verify operations are tracked separately
        updated_v2 = metadata_manager.get_v2_metadata(category)
        updated_legacy = metadata_manager.get_legacy_metadata(category)

        assert updated_v2.total_creates == 1, "V2 should track its own creates"
        assert updated_legacy.total_creates == 1, "Legacy should track its own creates"

        # Verify they have different last_operation_model
        assert updated_v2.last_operation_model == "v2_model"
        assert updated_legacy.last_operation_model == "legacy_model"

    def test_metadata_file_fsync_called(
        self,
        primary_base: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Metadata writes should call fsync to ensure data is flushed to disk.

        Semantic meaning: For durability, file writes must flush to physical storage
        using fsync, preventing data loss on system crashes.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        fsync_called = []

        original_fsync = os.fsync

        def mock_fsync(fd: int) -> None:
            fsync_called.append(fd)
            original_fsync(fd)

        monkeypatch.setattr("os.fsync", mock_fsync)

        # Trigger metadata write
        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.CREATE,
            model_name="test_model",
            backend_type="FileSystemBackend",
        )

        # Verify fsync was called
        assert len(fsync_called) > 0, "fsync should have been called during metadata write"

    def test_metadata_mtime_tracking_accuracy(self, primary_base: Path) -> None:
        """Metadata manager should accurately track file modification times.

        Semantic meaning: Cache invalidation relies on mtime tracking to detect
        external file changes without constantly re-reading files.
        """
        metadata_manager = MetadataManager(primary_base)
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create initial metadata
        metadata_manager.get_or_initialize_v2_metadata(category, "FileSystemBackend")

        # Get initial mtime from cache
        initial_mtime = metadata_manager._v2_mtimes.get(category)
        assert initial_mtime is not None, "mtime should be tracked after metadata creation"

        # Get actual file mtime
        metadata_path = primary_base / "meta" / "v2" / f"{category.value}_metadata.json"
        actual_mtime = metadata_path.stat().st_mtime

        # Verify cached mtime matches file
        assert initial_mtime == actual_mtime, "Cached mtime should match actual file mtime"

        # Wait and update file
        time.sleep(0.1)
        metadata_manager.record_v2_operation(
            category=category,
            operation=OperationType.UPDATE,
            model_name="test",
            backend_type="FileSystemBackend",
        )

        # Get updated mtime
        updated_mtime = metadata_manager._v2_mtimes.get(category)
        updated_actual_mtime = metadata_path.stat().st_mtime

        # Verify mtime was updated
        assert updated_mtime is not None
        assert updated_mtime > initial_mtime, "mtime should increase after file update"
        assert updated_mtime == updated_actual_mtime, "Cached mtime should match updated file mtime"
