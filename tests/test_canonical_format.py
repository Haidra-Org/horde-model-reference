"""Tests for canonical_format functionality.

NOTE: These tests create backends and managers with explicit settings parameters
rather than relying on environment variables, as the settings singleton is created
at module import time and cannot be easily changed during test runs.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference import HordeModelReferenceSettings, ReplicateMode
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager


def create_test_legacy_model(name: str) -> dict[str, Any]:
    """Create a minimal test legacy model record."""
    return {
        "name": name,
        "description": f"Test model {name}",
        "version": "1.0",
        "config": {
            "files": [
                {"path": f"{name}.ckpt", "sha256sum": "abc123"},
            ],
            "download": [
                {"file_name": f"{name}.ckpt", "file_url": f"https://example.com/{name}.ckpt"},
            ],
        },
    }


class TestCanonicalFormatSetting:
    """Test the canonical_format setting."""

    def test_default_canonical_format_is_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default canonical_format is 'v2'."""
        monkeypatch.delenv("HORDE_MODEL_REFERENCE_CANONICAL_FORMAT", raising=False)
        settings = HordeModelReferenceSettings()
        assert settings.canonical_format == "v2"

    def test_canonical_format_accepts_legacy(self) -> None:
        """Test that canonical_format can be set to 'legacy' via constructor."""
        settings = HordeModelReferenceSettings(canonical_format="legacy")
        assert settings.canonical_format == "legacy"

    def test_canonical_format_validation_warns_legacy_in_replica(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test validation warning when canonical_format='legacy' in REPLICA mode."""
        settings = HordeModelReferenceSettings(
            canonical_format="legacy",
            replicate_mode=ReplicateMode.REPLICA,
        )
        assert settings.canonical_format == "legacy"
        assert "v1 API will be read-only" in caplog.text


class TestFileSystemBackendLegacyWrites:
    """Test FileSystemBackend legacy write support."""

    def test_supports_legacy_writes_false_by_default(
        self,
        primary_base: Path,
        v2_canonical_mode: None,
    ) -> None:
        """Test that supports_legacy_writes returns False with default settings (v2)."""
        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        assert backend.supports_legacy_writes() is False

    def test_update_model_legacy_creates_file(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that update_model_legacy creates a legacy format file."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        assert backend.supports_legacy_writes() is True

        test_model = create_test_legacy_model("test_model_1")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        backend.update_model_legacy(category, "test_model_1", test_model)

        legacy_file = legacy_path / "stable_diffusion.json"
        assert legacy_file.exists()

        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "test_model_1" in data
        assert data["test_model_1"]["name"] == "test_model_1"

    def test_update_model_legacy_fails_when_canonical_format_v2(
        self,
        primary_base: Path,
        v2_canonical_mode: None,
    ) -> None:
        """Test that update_model_legacy raises error when canonical_format='v2'."""
        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_model = create_test_legacy_model("test_model_1")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(RuntimeError, match="Legacy writes are only supported"):
            backend.update_model_legacy(category, "test_model_1", test_model)

    def test_delete_model_legacy_removes_model(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that delete_model_legacy removes a model from legacy file."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_model = create_test_legacy_model("test_model_to_delete")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        backend.update_model_legacy(category, "test_model_to_delete", test_model)

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "test_model_to_delete" in data

        backend.delete_model_legacy(category, "test_model_to_delete")

        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)
        assert "test_model_to_delete" not in data

    def test_delete_model_legacy_raises_key_error_if_not_found(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that delete_model_legacy raises KeyError if model not found."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        legacy_file = legacy_path / "stable_diffusion.json"
        legacy_file.write_text("{}")

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(KeyError, match="not found in legacy category"):
            backend.delete_model_legacy(category, "nonexistent_model")


class TestModelReferenceManagerLegacyWrites:
    """Test ModelReferenceManager legacy write methods."""

    def test_manager_update_model_legacy(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that ModelReferencemanager.backend.update_model_legacy works."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_model = create_test_legacy_model("manager_test_model")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        manager.backend.update_model_legacy(category, "manager_test_model", test_model)

        legacy_file = legacy_path / "stable_diffusion.json"
        assert legacy_file.exists()

        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "manager_test_model" in data

    def test_manager_delete_model_legacy(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that ModelReferencemanager.backend.delete_model_legacy works."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_model = create_test_legacy_model("model_to_delete")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        manager.backend.update_model_legacy(category, "model_to_delete", test_model)
        manager.backend.delete_model_legacy(category, "model_to_delete")

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "model_to_delete" not in data

    def test_manager_legacy_writes_fail_when_canonical_format_v2(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test that legacy write methods fail when canonical_format='v2'."""
        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_model = create_test_legacy_model("test_model")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(RuntimeError, match="Legacy writes are only"):
            manager.backend.update_model_legacy(category, "test_model", test_model)

        with pytest.raises(RuntimeError, match="Legacy writes are only"):
            manager.backend.delete_model_legacy(category, "test_model")


class TestLegacyConverterStubs:
    """Test that v2 → legacy converter stubs raise NotImplementedError."""

    def test_converter_convert_from_v2_to_legacy_not_implemented(
        self,
        primary_base: Path,
    ) -> None:
        """Test that convert_from_v2_to_legacy raises NotImplementedError."""
        from horde_model_reference.legacy.classes.legacy_converters import BaseLegacyConverter

        converter = BaseLegacyConverter(
            legacy_folder_path=primary_base / "legacy",
            target_file_folder=primary_base,
            model_reference_category=MODEL_REFERENCE_CATEGORY.image_generation,
        )

        with pytest.raises(NotImplementedError, match="v2 → legacy conversion is not yet implemented"):
            converter.convert_from_v2_to_legacy({})


class TestReplicaModeWriteRestrictions:
    """Test that REPLICA mode instances reject write operations."""

    def test_filesystem_backend_rejects_replica_mode(
        self,
        primary_base: Path,
    ) -> None:
        """Test that FileSystemBackend cannot be instantiated in REPLICA mode."""
        with pytest.raises(ValueError, match="FileSystemBackend can only be used in PRIMARY mode"):
            FileSystemBackend(
                base_path=primary_base,
                replicate_mode=ReplicateMode.REPLICA,
            )

    def test_replica_manager_rejects_legacy_writes(
        self,
        primary_base: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that REPLICA mode manager rejects update_model_legacy."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        test_model = create_test_legacy_model("test_model")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(NotImplementedError, match="does not support legacy write"):
            manager.backend.update_model_legacy(category, "test_model", test_model)

    def test_replica_manager_rejects_legacy_deletes(
        self,
        primary_base: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that REPLICA mode manager rejects delete_model_legacy."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(NotImplementedError, match="does not support legacy write"):
            manager.backend.delete_model_legacy(category, "test_model")

    def test_replica_manager_rejects_v2_writes(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Test that REPLICA mode manager rejects v2 update_model."""
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        test_record = GenericModelRecord(
            name="test_model",
            description="Test",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(NotImplementedError, match="does not support write operations"):
            manager.backend.update_model_from_base_model(category, "test_model", test_record)

    def test_replica_manager_rejects_v2_deletes(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Test that REPLICA mode manager rejects v2 delete_model."""
        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(NotImplementedError, match="does not support write"):
            manager.backend.delete_model(category, "test_model")


class TestV2WriteOperations:
    """Test v2 write operations and their restrictions."""

    def test_v2_update_model_succeeds_in_primary_mode(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test that v2 update_model works in PRIMARY mode with v2 format."""
        from horde_model_reference import horde_model_reference_settings
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        assert horde_model_reference_settings.canonical_format == "v2"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_record = GenericModelRecord(
            name="v2_test_model",
            description="V2 test",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        category = MODEL_REFERENCE_CATEGORY.image_generation

        manager.backend.update_model_from_base_model(category, "v2_test_model", test_record)

        models = manager.get_raw_model_reference_json(category)
        assert models is not None
        assert "v2_test_model" in models

    def test_v2_delete_model_succeeds_in_primary_mode(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test that v2 delete_model works in PRIMARY mode with v2 format."""
        from horde_model_reference import horde_model_reference_settings
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        assert horde_model_reference_settings.canonical_format == "v2"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_record = GenericModelRecord(
            name="v2_model_to_delete",
            description="Will be deleted",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        category = MODEL_REFERENCE_CATEGORY.image_generation

        manager.backend.update_model_from_base_model(category, "v2_model_to_delete", test_record)
        manager.backend.delete_model(category, "v2_model_to_delete")

        models = manager.get_raw_model_reference_json(category)
        assert models is None or "v2_model_to_delete" not in models

    def test_v2_delete_nonexistent_model_raises_error(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Test that deleting a non-existent v2 model raises FileNotFoundError."""
        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(FileNotFoundError):
            manager.backend.delete_model(category, "nonexistent_v2_model")


class TestCrossFormatWriteRestrictions:
    """Test that write operations respect canonical_format setting."""

    def test_v2_writes_unavailable_in_legacy_canonical_mode(
        self,
        primary_base: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that v2 write operations should be restricted when canonical_format='legacy'.

        While the backend technically supports writes in PRIMARY mode, the service layer
        should check canonical_format and only allow legacy writes when in legacy mode.
        """
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Backend in PRIMARY mode supports writes
        assert backend.supports_writes() is True
        # When canonical_format='legacy', legacy writes ARE supported
        assert backend.supports_legacy_writes() is True

        # Note: The backend itself doesn't enforce canonical_format restrictions.
        # It's the responsibility of the service/API layer to check canonical_format
        # and route to the appropriate write methods (v2 vs legacy).
        # This test documents that backend behavior - the service layer tests
        # in test_http_backend.py verify the actual API-level restrictions.

    def test_legacy_writes_unavailable_in_v2_canonical_mode(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test that legacy write operations are unavailable when canonical_format='v2'."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "v2"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        assert backend.supports_writes() is True
        assert backend.supports_legacy_writes() is False

        test_model = create_test_legacy_model("test_model")
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(RuntimeError, match="Legacy writes are only supported"):
            backend.update_model_legacy(category, "test_model", test_model)


class TestInvalidDataValidation:
    """Test validation of invalid model data."""

    def test_backend_delete_from_empty_legacy_file(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that deleting from an empty legacy file raises KeyError."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        legacy_file = legacy_path / "stable_diffusion.json"
        legacy_file.write_text("{}")

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(KeyError, match="not found in legacy category"):
            backend.delete_model_legacy(category, "nonexistent_model")

    def test_backend_delete_from_nonexistent_legacy_file(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that deleting from non-existent legacy file raises appropriate error."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(FileNotFoundError):
            backend.delete_model_legacy(category, "any_model")


class TestCanonicalFormatEdgeCases:
    """Test edge cases and boundary conditions for canonical_format."""

    def test_invalid_canonical_format_value(self) -> None:
        """Test that invalid canonical_format values are rejected."""
        with pytest.raises(ValueError):
            HordeModelReferenceSettings(canonical_format="invalid")

    def test_legacy_writes_with_multiple_models(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test legacy writes with multiple models in same file."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        for i in range(3):
            test_model = create_test_legacy_model(f"test_model_{i}")
            manager.backend.update_model_legacy(category, f"test_model_{i}", test_model)

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 3
        for i in range(3):
            assert f"test_model_{i}" in data

        manager.backend.delete_model_legacy(category, "test_model_1")

        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 2
        assert "test_model_0" in data
        assert "test_model_1" not in data
        assert "test_model_2" in data

    def test_update_same_legacy_model_multiple_times(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that updating the same legacy model multiple times works (upsert)."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        test_model = create_test_legacy_model("updatable_model")
        manager.backend.update_model_legacy(category, "updatable_model", test_model)

        test_model_updated = create_test_legacy_model("updatable_model")
        test_model_updated["description"] = "Updated description"
        test_model_updated["version"] = "2.0"
        manager.backend.update_model_legacy(category, "updatable_model", test_model_updated)

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["updatable_model"]["description"] == "Updated description"
        assert data["updatable_model"]["version"] == "2.0"

    def test_filesystem_backend_supports_writes_in_primary_mode(
        self,
        primary_base: Path,
    ) -> None:
        """Test that FileSystemBackend supports writes in PRIMARY mode."""
        primary_backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        assert primary_backend.supports_writes() is True

    def test_supports_legacy_writes_depends_on_format(
        self,
        primary_base: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that supports_legacy_writes depends on canonical_format setting."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        primary_backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        assert primary_backend.supports_legacy_writes() is True

    def test_supports_legacy_writes_false_in_v2_mode(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test that supports_legacy_writes returns False when canonical_format='v2'."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "v2"

        primary_backend = FileSystemBackend(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        assert primary_backend.supports_legacy_writes() is False


class TestV2DeleteEdgeCases:
    """Test edge cases for v2 delete operations."""

    def test_v2_delete_from_existing_file_but_missing_model(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test deleting a model that doesn't exist in an existing category file."""
        from horde_model_reference import horde_model_reference_settings
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        assert horde_model_reference_settings.canonical_format == "v2"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        test_record = GenericModelRecord(
            name="existing_model",
            description="Existing",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        category = MODEL_REFERENCE_CATEGORY.image_generation
        manager.backend.update_model_from_base_model(category, "existing_model", test_record)

        with pytest.raises(KeyError):
            manager.backend.delete_model(category, "nonexistent_model")


class TestLegacyDeleteEdgeCases:
    """Test edge cases for legacy delete operations."""

    def test_legacy_delete_from_file_with_other_models(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that deleting one model doesn't affect other models in the same file."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        models = ["model_a", "model_b", "model_c"]
        for model_name in models:
            test_model = create_test_legacy_model(model_name)
            manager.backend.update_model_legacy(category, model_name, test_model)

        manager.backend.delete_model_legacy(category, "model_b")

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        assert "model_a" in data
        assert "model_b" not in data
        assert "model_c" in data


class TestSettingsValidation:
    """Test validation of settings."""

    def test_invalid_replicate_mode_string(self) -> None:
        """Test that invalid replicate_mode strings are rejected."""
        with pytest.raises(ValueError):
            HordeModelReferenceSettings(replicate_mode="invalid_mode")


class TestManagerSingletonBehavior:
    """Test ModelReferenceManager singleton behavior with different modes."""

    def test_manager_singleton_with_replica_mode(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Test that manager properly handles REPLICA mode initialization."""
        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        assert not manager.backend.supports_writes(), "REPLICA manager should not support writes"
        assert not manager.backend.supports_legacy_writes(), "REPLICA manager should not support legacy writes"

    def test_manager_write_operations_check_backend_support(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Test that manager write operations check backend support before attempting writes."""
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.REPLICA,
        )

        test_record = GenericModelRecord(
            name="test_model",
            description="Test",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        category = MODEL_REFERENCE_CATEGORY.image_generation

        with pytest.raises(NotImplementedError, match="does not support write"):
            manager.backend.update_model_from_base_model(category, "test_model", test_record)


class TestConcurrentOperations:
    """Test behavior with concurrent or sequential operations."""

    def test_sequential_updates_and_deletes(
        self,
        primary_base: Path,
        restore_manager_singleton: None,
        v2_canonical_mode: None,
    ) -> None:
        """Test multiple sequential create/update/delete operations."""
        from horde_model_reference import horde_model_reference_settings
        from horde_model_reference.meta_consts import MODEL_DOMAIN, MODEL_PURPOSE, ModelClassification
        from horde_model_reference.model_reference_records import GenericModelRecord

        assert horde_model_reference_settings.canonical_format == "v2"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.image_generation

        test_record = GenericModelRecord(
            name="sequential_model",
            description="Sequential test",
            version="1.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        manager.backend.update_model_from_base_model(category, "sequential_model", test_record)

        test_record_updated = GenericModelRecord(
            name="sequential_model",
            description="Version 2",
            version="2.0",
            record_type=MODEL_REFERENCE_CATEGORY.image_generation,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
        )
        manager.backend.update_model_from_base_model(category, "sequential_model", test_record_updated)

        models = manager.get_raw_model_reference_json(category)
        assert models is not None
        assert models["sequential_model"]["description"] == "Version 2"
        assert models["sequential_model"]["version"] == "2.0"

        manager.backend.delete_model(category, "sequential_model")

        models = manager.get_raw_model_reference_json(category)
        assert models is None or "sequential_model" not in models


class TestNameValidation:
    """Test name validation in various scenarios."""

    def test_legacy_model_name_with_special_characters(
        self,
        primary_base: Path,
        legacy_path: Path,
        legacy_canonical_mode: None,
        restore_manager_singleton: None,
    ) -> None:
        """Test that legacy models can have names with special characters."""
        from horde_model_reference import horde_model_reference_settings

        assert horde_model_reference_settings.canonical_format == "legacy"

        manager = ModelReferenceManager(
            base_path=primary_base,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        special_names = ["model_with_underscores", "model-with-hyphens", "model.with.dots"]
        category = MODEL_REFERENCE_CATEGORY.image_generation

        for name in special_names:
            test_model = create_test_legacy_model(name)
            manager.backend.update_model_legacy(category, name, test_model)

        legacy_file = legacy_path / "stable_diffusion.json"
        with open(legacy_file, encoding="utf-8") as f:
            data = json.load(f)

        for name in special_names:
            assert name in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
