from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger
from pydantic import ValidationError
from pytest import LogCaptureFixture

from horde_model_reference import ReplicateMode
from horde_model_reference.backends.base import ModelReferenceBackend
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.meta_consts import (
    MODEL_CLASSIFICATION_LOOKUP,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import GenericModelRecord


class _InMemoryReplicaBackend(ModelReferenceBackend):
    """Lightweight replica backend stub to record async fetch invocations."""

    def __init__(self) -> None:
        super().__init__(mode=ReplicateMode.REPLICA)
        self._stale_categories: set[MODEL_REFERENCE_CATEGORY] = set()
        self._data: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {
            category: {} for category in MODEL_REFERENCE_CATEGORY
        }
        self.async_calls: list[dict[str, Any]] = []

    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        if force_refresh:
            self._stale_categories.discard(category)
        return self._data[category]

    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        if force_refresh:
            self._stale_categories.clear()
        return self._data

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        return self.fetch_category(category, force_refresh=force_refresh)

    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        self.async_calls.append(
            {
                "httpx_client": httpx_client,
                "force_refresh": force_refresh,
            }
        )
        return self.fetch_all_categories(force_refresh=force_refresh)

    def needs_refresh(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        return category in self._stale_categories

    def _mark_stale_impl(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        self._stale_categories.add(category)

    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        return None

    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        return dict.fromkeys(MODEL_REFERENCE_CATEGORY)

    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        return None

    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        return None


def test_manager(
    caplog: LogCaptureFixture,
    restore_manager_singleton: None,
) -> None:
    """Test that the manager can fetch and cache all model reference categories via the backend.

    Verifies:
    - All category file paths are available from the backend
    - Backend can fetch all categories with force refresh
    - Manager returns properly structured model references
    - Data quality of fetched references
    """
    from tests.helpers import verify_model_references_structure

    model_reference_manager = ModelReferenceManager(replicate_mode=ReplicateMode.REPLICA, lazy_mode=True)

    legacy_reference_locations = model_reference_manager.backend.get_all_category_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations
    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    caplog.clear()
    model_reference_manager.backend.fetch_all_categories(force_refresh=True)

    log_messages = [record.message for record in caplog.records]
    assert any("Loaded converted JSON" in msg or "loading from disk" in msg for msg in log_messages), (
        "Expected 'Loaded converted JSON' or 'loading from disk' in log messages during cache loading"
    )

    all_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]]
    all_references = model_reference_manager.get_all_model_references(overwrite_existing=False)

    assert len(all_references) > 0
    assert all(ref_cat in all_references for ref_cat in MODEL_REFERENCE_CATEGORY)
    assert all(ref is not None for ref in all_references.values())

    verify_model_references_structure(all_references)

    cached_references = model_reference_manager.get_all_model_references_unsafe(overwrite_existing=False)
    assert len(cached_references) == len(all_references)


@pytest.mark.asyncio
async def test_manager_async(
    model_reference_manager: ModelReferenceManager,
    caplog: LogCaptureFixture,
) -> None:
    """Test that the manager can fetch all categories asynchronously via the backend.

    Verifies:
    - All category file paths are available from the backend
    - Backend can fetch all categories asynchronously with force refresh
    - Manager returns properly structured model references after async fetch
    - Data quality of fetched references
    """
    from tests.helpers import verify_model_references_structure

    legacy_reference_locations = model_reference_manager.backend.get_all_category_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations
    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    httpx_client = httpx.AsyncClient()

    async with httpx_client:
        await model_reference_manager.backend.fetch_all_categories_async(
            httpx_client=httpx_client,
            force_refresh=True,
        )

    all_references = model_reference_manager.get_all_model_references(overwrite_existing=False)

    assert len(all_references) > 0
    assert all(ref_cat in all_references for ref_cat in MODEL_REFERENCE_CATEGORY)

    verify_model_references_structure(all_references)

    cached_references = model_reference_manager.get_all_model_references_unsafe(overwrite_existing=False)
    assert len(cached_references) == len(all_references)


def test_manager_new_format(
    model_reference_manager: ModelReferenceManager,
    caplog: LogCaptureFixture,
    restore_manager_singleton: None,
) -> None:
    """Test the new format model reference manager."""
    ModelReferenceManager(replicate_mode=ReplicateMode.REPLICA, lazy_mode=False)

    def assert_all_model_references_exist(
        model_reference_manager: ModelReferenceManager,
        overwrite_existing: bool,
    ) -> None:
        """Assert that all model references exist."""
        all_model_references = model_reference_manager.get_all_model_references_unsafe(
            overwrite_existing=overwrite_existing
        )
        for model_reference_category in MODEL_REFERENCE_CATEGORY:
            if model_reference_category == MODEL_REFERENCE_CATEGORY.text_generation:
                logger.warning("Skipping text generation model references, they are not yet implemented.")
                continue
            if model_reference_category in [
                MODEL_REFERENCE_CATEGORY.video_generation,
                MODEL_REFERENCE_CATEGORY.audio_generation,
            ]:
                logger.info(
                    f"Skipping {model_reference_category} - no legacy format available, "
                    "empty file created during initialization."
                )
                continue

            assert model_reference_category in all_model_references, (
                f"Model reference category {model_reference_category} is missing"
            )

            model_reference_instance = all_model_references[model_reference_category]

            # Allow None or empty dict for categories without legacy data
            if model_reference_instance is None:
                logger.warning(
                    f"Model reference instance for {model_reference_category} is None - "
                    "this may occur in CI environments where files haven't been seeded yet"
                )
                continue

            if len(model_reference_instance) == 0:
                logger.info(f"Model reference instance for {model_reference_category} is empty - skipping validation")
                continue

            assert model_reference_category in MODEL_CLASSIFICATION_LOOKUP, (
                f"Model reference category {model_reference_category} is not in the classification lookup"
            )

            for _, model_entry in model_reference_instance.items():
                assert model_entry.model_classification == MODEL_CLASSIFICATION_LOOKUP[model_reference_category], (
                    f"Model entry for {model_reference_category} is not classified correctly"
                )

    assert_all_model_references_exist(model_reference_manager, overwrite_existing=True)

    assert_all_model_references_exist(model_reference_manager, overwrite_existing=False)


class TestSingleton:
    """Test the singleton behavior of ModelReferenceManager."""

    def test_singleton_behavior(self) -> None:
        """Test that ModelReferenceManager is a singleton."""
        mgr1 = ModelReferenceManager()
        mgr2 = ModelReferenceManager()
        assert mgr1 is mgr2, "ModelReferenceManager should be a singleton"

    def test_singleton_rejects_different_lazy_mode(self, restore_manager_singleton: None) -> None:
        """Test that singleton rejects re-instantiation with different lazy_mode."""
        ModelReferenceManager(lazy_mode=True)

        with pytest.raises(RuntimeError, match="different settings"):
            ModelReferenceManager(lazy_mode=False)

    def test_singleton_rejects_different_replicate_mode(self, restore_manager_singleton: None) -> None:
        """Test that singleton rejects re-instantiation with different replicate_mode."""
        ModelReferenceManager(replicate_mode=ReplicateMode.REPLICA)

        with pytest.raises(RuntimeError, match="different settings"):
            ModelReferenceManager(replicate_mode=ReplicateMode.PRIMARY)

    def test_singleton_rejects_different_backend(self, tmp_path: Path, restore_manager_singleton: None) -> None:
        """Test that singleton rejects re-instantiation with different backend instance."""
        backend1 = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        backend2 = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)

        ModelReferenceManager(
            backend=backend1,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        with pytest.raises(RuntimeError, match="different backend"):
            ModelReferenceManager(
                backend=backend2,
                lazy_mode=True,
                replicate_mode=ReplicateMode.PRIMARY,
            )


@pytest.mark.usefixtures("restore_manager_singleton")
class TestCacheAndStaleness:
    """Test cache invalidation and staleness handling in ModelReferenceManager."""

    def test_invalidate_cache(self) -> None:
        """Test that the cache invalidation works."""
        manager = ModelReferenceManager(lazy_mode=True)

        manager.get_all_model_references_unsafe(overwrite_existing=False)
        assert manager._cached_records
        manager._invalidate_cache()
        assert manager._cached_records == {}

    def test_selective_cache_invalidation(self) -> None:
        """Test that cache can be selectively invalidated by category."""
        manager = ModelReferenceManager(lazy_mode=True)

        manager.get_all_model_references_unsafe(overwrite_existing=False)
        initial_cache_size = len(manager._cached_records)
        assert initial_cache_size > 0, "Cache should be populated for this test"

        category_to_invalidate = MODEL_REFERENCE_CATEGORY.miscellaneous
        manager._invalidate_cache(category=category_to_invalidate)

        assert category_to_invalidate not in manager._cached_records, "Expected category to be invalidated"

        assert len(manager._cached_records) == initial_cache_size - 1, "Expected one category to be invalidated"

        manager._invalidate_cache()
        assert len(manager._cached_records) == 0, "Expected full cache invalidation"

    def test_backend_invalidation_triggers_manager_cache_clear(self, tmp_path: Path) -> None:
        """Test that backend invalidation callbacks properly clear manager's pydantic cache.

        Verifies that when a backend marks a category as stale, the manager's
        pydantic model cache is cleared via the registered callback.
        """
        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        # Create a test file
        import json

        test_data = {
            "model1": {
                "name": "model1",
                "model_classification": {
                    "domain": MODEL_DOMAIN.image.value,
                    "purpose": MODEL_PURPOSE.miscellaneous.value,
                },
            }
        }

        file_path = backend.get_category_file_path(category)
        assert file_path is not None
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(test_data))

        # Load data into manager cache
        _ = manager.get_all_model_references_unsafe()
        assert category in manager._cached_records

        # Mark backend as stale - this should trigger callback and clear manager cache
        backend.mark_stale(category)

        # Verify manager cache was cleared for this category
        assert category not in manager._cached_records

    def test_lazy_mode_false_fetches_immediately(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that lazy_mode=False triggers immediate fetch on initialization."""
        backend = _InMemoryReplicaBackend()
        fetch_called = {"count": 0}

        original_fetch = backend.fetch_all_categories

        def tracking_fetch(
            force_refresh: bool = False,
        ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
            fetch_called["count"] += 1
            return original_fetch(force_refresh=force_refresh)

        backend.fetch_all_categories = tracking_fetch  # type: ignore[method-assign]

        _manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=False,
            replicate_mode=ReplicateMode.REPLICA,
        )

        assert fetch_called["count"] == 1

    def test_lazy_mode_true_defers_fetch(self, tmp_path: Path) -> None:
        """Test that lazy_mode=True defers fetch until first access."""
        backend = _InMemoryReplicaBackend()

        backend.mark_stale(MODEL_REFERENCE_CATEGORY.miscellaneous)

        fetch_called = {"count": 0}

        original_fetch = backend.fetch_all_categories

        def tracking_fetch(
            force_refresh: bool = False,
        ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
            fetch_called["count"] += 1
            return original_fetch(force_refresh=force_refresh)

        backend.fetch_all_categories = tracking_fetch  # type: ignore[method-assign]

        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.REPLICA,
        )

        assert fetch_called["count"] == 0

        manager.get_all_model_references_unsafe()
        assert fetch_called["count"] == 1


class TestBackendConfiguration:
    """Test various backend configurations for ModelReferenceManager."""

    def test_primary_mode_uses_filesystem_backend(
        self,
        tmp_path: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Ensure PRIMARY mode instantiates a FileSystemBackend."""
        manager = ModelReferenceManager(
            base_path=tmp_path,
            replicate_mode=ReplicateMode.PRIMARY,
            lazy_mode=True,
        )
        assert isinstance(manager.backend, FileSystemBackend), "Expected FileSystemBackend for PRIMARY mode"
        assert manager.backend.replicate_mode == ReplicateMode.PRIMARY
        assert manager.backend.supports_writes() is True, "PRIMARY mode backend should support writes"

    def test_backend_mode_mismatch_raises(
        self,
        tmp_path: Path,
        restore_manager_singleton: None,
    ) -> None:
        """Verify mismatched backend/manager replicate modes fail fast."""
        backend = FileSystemBackend(base_path=tmp_path)
        with pytest.raises(RuntimeError, match="Backend replicate_mode"):
            ModelReferenceManager(
                backend=backend,
                replicate_mode=ReplicateMode.REPLICA,
            )


class TestModelReferenceConversion:
    """Test conversion between dict (JSON) and model reference structures."""

    def test_file_json_dict_to_model_reference_valid(self, model_reference_manager: ModelReferenceManager) -> None:
        """Test valid conversion from a dict (parsed JSON) to model reference."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_classification = ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.miscellaneous,
        )
        file_json_dict = {
            "test_model": {
                "name": "test_model",
                "model_classification": model_classification,
            }
        }
        result = model_reference_manager._file_json_dict_to_model_reference(category, file_json_dict)
        assert isinstance(result, dict)
        assert "test_model" in result
        assert isinstance(result["test_model"], GenericModelRecord)

    def test_file_json_dict_to_model_reference_invalid(self, model_reference_manager: ModelReferenceManager) -> None:
        """Test invalid conversion from a dict (parsed JSON) to model reference for both safe and non-safe modes."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        file_json_dict = {
            "invalid_model": {
                "description": "An invalid model without a name or classification",
            }
        }
        with pytest.raises(ValidationError):
            model_reference_manager._file_json_dict_to_model_reference(category, file_json_dict, safe_mode=True)

        non_safe_mode_result = model_reference_manager._file_json_dict_to_model_reference(
            category, file_json_dict, safe_mode=False
        )

        assert non_safe_mode_result is None, "Expected None result for invalid input in non-safe mode"

    def test_model_reference_to_json_dict(self, model_reference_manager: ModelReferenceManager) -> None:
        """Test conversion from model reference to dict (for JSON serialization)."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_classification = ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.miscellaneous,
        )
        file_json_dict = {
            "test_model": {
                "name": "test_model",
                "model_classification": model_classification,
            }
        }
        model_ref = model_reference_manager._file_json_dict_to_model_reference(
            category, file_json_dict, safe_mode=True
        )
        assert model_ref is not None, "Model reference conversion failed in safe mode, this should be impossible here."
        json_dict = model_reference_manager.model_reference_to_json_dict(model_ref)
        assert isinstance(json_dict, dict)
        assert "test_model" in json_dict
        assert json_dict["test_model"]["name"] == "test_model"

    def test_model_reference_to_json_dict_none(
        self,
        model_reference_manager: ModelReferenceManager,
        tmp_path: Path,
    ) -> None:
        """Test conversion from None model reference to dict returns None."""
        with pytest.raises(ValueError):
            model_reference_manager.model_reference_to_json_dict_safe(None)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            model_reference_manager.model_reference_to_json_dict(None, safe_mode=True)  # type: ignore[arg-type]

        with pytest.raises(ValueError):
            model_reference_manager.model_reference_to_json_dict(None, safe_mode=False)  # type: ignore[arg-type]

        """Test handling when some categories load successfully and others fail."""
        import json

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)

        valid_category = MODEL_REFERENCE_CATEGORY.miscellaneous
        valid_data = {
            "model1": {
                "name": "model1",
                "model_classification": {
                    "domain": MODEL_DOMAIN.image.value,
                    "purpose": MODEL_PURPOSE.miscellaneous.value,
                },
            }
        }

        valid_file_path = backend.get_category_file_path(valid_category)
        assert valid_file_path is not None
        valid_file_path.parent.mkdir(parents=True, exist_ok=True)
        valid_file_path.write_text(json.dumps(valid_data))

        corrupted_category = MODEL_REFERENCE_CATEGORY.blip
        corrupted_file_path = backend.get_category_file_path(corrupted_category)
        assert corrupted_file_path is not None
        corrupted_file_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted_file_path.write_text("{ bad json }")

        all_refs = backend.fetch_all_categories(force_refresh=True)

        assert all_refs[valid_category] is not None
        valid_refs = all_refs[valid_category]
        assert valid_refs is not None
        assert "model1" in valid_refs

        assert all_refs[corrupted_category] is None


@pytest.mark.asyncio
async def test_fetch_from_backend_if_needed_async_forwards(restore_manager_singleton: None) -> None:
    """Ensure manager async helper delegates to backend with given parameters."""
    backend = _InMemoryReplicaBackend()
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.REPLICA,
    )

    async with httpx.AsyncClient() as client:
        await manager._fetch_from_backend_if_needed_async(force_refresh=False, httpx_client=client)
        await manager._fetch_from_backend_if_needed_async(force_refresh=True, httpx_client=None)

    assert len(backend.async_calls) == 2
    first_call = backend.async_calls[0]
    assert first_call["force_refresh"] is False
    assert first_call["httpx_client"] is client
    second_call = backend.async_calls[1]
    assert second_call["force_refresh"] is True
    assert second_call["httpx_client"] is None


@pytest.mark.usefixtures("restore_manager_singleton")
class TestCRUDOperations:
    """Test create, update, delete operations in ModelReferenceManager."""

    def test_write_operations_require_primary_mode(self, tmp_path: Path) -> None:
        """Test that write operations fail in REPLICA mode."""
        backend = _InMemoryReplicaBackend()
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.REPLICA,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        record = GenericModelRecord(
            name="test_model",
            record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
            ),
        )

        with pytest.raises(NotImplementedError, match="does not support write"):
            manager.backend.update_model_from_base_model(category, record.name, record)

        with pytest.raises(NotImplementedError, match="does not support write"):
            manager.backend.update_model(category, record.name, record.model_dump())

        with pytest.raises(NotImplementedError, match="does not support write"):
            manager.backend.delete_model(category, "some_model")

    def test_update_model_invalidates_cache(self, tmp_path: Path) -> None:
        """Test that update_model properly invalidates the cache for the affected category."""
        backend = FileSystemBackend(
            base_path=tmp_path,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        record = GenericModelRecord(
            name="test_model",
            record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
            ),
        )

        manager.backend.update_model_from_base_model(category, record.name, record)

        _ = manager.get_all_model_references_unsafe()

        updated_record = GenericModelRecord(
            name="test_model",
            description="Updated description",
            record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
            ),
        )
        manager.backend.update_model_from_base_model(category, updated_record.name, updated_record)

        refs_after = manager.get_all_model_references_unsafe()
        assert refs_after[category] is not None
        category_refs = refs_after[category]
        assert category_refs is not None
        assert category_refs["test_model"].description == "Updated description"

    def test_delete_model_invalidates_cache(self, tmp_path: Path) -> None:
        """Test that delete_model properly invalidates the cache for the affected category."""
        backend = FileSystemBackend(
            base_path=tmp_path,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        record = GenericModelRecord(
            name="test_model",
            record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
            ),
        )

        manager.backend.update_model_from_base_model(category, record.name, record)
        _ = manager.get_all_model_references_unsafe()
        assert category in manager._cached_records

        manager.backend.delete_model(category, record.name)

        assert category not in manager._cached_records

        refs_after = manager.get_all_model_references_unsafe()
        assert refs_after[category] == {}


@pytest.mark.usefixtures("restore_manager_singleton")
class TestFileJsonIO:
    """Test reading raw JSON from files without pydantic validation."""

    def test_get_raw_model_reference_json_basic(self, tmp_path: Path) -> None:
        """Test get_raw_model_reference_json returns raw JSON without pydantic validation."""
        import json

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        test_data = {
            "model1": {
                "name": "model1",
                "model_classification": {
                    "domain": MODEL_DOMAIN.image.value,
                    "purpose": MODEL_PURPOSE.miscellaneous.value,
                },
            }
        }

        file_path = backend.get_category_file_path(category)
        assert file_path is not None
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(test_data))

        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        raw_json = manager.get_raw_model_reference_json(category)

        assert raw_json is not None
        assert raw_json == test_data
        assert "model1" in raw_json
        assert isinstance(raw_json, dict)

    def test_get_raw_model_reference_json_caching(self, tmp_path: Path) -> None:
        """Test that get_raw_model_reference_json properly caches results."""
        import json

        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        test_data = {
            "model1": {
                "name": "model1",
                "model_classification": {
                    "domain": MODEL_DOMAIN.image.value,
                    "purpose": MODEL_PURPOSE.miscellaneous.value,
                },
            }
        }

        file_path = backend.get_category_file_path(category)
        assert file_path is not None
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(test_data))

        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        first_result = manager.get_raw_model_reference_json(category)
        assert first_result == test_data

        # Second call should return cached data from backend
        second_result = manager.get_raw_model_reference_json(category)
        assert second_result == test_data

        # Force refresh should bypass cache
        third_result = manager.get_raw_model_reference_json(category, overwrite_existing=True)
        assert third_result == test_data

    def test_get_raw_model_reference_json_missing_file(self, tmp_path: Path) -> None:
        """Test that get_raw_model_reference_json returns None for missing files."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        result = manager.get_raw_model_reference_json(category)
        assert result is None

    def test_get_raw_model_reference_json_invalid_json(
        self,
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that get_raw_model_reference_json handles corrupted JSON gracefully."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        file_path = backend.get_category_file_path(category)
        assert file_path is not None
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text("{ this is not valid json }")

        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        result = manager.get_raw_model_reference_json(category)
        assert result is None

        # Backend logs "Failed to read" for JSON parse errors
        assert any("Failed to read" in record.message for record in caplog.records)

    def test_corrupted_json_in_get_all_model_references(
        self,
        tmp_path: Path,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that get_all_model_references_unsafe handles corrupted JSON files."""
        category = MODEL_REFERENCE_CATEGORY.miscellaneous

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)
        file_path = backend.get_category_file_path(category)
        assert file_path is not None
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.write_text("{ corrupted json content")

        manager = ModelReferenceManager(
            backend=backend,
            lazy_mode=True,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        all_refs = manager.get_all_model_references_unsafe()

        assert all_refs[category] is None

        # Backend logs "Failed to read" for JSON parse errors
        assert any("Failed to read" in record.message for record in caplog.records)

    def test_model_reference_to_json_dict_safe_success(self) -> None:
        """Test model_reference_to_json_dict_safe with valid input."""
        manager = ModelReferenceManager(lazy_mode=True, replicate_mode=ReplicateMode.REPLICA)

        category = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_classification = ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.miscellaneous,
        )
        file_json_dict = {
            "test_model": {
                "name": "test_model",
                "model_classification": model_classification,
            }
        }

        model_ref = manager._file_json_dict_to_model_reference(
            category,
            file_json_dict,
            safe_mode=True,
        )
        assert model_ref is not None

        json_dict = manager.model_reference_to_json_dict_safe(model_ref)

        assert isinstance(json_dict, dict)
        assert "test_model" in json_dict
        assert json_dict["test_model"]["name"] == "test_model"

    def test_partial_category_success(self, tmp_path: Path) -> None:
        """Test handling when some categories load successfully and others fail."""
        import json

        backend = FileSystemBackend(base_path=tmp_path, replicate_mode=ReplicateMode.PRIMARY)

        valid_category = MODEL_REFERENCE_CATEGORY.miscellaneous
        valid_data = {
            "model1": {
                "name": "model1",
                "model_classification": {
                    "domain": MODEL_DOMAIN.image.value,
                    "purpose": MODEL_PURPOSE.miscellaneous.value,
                },
            }
        }

        valid_file_path = backend.get_category_file_path(valid_category)
        assert valid_file_path is not None
        valid_file_path.parent.mkdir(parents=True, exist_ok=True)
        valid_file_path.write_text(json.dumps(valid_data))

        corrupted_category = MODEL_REFERENCE_CATEGORY.blip
        corrupted_file_path = backend.get_category_file_path(corrupted_category)
        assert corrupted_file_path is not None
        corrupted_file_path.parent.mkdir(parents=True, exist_ok=True)
        corrupted_file_path.write_text("{ bad json }")

        with_invalid_entry = backend.fetch_all_categories(force_refresh=True)

        assert with_invalid_entry[valid_category] is not None
        assert with_invalid_entry[corrupted_category] is None
