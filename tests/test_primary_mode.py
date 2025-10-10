import json
import os
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
    ReplicateMode,
    horde_model_reference_paths,
)
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import GenericModelRecord


def _write_category_file(
    base_path: Path,
    category: MODEL_REFERENCE_CATEGORY,
    payload: dict[str, Any],
) -> Path:
    file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=base_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload))
    return file_path


def _minimal_record_dict(name: str, description: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "model_classification": {
            "domain": MODEL_DOMAIN.image.value,
            "purpose": MODEL_PURPOSE.miscellaneous.value,
        },
    }
    if description is not None:
        record["description"] = description
    return {name: record}


def test_filesystem_backend_cache_expiry_respects_primary_mode(
    primary_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure PRIMARY filesystem backend serves cache until TTL then refreshes from disk."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    initial_payload = _minimal_record_dict("cached")
    file_path = _write_category_file(primary_base, category, initial_payload)

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    class _TimeStub:
        def __init__(self, value: float) -> None:
            self.value = value

        def time(self) -> float:
            return self.value

    time_stub = _TimeStub(1_000.0)
    monkeypatch.setattr(
        "horde_model_reference.backends.filesystem_backend.time.time",
        time_stub.time,
    )

    first = backend.fetch_category(category)
    assert first == initial_payload

    updated_payload = _minimal_record_dict("cached", description="updated")
    file_path.write_text(json.dumps(updated_payload))

    cached = backend.fetch_category(category)
    assert cached == initial_payload

    time_stub.value += 120

    refreshed = backend.fetch_category(category)
    assert refreshed == updated_payload


def test_filesystem_backend_primary_update_and_delete(primary_base: Path) -> None:
    """Verify PRIMARY filesystem backend persists updates and deletions to disk."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=30,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    backend.update_model(category, "new_model", _minimal_record_dict("new_model")["new_model"])
    file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=primary_base)
    stored = json.loads(file_path.read_text())
    assert "new_model" in stored
    assert category in backend._stale_categories

    backend.fetch_category(category)
    assert category not in backend._stale_categories

    backend.delete_model(category, "new_model")
    stored_after_delete = json.loads(file_path.read_text())
    assert stored_after_delete == {}
    assert category in backend._stale_categories


def test_model_reference_manager_primary_skips_replica_fetch(
    primary_base: Path,
    restore_manager_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirm PRIMARY manager never triggers replica fetch pathways."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    _write_category_file(primary_base, category, _minimal_record_dict("on_disk"))

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    def fail_fetch(*_args: object, **_kwargs: object) -> None:
        pytest.fail("fetch_all_categories should not be called in PRIMARY mode")

    monkeypatch.setattr(manager.backend, "fetch_all_categories", fail_fetch)

    all_refs = manager.get_all_model_references_unsafe()
    assert manager._replicate_mode == ReplicateMode.PRIMARY
    assert category in all_refs
    assert all_refs[category] is not None


def test_model_reference_manager_primary_write_paths(
    primary_base: Path,
    restore_manager_singleton: None,
) -> None:
    """Exercise PRIMARY manager write and delete flows."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    record = GenericModelRecord(
        name="primary_model",
        model_classification=ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.miscellaneous,
        ),
    )

    manager.update_model(category, record.name, record)

    raw = manager.get_raw_model_reference_json(category)
    assert raw is not None
    assert record.name in raw

    manager.delete_model(category, record.name)
    raw_after_delete = manager.get_raw_model_reference_json(category)
    assert raw_after_delete == {}


def test_manager_detects_external_file_update(
    primary_base: Path,
    restore_manager_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure external file edits invalidate cache and reload fresh data."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    initial_payload = _minimal_record_dict("initial")
    file_path = _write_category_file(primary_base, category, initial_payload)

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    marks: list[MODEL_REFERENCE_CATEGORY] = []
    original_mark_stale = manager.backend.mark_stale

    def tracking_mark_stale(target_category: MODEL_REFERENCE_CATEGORY) -> None:
        marks.append(target_category)
        original_mark_stale(target_category)

    monkeypatch.setattr(manager.backend, "mark_stale", tracking_mark_stale)

    first_refs = manager.get_all_model_references_unsafe()
    assert category in first_refs and first_refs[category] is not None
    first_category_refs = first_refs[category]
    assert first_category_refs is not None
    assert "initial" in first_category_refs
    assert not manager.backend.needs_refresh(category)

    updated_payload = _minimal_record_dict("updated")
    file_path.write_text(json.dumps(updated_payload))
    stat_before = file_path.stat()
    os.utime(file_path, (stat_before.st_atime, stat_before.st_mtime + 5))

    refreshed_refs = manager.get_all_model_references_unsafe()

    assert category in marks
    assert refreshed_refs[category] is not None
    refreshed_category_refs = refreshed_refs[category]
    assert refreshed_category_refs is not None
    assert "updated" in refreshed_category_refs
    assert "initial" not in refreshed_category_refs


def test_get_all_model_references_raises_when_missing(
    primary_base: Path,
    restore_manager_singleton: None,
) -> None:
    """Verify safe accessor raises when any category is unavailable."""
    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        lazy_mode=True,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    with pytest.raises(ValueError, match="Missing model references") as excinfo:
        manager.get_all_model_references()

    missing_message = str(excinfo.value)
    assert MODEL_REFERENCE_CATEGORY.miscellaneous.value in missing_message
