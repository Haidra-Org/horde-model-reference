"""Tests for the offline LocalReadOnlyBackend and ModelReferenceManager(offline=True).

These verify the worker-subprocess contract: references are read from local disk that the parent
process already populated, and no network access ever occurs.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    PrefetchStrategy,
    ReplicateMode,
    horde_model_reference_paths,
)
from horde_model_reference.backends.local_readonly_backend import LocalReadOnlyBackend
from horde_model_reference.model_reference_manager import ModelReferenceManager


def _write_category_file(
    base_path: Path,
    category: MODEL_REFERENCE_CATEGORY,
    payload: dict[str, Any],
) -> Path:
    file_path = horde_model_reference_paths.get_model_reference_file_path(category, base_path=base_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload))
    return file_path


def _minimal_record_dict(name: str) -> dict[str, Any]:
    return {
        name: {
            "name": name,
            "model_classification": {
                "domain": MODEL_DOMAIN.image.value,
                "purpose": MODEL_PURPOSE.miscellaneous.value,
            },
        },
    }


def _explode_if_network(*args: object, **kwargs: object) -> None:
    raise AssertionError("offline backend attempted network access")


def test_local_readonly_reads_disk_without_network(
    primary_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend serves whatever the parent wrote to disk and never downloads."""
    monkeypatch.setattr("requests.get", _explode_if_network)

    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    _write_category_file(primary_base, category, _minimal_record_dict("on_disk"))

    backend = LocalReadOnlyBackend(base_path=primary_base, cache_ttl_seconds=60)

    assert backend.replicate_mode == ReplicateMode.REPLICA
    assert backend.supports_writes() is False

    fetched = backend.fetch_category(category)
    assert fetched is not None
    assert "on_disk" in fetched


def test_local_readonly_missing_category_returns_none(primary_base: Path) -> None:
    """A category the parent never wrote returns None instead of triggering a download."""
    backend = LocalReadOnlyBackend(base_path=primary_base, cache_ttl_seconds=60)
    assert backend.fetch_category(MODEL_REFERENCE_CATEGORY.controlnet) is None


def test_local_readonly_refreshes_on_mtime_change(
    primary_base: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent refresh that rewrites the file is picked up on the next forced fetch."""
    monkeypatch.setattr("requests.get", _explode_if_network)

    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    file_path = _write_category_file(primary_base, category, _minimal_record_dict("first"))

    backend = LocalReadOnlyBackend(base_path=primary_base, cache_ttl_seconds=60)
    assert "first" in (backend.fetch_category(category) or {})

    # Simulate the parent rewriting the converted file with new content and a newer mtime.
    time.sleep(0.01)
    file_path.write_text(json.dumps(_minimal_record_dict("second")))
    import os

    new_mtime = time.time() + 5
    os.utime(file_path, (new_mtime, new_mtime))

    refreshed = backend.fetch_category(category, force_refresh=True)
    assert refreshed is not None
    assert "second" in refreshed


def test_manager_offline_selects_local_readonly_backend(
    primary_base: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_manager_singleton: None,
) -> None:
    """ModelReferenceManager(offline=True) uses the read-only backend and never downloads."""
    monkeypatch.setattr("requests.get", _explode_if_network)

    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    _write_category_file(primary_base, category, _minimal_record_dict("offline_model"))

    manager = ModelReferenceManager(
        base_path=primary_base,
        offline=True,
        prefetch_strategy=PrefetchStrategy.NONE,
    )

    assert manager.offline is True
    assert isinstance(manager.backend, LocalReadOnlyBackend)
    assert manager.backend.supports_writes() is False
    assert manager.backend.replicate_mode == ReplicateMode.REPLICA

    reference = manager.get_model_reference(category)
    assert reference is not None
    assert "offline_model" in reference
