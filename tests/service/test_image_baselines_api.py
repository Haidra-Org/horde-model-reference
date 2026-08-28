"""Public API behavior for the served image-generation baseline catalog."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    CanonicalFormat,
    ModelReferenceManager,
    horde_model_reference_settings,
)
from horde_model_reference.image_baseline import (
    BaselineCapabilities,
    HordeBaselinePolicy,
    ImageBaselineChange,
    ImageBaselineChangeSet,
    ImageBaselineRecord,
)
from horde_model_reference.image_baseline_store import ImageBaselineStore
from horde_model_reference.service.shared import get_model_reference_manager

_ROOT = "/api/model_references/v2/image_generation/baselines"
_BOOTSTRAP_PATH = Path(__file__).resolve().parents[2] / "src" / "horde_model_reference" / "data" / "baselines"


@dataclass
class _BaselineManager:
    image_baseline_store: ImageBaselineStore
    image_models: dict[str, dict[str, Any]]

    def get_raw_model_reference_json(self, _category: object) -> dict[str, dict[str, Any]]:
        return self.image_models


def _manager(tmp_path: Path) -> _BaselineManager:
    store = ImageBaselineStore(root_path=tmp_path / "baselines", bootstrap_path=_BOOTSTRAP_PATH)
    return _BaselineManager(
        image_baseline_store=store,
        image_models={"a_model": {"name": "a_model", "baseline": "stable_diffusion_1"}},
    )


def _new_baseline_record(name: str) -> ImageBaselineRecord:
    """Build the record a proposal would publish for a newly released family."""
    return ImageBaselineRecord(
        name=name,
        display_name="Test Future Family",
        native_resolution=1024,
        alternative_names=("test future family",),
        capabilities=BaselineCapabilities(controlnet=False, transparent=False, qr_code=False, remix=False),
        horde_policy=HordeBaselinePolicy(kudos=8, batching=8, ttl=3, resolution_floor=1024),
    )


def test_catalog_reads_expose_records_and_a_revision_marker(
    api_client: TestClient,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    tmp_path: Path,
) -> None:
    """The index, single read, and export all serve the same catalog revision."""
    dependency_override(get_model_reference_manager, lambda: _manager(tmp_path))

    listed = api_client.get(f"{_ROOT}/")
    single = api_client.get(f"{_ROOT}/stable_diffusion_xl")
    exported = api_client.get(f"{_ROOT}/export")
    missing = api_client.get(f"{_ROOT}/no_such_baseline")

    assert listed.status_code == 200
    assert listed.json()["total"] == len(exported.json()["baselines"])
    assert listed.headers["etag"] == 'W/"image-baselines-1"'
    assert single.status_code == 200
    assert single.json()["horde_policy"]["kudos_qr_code"] == 4
    assert single.json()["capabilities"]["flow_matching"] is False
    assert exported.status_code == 200
    assert exported.json()["metadata"]["revision"] == 1
    assert missing.status_code == 404


def test_a_published_baseline_reaches_model_submission_through_the_queue(
    api_client: TestClient,
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    mock_auth_success: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An approved baseline becomes submittable without rewriting canonical model records."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)
    manager.backend.update_model(
        MODEL_REFERENCE_CATEGORY.image_generation,
        "an_existing_model",
        {
            "name": "an_existing_model",
            "record_type": "image_generation",
            "model_classification": {"domain": "image", "purpose": "generation"},
            "baseline": "stable_diffusion_1",
            "nsfw": False,
        },
    )
    manager._invalidate_cache()
    reference_path = manager.backend.get_category_file_path(MODEL_REFERENCE_CATEGORY.image_generation)
    assert reference_path is not None
    reference_bytes_before = reference_path.read_bytes()
    headers = {"apikey": "test-key"}
    model_payload = {
        "name": "a_future_family_model",
        "record_type": "image_generation",
        "model_classification": {"domain": "image", "purpose": "generation"},
        "baseline": "test_future_family",
        "nsfw": False,
    }

    rejected = api_client.post(
        "/api/model_references/v2/image_generation/add",
        json=model_payload,
        headers=headers,
    )
    assert rejected.status_code == 422
    assert "Unknown baseline" in rejected.text

    submission = api_client.post(
        f"{_ROOT}/change-sets",
        json=ImageBaselineChangeSet(
            title="Publish the test future family",
            changes=[
                ImageBaselineChange(
                    operation="upsert",
                    name="test_future_family",
                    record=_new_baseline_record("test_future_family"),
                ),
            ],
        ).model_dump(mode="json"),
        headers=headers,
    )
    assert submission.status_code == 202
    change_id = submission.json()["change_id"]

    approval = api_client.post(
        "/api/model_references/v2/pending_queue/batches",
        json={"batch_title": "publish baseline", "approved_ids": [change_id]},
        headers=headers,
    )
    assert approval.status_code == 200

    applied = api_client.post(
        f"/api/model_references/v2/pending_queue/changes/{change_id}/apply",
        json={},
        headers=headers,
    )
    assert applied.status_code == 200
    assert applied.json()["record"]["status"] == "applied"

    published = api_client.get(f"{_ROOT}/test_future_family")
    assert published.status_code == 200
    assert published.json()["horde_policy"]["batching"] == 8
    assert reference_path.read_bytes() == reference_bytes_before

    accepted = api_client.post(
        "/api/model_references/v2/image_generation/add",
        json=model_payload,
        headers=headers,
    )
    assert accepted.status_code == 202


def test_model_submission_uses_the_served_catalog_not_the_static_enum(
    api_client: TestClient,
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    mock_auth_success: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retiring a catalog entry takes effect even while an older package enum still names it."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)
    current = manager.image_baseline_store.get("krea2_turbo")
    assert current is not None
    manager.image_baseline_store.apply_change_set(
        ImageBaselineChangeSet(
            title="Temporarily retire Krea in this isolated catalog",
            changes=[ImageBaselineChange(operation="delete", name="krea2_turbo", expected_before=current)],
        ),
        referenced_baselines=set(),
        editor_id="maintainer",
    )

    response = api_client.post(
        "/api/model_references/v2/image_generation/add",
        json={
            "name": "a_krea_model",
            "record_type": "image_generation",
            "model_classification": {"domain": "image", "purpose": "generation"},
            "baseline": "krea2_turbo",
            "nsfw": False,
        },
        headers={"apikey": "test-key"},
    )

    assert response.status_code == 422
    assert "Publish it through" in response.text
