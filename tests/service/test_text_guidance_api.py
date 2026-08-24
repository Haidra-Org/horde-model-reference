"""Public API behavior for reusable text-model guidance."""

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
from horde_model_reference.service.shared import get_model_reference_manager
from horde_model_reference.text_guidance import (
    GuidanceAssignmentChange,
    GuidanceProfileChange,
    TextGuidanceAssignment,
    TextGuidanceChangeSet,
    TextInteractionMode,
    TextPromptContract,
)
from horde_model_reference.text_guidance_store import TextGuidanceStore

_ROOT = "/api/model_references/v2/text_generation/guidance"


@dataclass
class _GuidanceManager:
    text_guidance_store: TextGuidanceStore
    text_models: dict[str, dict[str, Any]]

    def get_raw_model_reference_json(self, _category: object) -> dict[str, dict[str, Any]]:
        return self.text_models


def _manager(tmp_path: Path) -> _GuidanceManager:
    store = TextGuidanceStore(root_path=tmp_path / "guidance")
    models = {
        "publisher/model-a": {"name": "publisher/model-a", "instruct_format": "ChatML"},
        "publisher/model-b": {"name": "publisher/model-b", "instruct_format": "ChatML"},
        "koboldcpp/publisher/model-a": {"name": "koboldcpp/publisher/model-a"},
    }
    store.apply_change_set(
        TextGuidanceChangeSet(
            title="Publish shared contract",
            profile_changes=[
                GuidanceProfileChange(
                    operation="create",
                    profile_id="chatml",
                    profile=TextPromptContract(
                        profile_id="chatml",
                        display_name="ChatML",
                        summary="Serialize messages with role markers.",
                        interaction_modes=[TextInteractionMode.CHAT],
                    ),
                ),
            ],
            assignment_changes=[
                GuidanceAssignmentChange(
                    model_name="publisher/model-a",
                    assignment=TextGuidanceAssignment(
                        model_name="publisher/model-a",
                        primary_profile_id="chatml",
                    ),
                ),
            ],
        ),
        canonical_model_names={"publisher/model-a", "publisher/model-b"},
        editor_id="maintainer",
    )
    return _GuidanceManager(text_guidance_store=store, text_models=models)


def test_profile_index_counts_exact_assignments_and_exposes_revision(
    api_client: TestClient,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    tmp_path: Path,
) -> None:
    """The index reports reuse counts and a consumer freshness marker."""
    manager = _manager(tmp_path)
    dependency_override(get_model_reference_manager, lambda: manager)

    response = api_client.get(f"{_ROOT}/profiles")

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"] == [
        {
            "profile_id": "chatml",
            "kind": "prompt_contract",
            "display_name": "ChatML",
            "summary": "Serialize messages with role markers.",
            "aliases": [],
            "deprecated": False,
            "assigned_model_count": 1,
        },
    ]
    assert response.headers["etag"] == 'W/"text-guidance-2"'


def test_model_resolution_distinguishes_published_from_legacy_fallback(
    api_client: TestClient,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    tmp_path: Path,
) -> None:
    """Exact canonical records resolve curated and historical states without backend duplicates."""
    manager = _manager(tmp_path)
    dependency_override(get_model_reference_manager, lambda: manager)

    published = api_client.get(f"{_ROOT}/model", params={"name": "publisher/model-a"})
    legacy = api_client.get(f"{_ROOT}/model", params={"name": "publisher/model-b"})
    projected = api_client.get(f"{_ROOT}/model", params={"name": "koboldcpp/publisher/model-a"})

    assert published.status_code == 200
    assert published.json()["summary"]["status"] == "published"
    assert published.json()["primary_profile"]["profile_id"] == "chatml"
    assert legacy.status_code == 200
    assert legacy.json()["summary"]["status"] == "legacy_label"
    assert legacy.json()["primary_profile"] is None
    assert projected.status_code == 404


def _stale_assignment_change_set() -> dict[str, Any]:
    """Return a change set whose precondition cannot match an empty catalog."""
    return TextGuidanceChangeSet(
        title="Assign a contract",
        profile_changes=[
            GuidanceProfileChange(
                operation="create",
                profile_id="chatml",
                profile=TextPromptContract(
                    profile_id="chatml",
                    display_name="ChatML",
                    summary="Serialize messages with role markers.",
                    interaction_modes=[TextInteractionMode.CHAT],
                ),
            ),
        ],
        assignment_changes=[
            GuidanceAssignmentChange(
                model_name="publisher/model-a",
                assignment=TextGuidanceAssignment(model_name="publisher/model-a", primary_profile_id="chatml"),
                expected_before=TextGuidanceAssignment(model_name="publisher/model-a", primary_profile_id="other"),
            ),
        ],
    ).model_dump(mode="json")


def test_submit_change_set_reports_stale_precondition_as_conflict(
    api_client: TestClient,
    primary_manager_override_factory: Callable[[Callable[[], Any]], Any],
    mock_auth_success: None,
) -> None:
    """A change set reviewed against a superseded catalog state is rejected with 409, not 500."""
    primary_manager_override_factory(get_model_reference_manager)

    response = api_client.post(
        f"{_ROOT}/change-sets",
        json=_stale_assignment_change_set(),
        headers={"apikey": "test-key"},
    )

    assert response.status_code == 409
    assert "changed after review" in response.json()["detail"]


def test_submit_change_set_rejects_non_canonical_model_as_unprocessable(
    api_client: TestClient,
    primary_manager_override_factory: Callable[[Callable[[], Any]], Any],
    mock_auth_success: None,
) -> None:
    """Assignments for models absent from the canonical text reference are rejected with 422."""
    primary_manager_override_factory(get_model_reference_manager)
    payload = _stale_assignment_change_set()
    payload["assignment_changes"][0]["expected_before"] = None

    response = api_client.post(
        f"{_ROOT}/change-sets",
        json=payload,
        headers={"apikey": "test-key"},
    )

    assert response.status_code == 422
    assert "Unknown canonical text model" in response.json()["detail"]


def _text_record(name: str, instruct_format: str | None) -> dict[str, object]:
    """Build a minimal canonical text generation record."""
    record: dict[str, object] = {
        "name": name,
        "record_type": "text_generation",
        "model_classification": {"domain": "text", "purpose": "generation"},
        "parameters": 8_000_000_000,
        "baseline": "qwen3",
        "nsfw": False,
    }
    if instruct_format is not None:
        record["instruct_format"] = instruct_format
    return record


def test_migration_preview_reaches_publication_through_the_queue(
    api_client: TestClient,
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    mock_auth_success: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previewed legacy migration publishes guidance without rewriting canonical model records."""
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", CanonicalFormat.v2)
    manager = primary_manager_override_factory(get_model_reference_manager)
    for model_name, instruct_format in (
        ("publisher/model-a", "ChatML"),
        ("publisher/model-b", "ChatML"),
        ("publisher/model-c", None),
    ):
        manager.backend.update_model(
            MODEL_REFERENCE_CATEGORY.text_generation,
            model_name,
            _text_record(model_name, instruct_format),
        )
    manager._invalidate_cache()
    reference_path = manager.backend.get_category_file_path(MODEL_REFERENCE_CATEGORY.text_generation)
    assert reference_path is not None
    reference_bytes_before = reference_path.read_bytes()
    headers = {"apikey": "test-key"}

    preview = api_client.post(f"{_ROOT}/migration/preview", headers=headers)
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["format_count"] == 1
    assert preview_payload["source_model_count"] == 2

    submission = api_client.post(f"{_ROOT}/change-sets", json=preview_payload["change_set"], headers=headers)
    assert submission.status_code == 202
    change_id = submission.json()["change_id"]

    approval = api_client.post(
        "/api/model_references/v2/pending_queue/batches",
        json={"batch_title": "seed guidance", "approved_ids": [change_id]},
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

    resolved = api_client.get(f"{_ROOT}/model", params={"name": "publisher/model-a"})
    assert resolved.status_code == 200
    assert resolved.json()["summary"]["status"] == "published"
    unlabeled = api_client.get(f"{_ROOT}/model", params={"name": "publisher/model-c"})
    assert unlabeled.json()["summary"]["status"] == "undocumented"
    assert reference_path.read_bytes() == reference_bytes_before
