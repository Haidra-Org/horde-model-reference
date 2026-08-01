"""Test the first-class v2 licensing endpoints."""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.service.shared import get_model_reference_manager

pytestmark = pytest.mark.usefixtures("mock_auth_success")

_LICENSING_ROOT = "/model_references/v2/licensing"


def _model_payload(name: str) -> dict[str, object]:
    """Create a model payload suitable for exercising licensing API behavior."""
    return {
        "name": name,
        "record_type": "miscellaneous",
        "model_classification": {"domain": "image", "purpose": "miscellaneous"},
        "config": {
            "download": [
                {
                    "file_name": "weights.bin",
                    "file_url": "https://example.invalid/weights.bin",
                },
            ],
        },
    }


def _mit_assignment() -> dict[str, object]:
    """Return a known catalog-backed conclusion for endpoint scenarios."""
    return {
        "license_expression": "MIT",
        "license_ids": ["MIT"],
        "commercial_use": "allowed",
        "redistribution": "allowed_with_conditions",
        "obligations": ["include_license"],
    }


@pytest.fixture
def licensing_manager(
    primary_manager_override_factory: Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """Create an isolated PRIMARY manager and authorize a direct licensing editor."""
    monkeypatch.setattr(horde_model_reference_settings.licensing, "editor_ids", ["test-user-id"])
    yield primary_manager_override_factory(get_model_reference_manager)


def test_catalog_and_summary_are_machine_readable(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify public catalog endpoints expose definitions, assets, and conservative aggregates."""
    definitions_response = api_client.get(f"{_LICENSING_ROOT}/licenses")
    assets_response = api_client.get(f"{_LICENSING_ROOT}/assets?asset_kind=custom_node")
    summary_response = api_client.get(f"{_LICENSING_ROOT}/summary")

    assert definitions_response.status_code == 200
    assert definitions_response.json()["total"] >= 1
    assert any(item["license_id"] == "MIT" for item in definitions_response.json()["items"])
    assert assets_response.status_code == 200
    assert all(item["asset_kind"] == "custom_node" for item in assets_response.json()["items"])
    assert summary_response.status_code == 200
    assert summary_response.json()["total_assets"] == len(licensing_manager.licensing_store.list_assets())


def test_definition_crud_uses_independent_editor_authorization(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify definition writes persist directly and carry lifecycle metadata."""
    payload = {
        "license_id": "LicenseRef-Endpoint-Test",
        "name": "Endpoint Test License",
        "canonical_url": "https://example.invalid/licenses/endpoint-test",
        "commercial_use": "allowed",
        "redistribution": "allowed",
    }

    create_response = api_client.post(
        f"{_LICENSING_ROOT}/licenses",
        headers={"apikey": "test-key"},
        json=payload,
    )
    get_response = api_client.get(f"{_LICENSING_ROOT}/licenses/LicenseRef-Endpoint-Test")
    delete_response = api_client.delete(
        f"{_LICENSING_ROOT}/licenses/LicenseRef-Endpoint-Test",
        headers={"apikey": "test-key"},
    )

    assert create_response.status_code == 201
    assert create_response.json()["metadata"]["created_by"] == "test-user-id"
    assert get_response.status_code == 200
    assert delete_response.status_code == 204


def test_v2_reads_make_legacy_uncertainty_explicit_without_mutating_storage(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify consumers receive fail-closed data while legacy source records remain untouched."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    payload = _model_payload("legacy-unaudited")
    licensing_manager.backend.update_model(category, "legacy-unaudited", payload)
    licensing_manager._invalidate_cache()

    response = api_client.get(f"/model_references/v2/{category.value}/model/legacy-unaudited")
    stored_payload = licensing_manager.get_raw_model_json(category, "legacy-unaudited")

    assert response.status_code == 200
    assert response.json()["licensing"]["license_expression"] == "NOASSERTION"
    assert response.json()["licensing"]["commercial_use"] == "unknown"
    assert stored_payload is not None
    assert "licensing" not in stored_payload


def test_asset_filters_and_summary_distinguish_known_from_unaudited_models(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify policy filters operate on conclusions rather than field presence."""
    known_payload = _model_payload("known-license")
    known_payload["licensing"] = _mit_assignment()
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    licensing_manager.backend.update_model(category, "known-license", known_payload)
    licensing_manager.backend.update_model(category, "legacy-unknown", _model_payload("legacy-unknown"))
    licensing_manager._invalidate_cache()

    known_response = api_client.get(
        f"{_LICENSING_ROOT}/assets",
        params={"asset_kind": "model", "license_id": "MIT"},
    )
    unknown_response = api_client.get(
        f"{_LICENSING_ROOT}/assets",
        params={"asset_kind": "model", "commercial_use": "unknown"},
    )
    summary_response = api_client.get(f"{_LICENSING_ROOT}/summary")

    assert known_response.status_code == 200
    assert [asset["display_name"] for asset in known_response.json()["items"]] == ["known-license"]
    assert known_response.json()["items"][0]["definition_urls"]["MIT"].endswith("/licenses/MIT")
    assert unknown_response.status_code == 200
    assert [asset["display_name"] for asset in unknown_response.json()["items"]] == ["legacy-unknown"]
    assert summary_response.json()["commercial_use"]["unknown"] == 1


def test_definition_delete_is_blocked_by_a_file_override_reference(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify file-scoped foreign keys protect catalog rows just like model-level keys."""
    definition_payload = {
        "license_id": "LicenseRef-File-Only",
        "name": "File-only Test License",
        "canonical_url": "https://example.invalid/licenses/file-only",
        "commercial_use": "prohibited",
        "redistribution": "prohibited",
    }
    create_response = api_client.post(
        f"{_LICENSING_ROOT}/licenses",
        headers={"apikey": "test-key"},
        json=definition_payload,
    )
    model_payload = _model_payload("file-scoped-reference")
    model_payload["licensing"] = {
        "license_expression": "NOASSERTION",
        "commercial_use": "unknown",
        "redistribution": "unknown",
        "files": {
            "weights.bin": {
                "license_expression": "LicenseRef-File-Only",
                "license_ids": ["LicenseRef-File-Only"],
                "commercial_use": "prohibited",
                "redistribution": "prohibited",
            },
        },
    }
    licensing_manager.backend.update_model(
        MODEL_REFERENCE_CATEGORY.miscellaneous,
        "file-scoped-reference",
        model_payload,
    )
    licensing_manager._invalidate_cache()

    delete_response = api_client.delete(
        f"{_LICENSING_ROOT}/licenses/LicenseRef-File-Only",
        headers={"apikey": "test-key"},
    )

    assert create_response.status_code == 201
    assert delete_response.status_code == 409
    assert api_client.get(f"{_LICENSING_ROOT}/licenses/LicenseRef-File-Only").status_code == 200


def test_pending_queue_approver_does_not_inherit_license_editor_authority(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the direct-write trust boundary remains independent from model approval."""
    monkeypatch.setattr(horde_model_reference_settings.licensing, "editor_ids", ["different-user"])
    payload = {
        "license_id": "LicenseRef-Unauthorized",
        "name": "Unauthorized Test",
        "canonical_url": "https://example.invalid/licenses/unauthorized",
        "commercial_use": "allowed",
        "redistribution": "allowed",
    }

    response = api_client.post(
        f"{_LICENSING_ROOT}/licenses",
        headers={"apikey": "test-key"},
        json=payload,
    )

    assert response.status_code == 403
    assert licensing_manager.licensing_store.get_definition("LicenseRef-Unauthorized") is None


def test_invalid_asset_foreign_key_is_rejected_without_partial_creation(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify API validation cannot leave an orphaned auxiliary asset behind."""
    payload = {
        "asset_kind": "other",
        "asset_identifier": "orphan-api-asset",
        "display_name": "Orphan API Asset",
        "licensing": {
            "license_expression": "LicenseRef-Does-Not-Exist",
            "license_ids": ["LicenseRef-Does-Not-Exist"],
            "commercial_use": "unknown",
            "redistribution": "unknown",
        },
    }

    create_response = api_client.post(
        f"{_LICENSING_ROOT}/assets",
        headers={"apikey": "test-key"},
        json=payload,
    )
    read_response = api_client.get(f"{_LICENSING_ROOT}/assets/other/orphan-api-asset")

    assert create_response.status_code == 422
    assert read_response.status_code == 404


def test_deprecation_preserves_existing_references_but_blocks_new_assignments(
    api_client: TestClient,
    licensing_manager: ModelReferenceManager,
) -> None:
    """Verify retirement is non-destructive while preventing additional use of an obsolete definition."""
    definition = {
        "license_id": "LicenseRef-Retiring",
        "name": "Retiring License",
        "canonical_url": "https://example.invalid/licenses/retiring",
        "commercial_use": "allowed",
        "redistribution": "allowed",
    }
    first_asset = {
        "asset_kind": "other",
        "asset_identifier": "existing-reference",
        "display_name": "Existing Reference",
        "licensing": {
            "license_expression": "LicenseRef-Retiring",
            "license_ids": ["LicenseRef-Retiring"],
            "commercial_use": "allowed",
            "redistribution": "allowed",
        },
    }
    assert (
        api_client.post(
            f"{_LICENSING_ROOT}/licenses",
            headers={"apikey": "test-key"},
            json=definition,
        ).status_code
        == 201
    )
    assert (
        api_client.post(
            f"{_LICENSING_ROOT}/assets",
            headers={"apikey": "test-key"},
            json=first_asset,
        ).status_code
        == 201
    )

    deprecate_response = api_client.put(
        f"{_LICENSING_ROOT}/licenses/LicenseRef-Retiring",
        headers={"apikey": "test-key"},
        json={**definition, "deprecated": True},
    )
    second_asset = {
        **first_asset,
        "asset_identifier": "new-reference",
        "display_name": "New Reference",
    }
    second_create_response = api_client.post(
        f"{_LICENSING_ROOT}/assets",
        headers={"apikey": "test-key"},
        json=second_asset,
    )

    assert deprecate_response.status_code == 200
    assert deprecate_response.json()["deprecated"] is True
    assert api_client.get(f"{_LICENSING_ROOT}/assets/other/existing-reference").status_code == 200
    assert second_create_response.status_code == 422
    assert "Deprecated license definition" in second_create_response.json()["detail"]
