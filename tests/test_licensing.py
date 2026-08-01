"""Test normalized licensing validation, persistence, and report generation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from horde_model_reference.cli.license_report import generate_license_report
from horde_model_reference.licensing import (
    LicenseAssignment,
    LicensedAsset,
    LicensedAssetKind,
    LicenseDefinition,
    ModelLicensing,
    PermissionStatus,
    unknown_model_licensing,
)
from horde_model_reference.licensing_store import LicensingStore
from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import DownloadRecord, GenericModelRecord, GenericModelRecordConfig
from horde_model_reference.pending_queue.diff_utils import compute_field_diffs, has_critical_changes


def _test_definition() -> LicenseDefinition:
    """Create a reusable custom definition for store tests."""
    return LicenseDefinition(
        license_id="LicenseRef-Test-Permissive",
        name="Test Permissive License",
        canonical_url="https://example.invalid/licenses/test-permissive",
        commercial_use=PermissionStatus.ALLOWED,
        redistribution=PermissionStatus.ALLOWED,
    )


def _test_assignment() -> LicenseAssignment:
    """Create an assignment referencing the reusable test definition."""
    return LicenseAssignment(
        license_expression="LicenseRef-Test-Permissive",
        license_ids=("LicenseRef-Test-Permissive",),
        commercial_use=PermissionStatus.ALLOWED,
        redistribution=PermissionStatus.ALLOWED,
    )


def test_unknown_assignment_is_explicit_and_fail_closed() -> None:
    """Verify unaudited records expose a machine-readable conservative conclusion."""
    assignment = unknown_model_licensing()

    assert assignment.license_expression == "NOASSERTION"
    assert assignment.license_ids == ()
    assert assignment.commercial_use is PermissionStatus.UNKNOWN
    assert assignment.redistribution is PermissionStatus.UNKNOWN


def test_assignment_requires_expression_identifiers_to_match() -> None:
    """Verify an expression cannot silently diverge from its definition references."""
    with pytest.raises(ValidationError, match="exactly match"):
        LicenseAssignment(
            license_expression="MIT",
            license_ids=("Apache-2.0",),
            commercial_use=PermissionStatus.ALLOWED,
            redistribution=PermissionStatus.ALLOWED,
        )


@pytest.mark.parametrize(
    ("commercial_use", "redistribution"),
    [
        (PermissionStatus.ALLOWED, PermissionStatus.UNKNOWN),
        (PermissionStatus.UNKNOWN, PermissionStatus.PROHIBITED),
    ],
)
def test_noassertion_cannot_claim_a_permission_conclusion(
    commercial_use: PermissionStatus,
    redistribution: PermissionStatus,
) -> None:
    """Verify incomplete evidence cannot accidentally become an affirmative policy decision."""
    with pytest.raises(ValidationError, match="NOASSERTION requires"):
        LicenseAssignment(
            license_expression="NOASSERTION",
            commercial_use=commercial_use,
            redistribution=redistribution,
        )


def test_failed_asset_write_is_atomic(tmp_path: Path) -> None:
    """Verify a missing foreign key leaves records, revisions, and audit history unchanged."""
    store = LicensingStore(root_path=tmp_path)
    initial_revision = store.metadata().revision
    invalid_asset = LicensedAsset(
        asset_kind=LicensedAssetKind.OTHER,
        asset_identifier="orphan",
        display_name="Orphan",
        licensing=_test_assignment(),
    )

    with pytest.raises(ValueError, match="Unknown license definition"):
        store.create_asset(invalid_asset, editor_id="editor-1")

    assert store.list_assets() == []
    assert store.metadata().revision == initial_revision
    assert not (tmp_path / "audit.jsonl").exists()


def test_store_crud_validates_foreign_keys_and_reference_deletion(tmp_path: Path) -> None:
    """Verify direct records behave like normalized tables with protected references."""
    store = LicensingStore(root_path=tmp_path)
    definition = store.create_definition(_test_definition(), editor_id="editor-1")
    asset = LicensedAsset(
        asset_kind=LicensedAssetKind.OTHER,
        asset_identifier="test-asset",
        display_name="Test Asset",
        licensing=_test_assignment(),
    )

    persisted_asset = store.create_asset(asset, editor_id="editor-1")

    assert definition.metadata.created_by == "editor-1"
    assert persisted_asset.metadata.created_by == "editor-1"
    with pytest.raises(ValueError, match="referenced by assets"):
        store.delete_definition(definition.license_id, editor_id="editor-1")

    store.delete_asset(
        asset_kind=LicensedAssetKind.OTHER,
        asset_identifier="test-asset",
        editor_id="editor-1",
    )
    store.delete_definition(definition.license_id, editor_id="editor-1")
    assert store.get_definition(definition.license_id) is None
    audit_events = [
        json.loads(serialized_event)
        for serialized_event in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["operation"] for event in audit_events] == ["create", "create", "delete", "delete"]
    assert audit_events[1]["after"]["asset_identifier"] == "test-asset"
    assert audit_events[2]["before"]["asset_identifier"] == "test-asset"


def test_store_survives_reload_and_preserves_record_history(tmp_path: Path) -> None:
    """Verify persisted revisions and creation attribution survive process replacement."""
    store = LicensingStore(root_path=tmp_path)
    created = store.create_definition(_test_definition(), editor_id="creator")
    replacement = _test_definition().model_copy(update={"notes": "Reviewed again"})
    store.replace_definition(replacement, editor_id="reviewer")

    reloaded_store = LicensingStore(root_path=tmp_path, writable=False)
    reloaded = reloaded_store.get_definition(created.license_id)

    assert reloaded is not None
    assert reloaded.notes == "Reviewed again"
    assert reloaded.metadata.revision == 2
    assert reloaded.metadata.created_by == "creator"
    assert reloaded.metadata.updated_by == "reviewer"
    assert reloaded_store.metadata() == store.metadata()


def test_model_file_overrides_resolve_by_declared_file_name() -> None:
    """Verify consumers can resolve a file-specific assignment without custom merging."""
    override = _test_assignment()
    licensing = ModelLicensing(
        license_expression="NOASSERTION",
        commercial_use=PermissionStatus.UNKNOWN,
        redistribution=PermissionStatus.UNKNOWN,
        files={"weights.bin": override},
    )

    assert licensing.assignment_for_file("weights.bin") == override
    assert licensing.assignment_for_file("other.bin").license_expression == "NOASSERTION"


def test_model_rejects_an_override_for_a_file_it_does_not_distribute() -> None:
    """Verify a typo cannot make a restrictive file conclusion silently ineffective."""
    licensing = ModelLicensing(
        license_expression="NOASSERTION",
        commercial_use=PermissionStatus.UNKNOWN,
        redistribution=PermissionStatus.UNKNOWN,
        files={"misspelled.bin": _test_assignment()},
    )

    with pytest.raises(ValidationError, match="not declared downloads"):
        GenericModelRecord(
            name="licensed-model",
            record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
            ),
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(
                        file_name="weights.bin",
                        file_url="https://example.invalid/weights.bin",
                    ),
                ],
            ),
            licensing=licensing,
        )


def test_nested_licensing_change_is_always_critical_for_review() -> None:
    """Verify a permission change receives critical review even in a category without custom critical fields."""
    assignment_payload = _test_assignment().model_dump(mode="json")
    before = {
        "name": "reviewed-model",
        "description": "Original wording",
        "licensing": assignment_payload,
    }
    after = {
        **before,
        "description": "Updated wording",
        "licensing": {
            **assignment_payload,
            "commercial_use": "prohibited",
        },
    }

    licensing_diffs = compute_field_diffs(before, after)
    prose_diffs = compute_field_diffs(before, {**before, "description": "Updated wording"})

    assert has_critical_changes(MODEL_REFERENCE_CATEGORY.miscellaneous, licensing_diffs)
    assert not has_critical_changes(MODEL_REFERENCE_CATEGORY.miscellaneous, prose_diffs)


def test_replica_refresh_hydrates_auxiliary_records_from_export(tmp_path: Path) -> None:
    """Verify a read-only replica can replace its packaged snapshot from PRIMARY."""
    definition = _test_definition()
    store = LicensingStore(root_path=tmp_path, writable=False)
    store.refresh_replica_export(
        {
            "metadata": {"schema_version": 1, "revision": 7, "updated_at": 123},
            "licenses": [definition.model_dump(mode="json")],
            "assets": [
                {
                    "asset_kind": "other",
                    "asset_identifier": "replicated-asset",
                    "display_name": "Replicated Asset",
                    "licensing": ModelLicensing(
                        **_test_assignment().model_dump(mode="python"),
                    ).model_dump(mode="json"),
                },
                {
                    "asset_kind": "model",
                    "asset_identifier": "miscellaneous:embedded-model",
                    "display_name": "Embedded Model",
                    "licensing": unknown_model_licensing().model_dump(mode="json"),
                },
            ],
        },
    )

    assert store.metadata().revision == 7
    assert store.get_definition(definition.license_id) == definition
    assert len(store.list_assets()) == 1
    assert store.list_assets()[0].asset_identifier == "replicated-asset"
    with pytest.raises(PermissionError):
        store.delete_definition(definition.license_id, editor_id="replica")


def test_replica_refresh_rejects_an_orphaned_snapshot_atomically(tmp_path: Path) -> None:
    """Verify a malformed PRIMARY snapshot cannot replace the last coherent replica cache."""
    writable_store = LicensingStore(root_path=tmp_path)
    original_definition = writable_store.create_definition(_test_definition(), editor_id="seed")
    replica_store = LicensingStore(root_path=tmp_path, writable=False)
    invalid_assignment = LicenseAssignment(
        license_expression="LicenseRef-Missing",
        license_ids=("LicenseRef-Missing",),
        commercial_use=PermissionStatus.UNKNOWN,
        redistribution=PermissionStatus.UNKNOWN,
    )

    with pytest.raises(ValueError, match="Unknown license definition"):
        replica_store.refresh_replica_export(
            {
                "metadata": {"schema_version": 1, "revision": 99},
                "licenses": [original_definition.model_dump(mode="json")],
                "assets": [
                    {
                        "asset_kind": "other",
                        "asset_identifier": "orphaned-snapshot-asset",
                        "display_name": "Orphaned Snapshot Asset",
                        "licensing": invalid_assignment.model_dump(mode="json"),
                    },
                ],
            },
        )

    assert replica_store.metadata().revision == writable_store.metadata().revision
    assert replica_store.get_definition(original_definition.license_id) == original_definition
    assert replica_store.list_assets() == []


def test_report_generation_is_stable_across_input_order_and_surfaces_conditions(tmp_path: Path) -> None:
    """Verify report meaning and order do not depend on backend dictionary ordering."""
    definition = _test_definition()
    conditional_definition = LicenseDefinition(
        license_id="LicenseRef-Test-Conditional",
        name="Conditional Test License",
        canonical_url="https://example.invalid/licenses/test-conditional",
        commercial_use=PermissionStatus.ALLOWED_WITH_CONDITIONS,
        redistribution=PermissionStatus.ALLOWED_WITH_CONDITIONS,
    )
    conditional_assignment = LicenseAssignment(
        license_expression="LicenseRef-Test-Conditional",
        license_ids=("LicenseRef-Test-Conditional",),
        commercial_use=PermissionStatus.ALLOWED_WITH_CONDITIONS,
        redistribution=PermissionStatus.ALLOWED_WITH_CONDITIONS,
    )
    first_export = {
        "schema_version": 1,
        "metadata": {"schema_version": 1, "revision": 2},
        "licenses": [conditional_definition.model_dump(mode="json"), definition.model_dump(mode="json")],
        "assets": [
            {
                "asset_kind": "other",
                "asset_identifier": "z-conditional",
                "display_name": "Z Conditional Asset",
                "licensing": conditional_assignment.model_dump(mode="json"),
            },
            {
                "asset_kind": "other",
                "asset_identifier": "a-permissive",
                "display_name": "A Permissive Asset",
                "licensing": _test_assignment().model_dump(mode="json"),
            },
        ],
    }
    reversed_export = {
        **first_export,
        "licenses": list(reversed(first_export["licenses"])),
        "assets": list(reversed(first_export["assets"])),
    }

    first_paths = generate_license_report(first_export, output_directory=tmp_path)
    first_contents = {path.name: path.read_text(encoding="utf-8") for path in first_paths}
    second_paths = generate_license_report(reversed_export, output_directory=tmp_path)
    second_contents = {path.name: path.read_text(encoding="utf-8") for path in second_paths}

    assert first_contents == second_contents
    assert first_contents["README.md"].startswith("<!-- GENERATED FILE:")
    assert first_contents["README.md"].index("LicenseRef-Test-Conditional") < first_contents["README.md"].index(
        "LicenseRef-Test-Permissive",
    )
    assert "Allowed With Conditions" in first_contents["custom_nodes.md"]
    assert first_contents["custom_nodes.md"].index("A Permissive Asset") < first_contents["custom_nodes.md"].index(
        "Z Conditional Asset",
    )
