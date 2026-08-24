"""Behavioral contracts for the reusable text-guidance catalog."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from horde_model_reference.cli.text_guidance_report import generate_text_guidance_report
from horde_model_reference.legacy.text_csv_utils import (
    TextCSVRow,
    csv_rows_to_legacy_dict,
    parse_legacy_text_csv_file,
    write_legacy_text_csv,
)
from horde_model_reference.text_guidance import (
    GuidanceAssignmentChange,
    GuidanceProfileChange,
    GuidanceProfileKind,
    TextGuidanceAssignment,
    TextGuidanceChangeSet,
    TextGuidanceStatus,
    TextInteractionMode,
    TextPromptContract,
    TextUsageRecipe,
)
from horde_model_reference.text_guidance_store import GuidanceConflictError, TextGuidanceStore


def _contract(profile_id: str = "chatml") -> TextPromptContract:
    return TextPromptContract(
        profile_id=profile_id,
        display_name="ChatML",
        aliases=["chatml"],
        summary="Serialize chat messages with explicit role markers.",
        interaction_modes=[TextInteractionMode.CHAT],
        accepted_roles=["system", "user", "assistant"],
    )


def _recipe() -> TextUsageRecipe:
    return TextUsageRecipe(
        profile_id="roleplay",
        display_name="Role-play",
        summary="Keep character and scene constraints in the system message.",
    )


def _create_catalog_change() -> TextGuidanceChangeSet:
    return TextGuidanceChangeSet(
        title="Document shared ChatML guidance",
        profile_changes=[
            GuidanceProfileChange(operation="create", profile_id="chatml", profile=_contract()),
            GuidanceProfileChange(operation="create", profile_id="roleplay", profile=_recipe()),
        ],
        assignment_changes=[
            GuidanceAssignmentChange(
                model_name="org/model-a",
                assignment=TextGuidanceAssignment(
                    model_name="org/model-a",
                    primary_profile_id="chatml",
                    supplemental_profile_ids=["roleplay"],
                ),
            ),
            GuidanceAssignmentChange(
                model_name="org/model-b",
                assignment=TextGuidanceAssignment(model_name="org/model-b", primary_profile_id="chatml"),
            ),
        ],
    )


def test_profiles_are_reused_and_resolved_for_exact_models(tmp_path: Path) -> None:
    """One prompt contract can explain several records without making a family an execution target."""
    store = TextGuidanceStore(root_path=tmp_path)

    store.apply_change_set(
        _create_catalog_change(),
        canonical_model_names={"org/model-a", "org/model-b"},
        editor_id="maintainer-1",
    )

    first = store.resolve("org/model-a", legacy_instruct_format="legacy-label")
    second = store.resolve("org/model-b", legacy_instruct_format=None)

    assert first.summary.status is TextGuidanceStatus.PUBLISHED
    assert first.primary_profile is not None
    assert second.primary_profile is not None
    assert first.primary_profile.profile_id == second.primary_profile.profile_id == "chatml"
    assert [profile.profile_id for profile in first.supplemental_profiles] == ["roleplay"]
    assert second.supplemental_profiles == []


def test_failed_change_set_does_not_publish_its_valid_sibling_edits(tmp_path: Path) -> None:
    """A proposal is coherent: an invalid assignment cannot leave a newly created profile behind."""
    store = TextGuidanceStore(root_path=tmp_path)
    invalid = TextGuidanceChangeSet(
        title="Invalid coherent change",
        profile_changes=[GuidanceProfileChange(operation="create", profile_id="chatml", profile=_contract())],
        assignment_changes=[
            GuidanceAssignmentChange(
                model_name="missing/model",
                assignment=TextGuidanceAssignment(model_name="missing/model", primary_profile_id="chatml"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="Unknown canonical text model"):
        store.apply_change_set(invalid, canonical_model_names={"known/model"}, editor_id="maintainer-1")

    assert store.list_profiles(include_deprecated=True) == []
    assert store.list_assignments() == []
    assert store.metadata().revision == 1


def test_review_precondition_detects_a_changed_touched_record(tmp_path: Path) -> None:
    """Approval refuses a stale proposal when the exact record reviewed has changed."""
    store = TextGuidanceStore(root_path=tmp_path)
    store.apply_change_set(
        TextGuidanceChangeSet(
            title="Create contract",
            profile_changes=[GuidanceProfileChange(operation="create", profile_id="chatml", profile=_contract())],
        ),
        canonical_model_names=set(),
        editor_id="maintainer-1",
    )
    reviewed = store.get_profile("chatml")
    assert reviewed is not None
    newer = reviewed.model_copy(update={"summary": "A more accurate published explanation."})
    store.apply_change_set(
        TextGuidanceChangeSet(
            title="Publish intervening edit",
            profile_changes=[
                GuidanceProfileChange(
                    operation="update",
                    profile_id="chatml",
                    profile=newer,
                    expected_before=reviewed,
                ),
            ],
        ),
        canonical_model_names=set(),
        editor_id="maintainer-2",
    )

    stale = TextGuidanceChangeSet(
        title="Stale edit",
        profile_changes=[
            GuidanceProfileChange(
                operation="update",
                profile_id="chatml",
                profile=reviewed.model_copy(update={"display_name": "Chat Markup"}),
                expected_before=reviewed,
            ),
        ],
    )
    with pytest.raises(GuidanceConflictError, match="changed after review"):
        store.apply_change_set(stale, canonical_model_names=set(), editor_id="maintainer-3")

    published = store.get_profile("chatml")
    assert published is not None
    assert published.summary == "A more accurate published explanation."


def test_legacy_label_is_a_distinct_fallback_state(tmp_path: Path) -> None:
    """Consumers can distinguish curated guidance from a historical instruct-format label."""
    store = TextGuidanceStore(root_path=tmp_path)

    legacy = store.resolve("org/model", legacy_instruct_format="Alpaca")
    undocumented = store.resolve("org/other", legacy_instruct_format="  ")

    assert legacy.summary.status is TextGuidanceStatus.LEGACY_LABEL
    assert legacy.legacy_instruct_format == "Alpaca"
    assert undocumented.summary.status is TextGuidanceStatus.UNDOCUMENTED


def test_raw_html_is_rejected_from_human_readable_guidance() -> None:
    """Catalog prose remains safe for renderers that accept CommonMark."""
    with pytest.raises(ValidationError, match="must not contain raw HTML"):
        TextPromptContract(
            profile_id="unsafe",
            display_name="Unsafe",
            summary="<script>alert('no')</script>",
            interaction_modes=[TextInteractionMode.CHAT],
        )


def test_profile_kind_controls_assignment_role(tmp_path: Path) -> None:
    """A usage recipe cannot silently stand in for the model's prompt serialization contract."""
    store = TextGuidanceStore(root_path=tmp_path)
    change = TextGuidanceChangeSet(
        title="Wrong profile role",
        profile_changes=[GuidanceProfileChange(operation="create", profile_id="roleplay", profile=_recipe())],
        assignment_changes=[
            GuidanceAssignmentChange(
                model_name="org/model",
                assignment=TextGuidanceAssignment(model_name="org/model", primary_profile_id="roleplay"),
            ),
        ],
    )

    with pytest.raises(ValueError, match="requires an active prompt contract"):
        store.preview_change_set(change, canonical_model_names={"org/model"})

    assert _recipe().kind is GuidanceProfileKind.USAGE_RECIPE


def test_durable_text_claims_survive_legacy_canonical_storage(tmp_path: Path) -> None:
    """The production LEGACY write path preserves first-class v2 claims instead of dropping them."""
    csv_path = tmp_path / "models.csv"
    context_window = {
        "maximum_tokens": 32768,
        "sources": [{"url": "https://publisher.test/model-card"}],
    }
    interaction_modes = {
        "chat": {
            "status": "supported",
            "sources": [{"url": "https://publisher.test/model-card"}],
        },
    }
    capabilities = {"tool_calling": {"status": "unknown", "sources": []}}
    write_legacy_text_csv(
        [
            TextCSVRow(
                name="publisher/model-7b",
                parameters_bn=7,
                parameters=7_000_000_000,
                description="",
                version="",
                style="",
                nsfw=False,
                baseline="llama",
                url="",
                tags=[],
                instruct_format="ChatML",
                settings=None,
                display_name="",
                context_window=context_window,
                interaction_modes=interaction_modes,
                capabilities=capabilities,
            ),
        ],
        csv_path,
    )

    parsed_rows, issues = parse_legacy_text_csv_file(csv_path)
    public_records = csv_rows_to_legacy_dict(parsed_rows, with_backend_prefixes=False)

    assert issues == []
    assert public_records["publisher/model-7b"]["context_window"] == context_window
    assert public_records["publisher/model-7b"]["interaction_modes"] == interaction_modes
    assert public_records["publisher/model-7b"]["capabilities"] == capabilities


def test_generated_markdown_communicates_reuse_without_becoming_source_data(tmp_path: Path) -> None:
    """The human artifact shows profile reuse and warns maintainers to edit structured data instead."""
    store = TextGuidanceStore(root_path=tmp_path / "guidance")
    store.apply_change_set(
        _create_catalog_change(),
        canonical_model_names={"org/model-a", "org/model-b"},
        editor_id="maintainer",
    )
    output_path = tmp_path / "TEXT_GUIDANCE.md"

    generate_text_guidance_report(store.export().model_dump(mode="json"), output_path=output_path)
    markdown = output_path.read_text(encoding="utf-8")

    assert "GENERATED FILE" in markdown
    assert "| `chatml` | Prompt Contract | 2 | Published |" in markdown
    assert "- org/model-a" in markdown
    assert "- org/model-b" in markdown
    assert "<a " not in markdown
