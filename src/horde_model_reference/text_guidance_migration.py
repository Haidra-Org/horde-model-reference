"""Synthesize reviewable guidance proposals from legacy instruct-format labels.

The legacy text reference carries one free-text ``instruct_format`` label per model. Turning those
labels into first-class prompt contracts is the same computation whether it is previewed through the
v2 API or produced offline by a seeding script, so the heuristic lives here rather than in a router.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel

from horde_model_reference.text_guidance import (
    GuidanceAssignmentChange,
    GuidanceProfileChange,
    TextGuidanceAssignment,
    TextGuidanceChangeSet,
    TextInteractionMode,
    TextPromptContract,
    TextUsageProfile,
)

__all__ = [
    "MIGRATION_CHANGE_SET_TITLE",
    "GuidanceMigrationPreview",
    "build_legacy_migration_change_set",
]

MIGRATION_CHANGE_SET_TITLE = "Migrate legacy instruct-format guidance"
"""Title carried by every proposal built from legacy labels."""

_PROFILE_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class GuidanceMigrationPreview(BaseModel):
    """Reviewable proposal synthesized from legacy instruct-format labels."""

    change_set: TextGuidanceChangeSet | None
    source_model_count: int
    format_count: int


def build_legacy_migration_change_set(
    records: Mapping[str, Mapping[str, object]],
    *,
    existing_profiles: Sequence[TextUsageProfile],
    current_assignment: Callable[[str], TextGuidanceAssignment | None],
) -> GuidanceMigrationPreview:
    """Propose one prompt contract per distinct legacy label and assign unguided models to it.

    Labels are matched against the display names and aliases of published profiles so a rerun after
    an earlier migration reuses contracts instead of duplicating them. Models that already carry an
    explicit assignment are left untouched, which makes repeated runs converge on no change.

    Args:
        records: Canonical text records keyed by exact model name.
        existing_profiles: Published profiles, including deprecated ones, used for alias reuse.
        current_assignment: Lookup returning the stored assignment for one exact model name.

    Returns:
        The proposal, whose ``change_set`` is ``None`` when nothing is left to migrate.
    """
    known_aliases = {
        alias.casefold(): profile.profile_id
        for profile in existing_profiles
        for alias in [profile.display_name, *profile.aliases]
    }
    profiles_by_label: dict[str, TextPromptContract] = {}
    assignments: list[GuidanceAssignmentChange] = []

    for model_name, record in records.items():
        raw_label = record.get("instruct_format")
        if not isinstance(raw_label, str) or not raw_label.strip():
            continue
        label = raw_label.strip()
        profile_id = known_aliases.get(label.casefold())
        if profile_id is None:
            slug = _PROFILE_SLUG_PATTERN.sub("-", label.casefold()).strip("-") or "prompt-contract"
            profile_id = slug
            suffix = 2
            reserved_ids = {profile.profile_id for profile in existing_profiles} | set(profiles_by_label)
            while profile_id in reserved_ids:
                profile_id = f"{slug}-{suffix}"
                suffix += 1
            profiles_by_label[profile_id] = TextPromptContract(
                profile_id=profile_id,
                display_name=label,
                aliases=[label],
                summary=f"Review and document the legacy {label} prompt contract.",
                interaction_modes=[TextInteractionMode.INSTRUCTION],
            )
            known_aliases[label.casefold()] = profile_id
        if current_assignment(model_name) is None:
            assignments.append(
                GuidanceAssignmentChange(
                    model_name=model_name,
                    assignment=TextGuidanceAssignment(model_name=model_name, primary_profile_id=profile_id),
                    expected_before=None,
                ),
            )

    profile_changes = [
        GuidanceProfileChange(
            operation="create",
            profile_id=profile.profile_id,
            profile=profile,
            expected_before=None,
        )
        for profile in profiles_by_label.values()
    ]
    change_set = (
        TextGuidanceChangeSet(
            title=MIGRATION_CHANGE_SET_TITLE,
            profile_changes=profile_changes,
            assignment_changes=assignments,
        )
        if profile_changes or assignments
        else None
    )
    return GuidanceMigrationPreview(
        change_set=change_set,
        source_model_count=len(assignments),
        format_count=len(profile_changes),
    )
