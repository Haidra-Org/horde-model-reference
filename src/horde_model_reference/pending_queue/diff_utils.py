"""Shared diff utilities for pending queue and audit trail.

This module provides field-level diff computation and related models
that are reused across both pending change preview diffs and applied
batch net-change analysis.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from strenum import StrEnum

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class NetChangeType(StrEnum):
    """Type of net change for a model across a batch or pending change."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class FieldChangeType(StrEnum):
    """Type of field-level change."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class FieldDiff(BaseModel):
    """Field-level difference between before and after states."""

    field_path: str
    old_value: Any | None = None
    new_value: Any | None = None
    change_type: FieldChangeType


# Critical fields per category - changes to these fields are flagged as critical
CRITICAL_FIELDS_BY_CATEGORY: dict[MODEL_REFERENCE_CATEGORY, set[str]] = {
    MODEL_REFERENCE_CATEGORY.image_generation: {"baseline", "nsfw", "inpainting", "trigger", "homepage"},
    MODEL_REFERENCE_CATEGORY.text_generation: {"baseline", "nsfw", "url", "parameters"},
    MODEL_REFERENCE_CATEGORY.controlnet: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.blip: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.clip: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.codeformer: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.esrgan: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.gfpgan: {"style", "nsfw"},
    MODEL_REFERENCE_CATEGORY.safety_checker: {"style", "nsfw"},
}

# Fields that contain download URLs across categories
DOWNLOAD_URL_FIELDS = {"config.download"}


def compute_field_diffs(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    prefix: str = "",
    recursive: bool = True,
) -> list[FieldDiff]:
    """Compute field-level differences between two states.

    Args:
        before: The original/current state (None if model doesn't exist).
        after: The proposed/new state (None if model is being deleted).
        prefix: Path prefix for nested field tracking (used internally).
        recursive: If True, recursively diff nested dicts for granular changes.

    Returns:
        List of FieldDiff objects describing each changed field.
    """
    diffs: list[FieldDiff] = []

    if before is None and after is None:
        return diffs

    if before is None:
        # All fields in after are additions
        for key, value in (after or {}).items():
            field_path = f"{prefix}{key}" if prefix else key
            if recursive and isinstance(value, dict):
                # Recursively add nested fields
                diffs.extend(compute_field_diffs(None, value, prefix=f"{field_path}.", recursive=True))
            else:
                diffs.append(
                    FieldDiff(
                        field_path=field_path,
                        old_value=None,
                        new_value=value,
                        change_type=FieldChangeType.ADDED,
                    )
                )
        return diffs

    if after is None:
        # All fields in before are removals
        for key, value in before.items():
            field_path = f"{prefix}{key}" if prefix else key
            if recursive and isinstance(value, dict):
                # Recursively remove nested fields
                diffs.extend(compute_field_diffs(value, None, prefix=f"{field_path}.", recursive=True))
            else:
                diffs.append(
                    FieldDiff(
                        field_path=field_path,
                        old_value=value,
                        new_value=None,
                        change_type=FieldChangeType.REMOVED,
                    )
                )
        return diffs

    # Both states exist - find modifications
    all_keys = set(before.keys()) | set(after.keys())
    for key in sorted(all_keys):
        old_val = before.get(key)
        new_val = after.get(key)
        field_path = f"{prefix}{key}" if prefix else key

        if old_val == new_val:
            continue

        if key not in before:
            # Field added
            if recursive and isinstance(new_val, dict):
                diffs.extend(compute_field_diffs(None, new_val, prefix=f"{field_path}.", recursive=True))
            else:
                diffs.append(
                    FieldDiff(
                        field_path=field_path,
                        old_value=None,
                        new_value=new_val,
                        change_type=FieldChangeType.ADDED,
                    )
                )
        elif key not in after:
            # Field removed
            if recursive and isinstance(old_val, dict):
                diffs.extend(compute_field_diffs(old_val, None, prefix=f"{field_path}.", recursive=True))
            else:
                diffs.append(
                    FieldDiff(
                        field_path=field_path,
                        old_value=old_val,
                        new_value=None,
                        change_type=FieldChangeType.REMOVED,
                    )
                )
        elif recursive and isinstance(old_val, dict) and isinstance(new_val, dict):
            # Both are dicts - recurse for nested changes
            diffs.extend(compute_field_diffs(old_val, new_val, prefix=f"{field_path}.", recursive=True))
        else:
            # Field modified (or type changed from/to dict)
            diffs.append(
                FieldDiff(
                    field_path=field_path,
                    old_value=old_val,
                    new_value=new_val,
                    change_type=FieldChangeType.MODIFIED,
                )
            )

    return diffs


def has_critical_changes(category: MODEL_REFERENCE_CATEGORY, diffs: list[FieldDiff]) -> bool:
    """Check if any field diffs involve critical fields for the category.

    Args:
        category: The model reference category being modified.
        diffs: List of field diffs to check.

    Returns:
        True if any diff involves a critical field, False otherwise.
    """
    critical_fields = CRITICAL_FIELDS_BY_CATEGORY.get(category, set())
    if not critical_fields:
        return False

    for diff in diffs:
        # Check direct field matches
        if diff.field_path in critical_fields:
            return True

        # Check nested field matches (e.g., field_path="config.download.url" matches "config.download")
        for critical_path in critical_fields:
            if diff.field_path.startswith(f"{critical_path}.") or diff.field_path == critical_path:
                return True

    # Check download URL fields if defined
    for diff in diffs:
        for download_field in DOWNLOAD_URL_FIELDS:
            if diff.field_path.startswith(download_field):
                return True

    return False


def categorize_field_diffs(diffs: list[FieldDiff]) -> tuple[list[str], list[str], list[str]]:
    """Categorize field diffs by change type.

    Args:
        diffs: List of field diffs to categorize.

    Returns:
        Tuple of (fields_added, fields_removed, fields_modified) lists.
    """
    fields_added: list[str] = []
    fields_removed: list[str] = []
    fields_modified: list[str] = []

    for diff in diffs:
        if diff.change_type == FieldChangeType.ADDED:
            fields_added.append(diff.field_path)
        elif diff.change_type == FieldChangeType.REMOVED:
            fields_removed.append(diff.field_path)
        else:
            fields_modified.append(diff.field_path)

    return fields_added, fields_removed, fields_modified
