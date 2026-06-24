"""Write validation helpers for v2 API create and update operations."""

from __future__ import annotations

from horde_model_reference import CanonicalFormat, ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import assert_canonical_write_enabled, assert_primary_mode


def assert_v2_write_enabled(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY | str | None = None,
) -> None:
    """Ensure writes are only attempted when the canonical v2 PRIMARY backend supports them.

    Passing *category* exempts a category with no legacy representation (``has_legacy_format=False``) from the
    v2 canonical-format requirement, since v2 is its only possible write path.
    """
    assert_canonical_write_enabled(manager, canonical_format=CanonicalFormat.v2, category=category)


def assert_primary_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure the backend is PRIMARY, without requiring a specific canonical format.

    Use for text utility metadata operations (schemas, aliases, families) that
    are v2-only endpoints but write to auxiliary stores, not model records.
    """
    assert_primary_mode(manager)
