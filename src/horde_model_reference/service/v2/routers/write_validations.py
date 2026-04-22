"""Write validation helpers for v2 API create and update operations."""

from __future__ import annotations

from horde_model_reference import CanonicalFormat, ModelReferenceManager
from horde_model_reference.service.shared import assert_canonical_write_enabled, assert_primary_mode


def assert_v2_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure writes are only attempted when canonical v2 PRIMARY backend supports them."""
    assert_canonical_write_enabled(manager, canonical_format=CanonicalFormat.v2)


def assert_primary_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure the backend is PRIMARY, without requiring a specific canonical format.

    Use for text utility metadata operations (schemas, aliases, families) that
    are v2-only endpoints but write to auxiliary stores, not model records.
    """
    assert_primary_mode(manager)
