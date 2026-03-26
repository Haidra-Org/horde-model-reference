"""Validation helpers for v1 (legacy) write operations."""

from horde_model_reference import CanonicalFormat, ModelReferenceManager
from horde_model_reference.service.shared import assert_canonical_write_enabled


def assert_v1_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure writes only run when legacy canonical mode and backend allows them."""
    assert_canonical_write_enabled(manager, canonical_format=CanonicalFormat.LEGACY)
