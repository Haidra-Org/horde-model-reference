"""Write validation helpers for v2 API create and update operations."""

from __future__ import annotations

from horde_model_reference import CanonicalFormat, ModelReferenceManager
from horde_model_reference.service.shared import assert_canonical_write_enabled


def assert_v2_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure writes are only attempted when canonical v2 PRIMARY backend supports them."""
    assert_canonical_write_enabled(manager, canonical_format=CanonicalFormat.v2)
