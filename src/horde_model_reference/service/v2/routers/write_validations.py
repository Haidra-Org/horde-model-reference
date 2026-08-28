"""Write validation helpers for v2 API create and update operations."""

from __future__ import annotations

from fastapi import HTTPException, status

from horde_model_reference import CanonicalFormat, ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord, ImageGenerationModelRecord
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


def assert_known_image_baseline(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY | str,
    model_record: GenericModelRecord,
) -> None:
    """Reject an image generation submission whose baseline is not a known one.

    Record parsing keeps an unknown baseline so that a reader on an older baseline vocabulary can still
    load the reference, which leaves the authoritative source as the only place able to tell a new
    baseline apart from a typo.

    Raises:
        HTTPException: 422 if the record names a baseline the served catalog does not publish.

    """
    if category != MODEL_REFERENCE_CATEGORY.image_generation:
        return
    if not isinstance(model_record, ImageGenerationModelRecord):
        return
    if manager.image_baseline_store.get(str(model_record.baseline)) is not None:
        return

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail=(
            f"Unknown baseline '{model_record.baseline}'. Publish it through "
            "/model_references/v2/image_generation/baselines/change-sets before submitting models that use it."
        ),
    )


def assert_primary_write_enabled(manager: ModelReferenceManager) -> None:
    """Ensure the backend is PRIMARY, without requiring a specific canonical format.

    Use for text utility metadata operations (schemas, aliases, families) that
    are v2-only endpoints but write to auxiliary stores, not model records.
    """
    assert_primary_mode(manager)
