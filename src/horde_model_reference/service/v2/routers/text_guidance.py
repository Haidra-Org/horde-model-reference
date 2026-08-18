"""Public and queued-management routes for reusable text-model guidance."""

from __future__ import annotations

import re
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel

from horde_model_reference import ModelReferenceManager
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue import PendingChangeRecord, PendingResourceKind
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
)
from horde_model_reference.service.v2.routers.write_validations import assert_primary_write_enabled
from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix
from horde_model_reference.text_guidance import (
    GuidanceAssignmentChange,
    GuidanceProfileChange,
    GuidanceProfileKind,
    ResolvedTextGuidance,
    TextGuidanceAssignment,
    TextGuidanceCatalog,
    TextGuidanceCatalogMetadata,
    TextGuidanceChangeSet,
    TextInteractionMode,
    TextPromptContract,
    TextUsageProfile,
)
from horde_model_reference.text_model_write_processor import get_valid_settings_keys

router = APIRouter(prefix="/text_generation/guidance")
_PROFILE_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class TextUsageProfileSummary(BaseModel):
    """Compact profile row for catalog indexes."""

    profile_id: str
    kind: GuidanceProfileKind
    display_name: str
    summary: str
    aliases: list[str]
    deprecated: bool
    assigned_model_count: int


class TextUsageProfilePage(BaseModel):
    """Published guidance profile index and catalog metadata."""

    items: list[TextUsageProfileSummary]
    total: int
    metadata: TextGuidanceCatalogMetadata


class TextGuidanceAssignmentPage(BaseModel):
    """Published explicit assignment index."""

    items: list[TextGuidanceAssignment]
    total: int
    metadata: TextGuidanceCatalogMetadata


class GuidanceMigrationPreview(BaseModel):
    """Reviewable proposal synthesized from legacy instruct-format labels."""

    change_set: TextGuidanceChangeSet | None
    source_model_count: int
    format_count: int


def _canonical_text_models(manager: ModelReferenceManager) -> dict[str, dict[str, object]]:
    """Return canonical v2-shaped text records without backend projections."""
    records = manager.get_raw_model_reference_json(MODEL_REFERENCE_CATEGORY.text_generation) or {}
    return {
        model_name: record
        for model_name, record in records.items()
        if isinstance(record, dict) and not has_legacy_text_backend_prefix(model_name)
    }


def _set_catalog_etag(response: Response, metadata: TextGuidanceCatalogMetadata) -> None:
    """Expose a weak revision ETag for inexpensive consumer freshness checks."""
    response.headers["ETag"] = f'W/"text-guidance-{metadata.revision}"'


def _validate_change_set_settings(change_set: TextGuidanceChangeSet) -> None:
    """Reject unknown generation settings in profiles and AI Horde examples."""
    valid_keys = set(get_valid_settings_keys())
    for change in change_set.profile_changes:
        if change.profile is None:
            continue
        settings_groups = [change.profile.recommended_settings]
        settings_groups.extend(example.parameters for example in change.profile.ai_horde_examples)
        invalid = sorted({key for settings in settings_groups for key in settings if key not in valid_keys})
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unknown text generation settings: {', '.join(invalid)}",
            )


@router.get("/profiles", response_model=TextUsageProfilePage, tags=["text_guidance"])
def list_profiles(
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    include_deprecated: bool = False,
) -> TextUsageProfilePage:
    """List published prompt contracts and usage recipes."""
    profiles = manager.text_guidance_store.list_profiles(include_deprecated=include_deprecated)
    assignments = manager.text_guidance_store.list_assignments()
    counts: dict[str, int] = {}
    for assignment in assignments:
        for profile_id in [assignment.primary_profile_id, *assignment.supplemental_profile_ids]:
            counts[profile_id] = counts.get(profile_id, 0) + 1
    metadata = manager.text_guidance_store.metadata()
    _set_catalog_etag(response, metadata)
    return TextUsageProfilePage(
        items=[
            TextUsageProfileSummary(
                profile_id=profile.profile_id,
                kind=profile.kind,
                display_name=profile.display_name,
                summary=profile.summary,
                aliases=profile.aliases,
                deprecated=profile.deprecated,
                assigned_model_count=counts.get(profile.profile_id, 0),
            )
            for profile in profiles
        ],
        total=len(profiles),
        metadata=metadata,
    )


@router.get("/profiles/{profile_id}", response_model=TextUsageProfile, tags=["text_guidance"])
def get_profile(
    profile_id: str,
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> TextUsageProfile:
    """Return one current published guidance profile."""
    profile = manager.text_guidance_store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Guidance profile '{profile_id}' not found."
        )
    _set_catalog_etag(response, manager.text_guidance_store.metadata())
    return profile


@router.get("/assignments", response_model=TextGuidanceAssignmentPage, tags=["text_guidance"])
def list_assignments(
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    model_name: Annotated[str | None, Query()] = None,
    profile_id: Annotated[str | None, Query()] = None,
) -> TextGuidanceAssignmentPage:
    """List explicit exact-model guidance assignments."""
    assignments = manager.text_guidance_store.list_assignments()
    if model_name is not None:
        assignments = [assignment for assignment in assignments if assignment.model_name == model_name]
    if profile_id is not None:
        assignments = [
            assignment
            for assignment in assignments
            if profile_id in [assignment.primary_profile_id, *assignment.supplemental_profile_ids]
        ]
    metadata = manager.text_guidance_store.metadata()
    _set_catalog_etag(response, metadata)
    return TextGuidanceAssignmentPage(items=assignments, total=len(assignments), metadata=metadata)


@router.get("/model", response_model=ResolvedTextGuidance, tags=["text_guidance"])
def resolve_model_guidance(
    name: Annotated[str, Query(min_length=1)],
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ResolvedTextGuidance:
    """Resolve published guidance and legacy fallback for one exact model."""
    record = _canonical_text_models(manager).get(name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Text model '{name}' not found.")
    legacy_format = record.get("instruct_format")
    resolved = manager.text_guidance_store.resolve(
        name,
        legacy_instruct_format=legacy_format if isinstance(legacy_format, str) else None,
    )
    _set_catalog_etag(response, resolved.catalog_metadata)
    return resolved


@router.get("/export", response_model=TextGuidanceCatalog, tags=["text_guidance"])
def export_catalog(
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> TextGuidanceCatalog:
    """Export the complete validated published guidance catalog."""
    catalog = manager.text_guidance_store.export()
    _set_catalog_etag(response, catalog.metadata)
    return catalog


@router.post("/migration/preview", response_model=GuidanceMigrationPreview, tags=["text_guidance"])
async def preview_legacy_migration(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GuidanceMigrationPreview:
    """Build an editable proposal from legacy instruct-format strings without storing it."""
    await authenticate_queue_requestor(apikey)
    records = _canonical_text_models(manager)
    existing_profiles = manager.text_guidance_store.list_profiles(include_deprecated=True)
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
        current_assignment = manager.text_guidance_store.get_assignment(model_name)
        if current_assignment is None:
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
            title="Migrate legacy instruct-format guidance",
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


@router.post(
    "/change-sets",
    response_model=PendingChangeRecord,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["text_guidance"],
)
async def submit_change_set(
    change_set: TextGuidanceChangeSet,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> PendingChangeRecord:
    """Validate and enqueue one coherent guidance proposal in the shared queue."""
    requestor = await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)
    _validate_change_set_settings(change_set)
    manager.text_guidance_store.preview_change_set(
        change_set,
        canonical_model_names=set(_canonical_text_models(manager)),
    )
    resource_id = str(uuid4())
    return require_pending_queue_service(manager).enqueue_change(
        category=MODEL_REFERENCE_CATEGORY.text_generation,
        model_name=f"guidance:{resource_id}",
        operation=AuditOperation.UPDATE,
        payload=change_set.model_dump(mode="json"),
        requestor_id=requestor.user_id,
        requestor_username=requestor.username,
        notes=change_set.title,
        request_metadata={
            "route": "submit_text_guidance_change_set",
            "title": change_set.title,
            "affected_models": [change.model_name for change in change_set.assignment_changes],
        },
        related_models=[change.model_name for change in change_set.assignment_changes],
        resource_kind=PendingResourceKind.TEXT_GUIDANCE,
        resource_id=resource_id,
    )
