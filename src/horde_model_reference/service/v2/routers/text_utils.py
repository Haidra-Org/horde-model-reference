"""Text model group utilities for the v2 API.

Exposes the backend's text model name parsing, group member retrieval,
and name composition as API endpoints for the frontend group editing UX.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from horde_model_reference import ModelReferenceManager
from horde_model_reference.analytics.text_model_parser import (
    get_base_model_name,
    infer_name_format,
    parse_text_model_name,
)
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.group_aliases import GroupAliasStore
from horde_model_reference.group_families import GroupFamilyStore, detect_families
from horde_model_reference.group_schema_store import GroupSchemaStore
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import TextModelGroupNameSchema
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
)
from horde_model_reference.service.v2.routers.write_validations import assert_primary_write_enabled
from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

router = APIRouter()

_CATEGORY = MODEL_REFERENCE_CATEGORY.text_generation

# Fields that can be shared/edited at group level
_COMMON_FIELD_KEYS = ("baseline", "description", "url", "nsfw", "tags", "style", "instruct_format")


class ExtraPartInfo(BaseModel):
    """A name segment that didn't match any primary part category."""

    value: str
    position: int
    inferred_type: str


class ParsedNameResponse(BaseModel):
    """Structured result of parsing a text model name."""

    original_name: str
    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    version: str | None = None
    suggested_group: str
    extras: list[ExtraPartInfo] = []


class ParsedNameInfo(BaseModel):
    """Parsed name components for a single group member."""

    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    version: str | None = None
    extras: list[ExtraPartInfo] = []


class GroupMemberInfo(BaseModel):
    """A single member of a text model group with parsed name info."""

    name: str
    parsed: ParsedNameInfo
    parameters: int | None = None
    baseline: str | None = None
    nsfw: bool | None = None
    description: str | None = None
    url: str | None = None
    style: str | None = None
    tags: list[str] | None = None
    display_name: str | None = None
    instruct_format: str | None = None
    is_backend_duplicate: bool = False
    backend_prefix: str | None = None


class NameFormatInfo(BaseModel):
    """Serializable representation of an inferred name format schema."""

    separator: str
    part_order: list[str]
    author_included: bool
    common_author: str | None = None
    template: str
    extra_parts: list[str] = []


class NameExceptionInfo(BaseModel):
    """A member that does not follow the group naming schema."""

    name: str
    reason: str


class GroupMembersResponse(BaseModel):
    """Full response for a text model group."""

    group_name: str
    members: list[GroupMemberInfo]
    common_fields: dict[str, Any]
    available_sizes: list[str]
    available_variants: list[str | None]
    available_quants: list[str | None]
    available_versions: list[str | None]
    size_usage: dict[str, int]
    variant_usage: dict[str, int]
    quant_usage: dict[str, int]
    name_format: NameFormatInfo
    canonical_count: int
    backend_duplicate_count: int
    name_schema_is_custom: bool = False
    exception_members: list[NameExceptionInfo] = []
    related_family: GroupFamilyResponse | None = None


class DistinctBaselinesResponse(BaseModel):
    """Response containing sorted unique baseline values for text models."""

    baselines: list[str]


class ComposeNameRequest(BaseModel):
    """Request body for composing a model name from parts."""

    author: str | None = None
    base_name: str
    size: str
    variant: str | None = None
    version: str | None = None
    quant: str | None = None
    separator: str | None = None
    part_order: list[str] | None = None
    extra_parts: dict[str, str] | None = None


class ComposeNameResponse(BaseModel):
    """Response from the name composition endpoint."""

    composed_name: str
    already_exists: bool
    suggested_group: str
    template: str
    rendered_example: str


class CommonFieldsUpdateRequest(BaseModel):
    """Request body for batch-updating common fields across a group."""

    baseline: str | None = None
    description: str | None = None
    url: str | None = None
    nsfw: bool | None = None
    tags: list[str] | None = None
    style: str | None = None
    instruct_format: str | None = None


class BatchUpdateResponse(BaseModel):
    """Response from batch group common field update."""

    updated_count: int
    batch_id: str
    pending_change_ids: list[int]


class GroupNameSchemaResponse(BaseModel):
    """Response for a group's naming schema."""

    group_name: str
    name_schema: TextModelGroupNameSchema
    is_custom: bool


class GroupNameSchemaUpdateRequest(BaseModel):
    """Partial update for a group's naming schema."""

    separator: str | None = None
    part_order: list[str] | None = None
    author_included: bool | None = None
    common_author: str | None = None
    template: str | None = None
    extra_parts: list[str] | None = None


class NameExceptionRequest(BaseModel):
    """Set or clear a name schema exception on a model."""

    reason: str | None


class GroupListResponse(BaseModel):
    """Response listing all text model group names."""

    groups: list[str]


class GroupHealthIssue(BaseModel):
    """A single health problem detected for a group."""

    group_name: str
    issue_type: str
    message: str
    severity: str = "warning"


class GroupSummaryEntry(BaseModel):
    """Enriched metadata for a single text model group."""

    group_name: str
    canonical_count: int
    backend_duplicate_count: int
    has_custom_schema: bool = False
    family_name: str | None = None
    alias_canonical: str | None = None
    aliases: list[str] = []
    available_sizes: list[str] = []
    health_issues: list[GroupHealthIssue] = []


class GroupsSummaryResponse(BaseModel):
    """Enriched overview of all text model groups."""

    groups: list[GroupSummaryEntry]
    total_groups: int
    total_models: int
    groups_with_families: int
    groups_with_aliases: int
    groups_with_issues: int


class GroupHealthResponse(BaseModel):
    """Aggregate health check across all text model groups."""

    issues: list[GroupHealthIssue]
    total_groups_checked: int
    groups_with_issues: int
    issue_counts_by_type: dict[str, int]


def _get_all_text_models(manager: ModelReferenceManager) -> dict[str, dict[str, Any]]:
    """Load all text generation models as raw dicts."""
    raw = manager.get_raw_model_reference_json(_CATEGORY)
    if raw is None:
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def _get_group_members(
    all_models: dict[str, dict[str, Any]],
    group_name: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Filter models belonging to a specific group."""
    members: list[tuple[str, dict[str, Any]]] = []
    for key, data in all_models.items():
        model_group = data.get("text_model_group")
        if model_group == group_name:
            members.append((key, data))
    return members


def _compute_common_fields(
    canonical_members: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Find fields that are identical across all canonical members."""
    if not canonical_members:
        return {}

    common: dict[str, Any] = {}
    for field in _COMMON_FIELD_KEYS:
        values = []
        for _, data in canonical_members:
            values.append(data.get(field))

        if len(values) > 0 and all(v == values[0] for v in values) and values[0] is not None:
            common[field] = values[0]

    return common


def _compose_name_from_parts(
    base_name: str,
    size: str,
    variant: str | None = None,
    version: str | None = None,
    quant: str | None = None,
    author: str | None = None,
    separator: str = "-",
    part_order: list[str] | None = None,
    extra_parts: dict[str, str] | None = None,
) -> str:
    """Compose a model name from structured parts.

    When part_order is provided, parts are arranged in that order, using separator.
    Otherwise uses default: [author/]base{sep}size[{sep}variant][{sep}version][{sep}quant]

    If part_order is provided but does not include "base", base_name is defensively
    prepended so callers cannot accidentally drop the base segment.

    extra_parts keys are typically prefixed with "extra:" (e.g. "extra:date") so they
    can be referenced by position in part_order alongside the standard parts.
    """
    available_parts = {
        "base": base_name,
        "size": size,
    }
    if variant:
        available_parts["variant"] = variant
    if version:
        available_parts["version"] = version
    if quant:
        available_parts["quant"] = quant
    if extra_parts:
        for k, v in extra_parts.items():
            if v:
                available_parts[k] = v

    if part_order:
        ordered = [available_parts[p] for p in part_order if p in available_parts]
        if "base" not in part_order and base_name:
            ordered.insert(0, base_name)
    else:
        ordered = [base_name, size]
        if variant:
            ordered.append(variant)
        if version:
            ordered.append(version)
        if quant:
            ordered.append(quant)
        if extra_parts:
            for v in extra_parts.values():
                if v:
                    ordered.append(v)

    model_name = separator.join(ordered)

    if author:
        return f"{author}/{model_name}"
    return model_name


def _build_template_and_example(
    base_name: str,
    size: str,
    variant: str | None,
    version: str | None,
    quant: str | None,
    author: str | None,
    separator: str,
    part_order: list[str] | None,
    extra_parts: dict[str, str] | None,
) -> tuple[str, str]:
    """Build a human-readable template string and a rendered example name.

    The template uses ``{placeholder}`` tokens reflecting the full effective part
    order (including optional parts the user has not filled in yet). The rendered
    example substitutes the supplied values and matches the composed_name exactly,
    omitting parts whose values are missing.
    """
    if part_order:
        effective_template = list(part_order)
        if "base" not in effective_template and base_name:
            effective_template.insert(0, "base")
    else:
        effective_template = ["base", "size"]
        if variant:
            effective_template.append("variant")
        if version:
            effective_template.append("version")
        if quant:
            effective_template.append("quant")
        if extra_parts:
            for k, v in extra_parts.items():
                if v:
                    effective_template.append(k)

    template_body = separator.join(f"{{{p}}}" for p in effective_template)
    template = f"{{author}}/{template_body}" if author else template_body

    supplied_values: dict[str, str] = {
        "base": base_name,
        "size": size,
    }
    if variant:
        supplied_values["variant"] = variant
    if version:
        supplied_values["version"] = version
    if quant:
        supplied_values["quant"] = quant
    if extra_parts:
        for k, v in extra_parts.items():
            if v:
                supplied_values[k] = v

    example_parts = [supplied_values[p] for p in effective_template if p in supplied_values]
    rendered_body = separator.join(example_parts)
    rendered_example = f"{author}/{rendered_body}" if author else rendered_body

    return template, rendered_example


@router.get(
    "/text_generation/parse_name",
    response_model=ParsedNameResponse,
    summary="Parse a text model name into structured components",
    tags=["text_utils"],
)
def parse_name(
    name: Annotated[str, Query(description="The model name to parse")],
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ParsedNameResponse:
    """Parse a text model name into base name, size, variant, and quantization components."""
    parsed = parse_text_model_name(name)
    suggested_group = get_base_model_name(name)

    alias_store = manager.group_alias_store
    if alias_store is not None:
        suggested_group = alias_store.resolve(suggested_group)

    return ParsedNameResponse(
        original_name=name,
        base_name=parsed.base_name,
        size=parsed.size,
        variant=parsed.variant,
        quant=parsed.quant,
        version=parsed.version,
        suggested_group=suggested_group,
        extras=[
            ExtraPartInfo(value=e.value, position=e.position, inferred_type=e.inferred_type.value)
            for e in parsed.extras
        ],
    )


@router.get(
    "/text_generation/group/{group_name}",
    response_model=GroupMembersResponse,
    summary="Get all members of a text model group",
    tags=["text_utils"],
)
def get_group(
    group_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupMembersResponse:
    """Get all models in a text model group with parsed name info and common fields."""
    all_models = _get_all_text_models(manager)
    raw_members = _get_group_members(all_models, group_name)

    if not raw_members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No models found in group '{group_name}'",
        )

    members: list[GroupMemberInfo] = []
    canonical_members: list[tuple[str, dict[str, Any]]] = []
    sizes: set[str] = set()
    variants: set[str | None] = set()
    quants: set[str | None] = set()
    versions: set[str | None] = set()
    size_usage: dict[str, int] = {}
    variant_usage: dict[str, int] = {}
    quant_usage: dict[str, int] = {}

    for key, data in raw_members:
        is_dup = has_legacy_text_backend_prefix(key)
        backend_prefix: str | None = None
        if is_dup:
            for prefix in ("aphrodite/", "koboldcpp/"):
                if key.startswith(prefix):
                    backend_prefix = prefix.rstrip("/")
                    break

        # Parse the canonical name (strip backend prefix for parsing)
        parse_target = key
        if backend_prefix:
            parse_target = key[len(backend_prefix) + 1 :]

        parsed = parse_text_model_name(parse_target)

        member = GroupMemberInfo(
            name=key,
            parsed=ParsedNameInfo(
                base_name=parsed.base_name,
                size=parsed.size,
                variant=parsed.variant,
                quant=parsed.quant,
                version=parsed.version,
                extras=[
                    ExtraPartInfo(value=e.value, position=e.position, inferred_type=e.inferred_type.value)
                    for e in parsed.extras
                ],
            ),
            parameters=data.get("parameters"),
            baseline=data.get("baseline"),
            nsfw=data.get("nsfw"),
            description=data.get("description"),
            url=data.get("url"),
            style=data.get("style"),
            tags=data.get("tags"),
            display_name=data.get("display_name"),
            instruct_format=data.get("instruct_format"),
            is_backend_duplicate=is_dup,
            backend_prefix=backend_prefix,
        )
        members.append(member)

        if not is_dup:
            canonical_members.append((key, data))
            if parsed.size:
                sizes.add(parsed.size)
                size_usage[parsed.size] = size_usage.get(parsed.size, 0) + 1
            variants.add(parsed.variant)
            if parsed.variant:
                variant_usage[parsed.variant] = variant_usage.get(parsed.variant, 0) + 1
            quants.add(parsed.quant)
            if parsed.quant:
                quant_usage[parsed.quant] = quant_usage.get(parsed.quant, 0) + 1
            versions.add(parsed.version)

    common_fields = _compute_common_fields(canonical_members)

    # Sort sizes numerically where possible
    def _size_sort_key(s: str) -> float:
        try:
            # Handle "8x7B" style MoE sizes
            if "x" in s.upper():
                parts = s.upper().replace("B", "").replace("M", "").replace("K", "").split("X")
                return float(parts[0]) * float(parts[1])
            numeric = s.upper().replace("B", "").replace("M", "").replace("K", "")
            multiplier = 1.0
            if s.upper().endswith("M"):
                multiplier = 0.001
            elif s.upper().endswith("K"):
                multiplier = 0.000001
            return float(numeric) * multiplier
        except (ValueError, IndexError):
            return 0.0

    sorted_sizes = sorted(sizes, key=_size_sort_key)
    sorted_variants = sorted(variants, key=lambda v: v or "")
    sorted_quants = sorted(quants, key=lambda q: q or "")
    sorted_versions = sorted(versions, key=lambda ver: ver or "")

    canonical_count = len(canonical_members)
    dup_count = len(members) - canonical_count

    # Use persisted schema if available, otherwise infer from member names
    name_schema_is_custom = False
    store = manager.group_schema_store
    persisted = store.get(group_name) if store else None
    if persisted:
        name_schema_is_custom = True
        name_format = NameFormatInfo(
            separator=persisted.separator,
            part_order=persisted.part_order,
            author_included=persisted.author_included,
            common_author=persisted.common_author,
            template=persisted.template or "",
            extra_parts=persisted.extra_parts,
        )
    else:
        canonical_names = [k for k, _ in canonical_members]
        schema = infer_name_format(canonical_names)
        name_format = NameFormatInfo(
            separator=schema.separator,
            part_order=schema.part_order,
            author_included=schema.author_included,
            common_author=schema.common_author,
            template=schema.template,
            extra_parts=schema.extra_parts,
        )

    # Collect members with naming schema exceptions
    exception_members: list[NameExceptionInfo] = []
    for key, data in canonical_members:
        reason = data.get("name_schema_exception")
        if reason:
            exception_members.append(NameExceptionInfo(name=key, reason=reason))

    # Look up related family for this group
    related_family: GroupFamilyResponse | None = None
    family_store = manager.group_family_store
    if family_store is not None:
        family = family_store.get_family_for_group(group_name)
        if family is not None:
            related_family = GroupFamilyResponse(
                family_name=family.family_name,
                members=family.members,
            )

    return GroupMembersResponse(
        group_name=group_name,
        members=members,
        common_fields=common_fields,
        available_sizes=sorted_sizes,
        available_variants=sorted_variants,
        available_quants=sorted_quants,
        available_versions=sorted_versions,
        size_usage=size_usage,
        variant_usage=variant_usage,
        quant_usage=quant_usage,
        name_format=name_format,
        canonical_count=canonical_count,
        backend_duplicate_count=dup_count,
        name_schema_is_custom=name_schema_is_custom,
        exception_members=exception_members,
        related_family=related_family,
    )


@router.get(
    "/text_generation/distinct_baselines",
    response_model=DistinctBaselinesResponse,
    summary="Get unique baseline values for text generation models",
    tags=["text_utils"],
)
def get_distinct_baselines(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> DistinctBaselinesResponse:
    """Return sorted unique non-empty baselines from text_generation models."""
    all_models = _get_all_text_models(manager)
    baselines = {
        baseline.strip()
        for data in all_models.values()
        if isinstance(data.get("baseline"), str)
        for baseline in [str(data.get("baseline"))]
        if baseline.strip()
    }
    return DistinctBaselinesResponse(baselines=sorted(baselines))


@router.post(
    "/text_generation/compose_name",
    response_model=ComposeNameResponse,
    summary="Compose a model name from structured parts and check for collisions",
    tags=["text_utils"],
)
def compose_name(
    request: ComposeNameRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ComposeNameResponse:
    """Compose a model name from base name, size, variant, and quant parts.

    Checks whether the composed name already exists in the text_generation category.
    """
    separator = request.separator or "-"
    composed = _compose_name_from_parts(
        base_name=request.base_name,
        size=request.size,
        variant=request.variant,
        version=request.version,
        quant=request.quant,
        author=request.author,
        separator=separator,
        part_order=request.part_order,
        extra_parts=request.extra_parts,
    )

    template, rendered_example = _build_template_and_example(
        base_name=request.base_name,
        size=request.size,
        variant=request.variant,
        version=request.version,
        quant=request.quant,
        author=request.author,
        separator=separator,
        part_order=request.part_order,
        extra_parts=request.extra_parts,
    )

    all_models = _get_all_text_models(manager)
    already_exists = composed in all_models

    suggested_group = get_base_model_name(composed)

    return ComposeNameResponse(
        composed_name=composed,
        already_exists=already_exists,
        suggested_group=suggested_group,
        template=template,
        rendered_example=rendered_example,
    )


@router.put(
    "/text_generation/group/{group_name}/common_fields",
    response_model=BatchUpdateResponse,
    summary="Batch-update common fields across all canonical members of a group",
    tags=["text_utils"],
)
async def update_group_common_fields(
    group_name: str,
    request: CommonFieldsUpdateRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update shared fields across all canonical members of a text model group.

    Creates one PendingChangeRecord per canonical member with a shared batch_id.
    """
    requestor = await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    all_models = _get_all_text_models(manager)
    raw_members = _get_group_members(all_models, group_name)

    if not raw_members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No models found in group '{group_name}'",
        )

    # Only update canonical (non-backend-prefixed) members
    canonical_members = [(k, d) for k, d in raw_members if not has_legacy_text_backend_prefix(k)]

    if not canonical_members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No canonical models found in group '{group_name}'",
        )

    # Build the field updates (only non-None fields from the request)
    updates = request.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update. Provide at least one field.",
        )

    import uuid

    batch_id = str(uuid.uuid4())
    queue_service = require_pending_queue_service(manager)
    pending_ids: list[int] = []

    for model_name, existing_data in canonical_members:
        # Merge updates into the existing model data
        merged = dict(existing_data)
        merged.update(updates)

        change = queue_service.enqueue_change(
            category=_CATEGORY,
            model_name=model_name,
            operation=AuditOperation.UPDATE,
            payload=merged,
            requestor_id=requestor.user_id,
            requestor_username=requestor.username,
            notes=f"Batch group update for '{group_name}'",
            request_metadata={"route": "update_group_common_fields", "batch_id": batch_id},
        )
        pending_ids.append(change.change_id)

    logger.info(
        f"Queued {len(pending_ids)} pending changes for group '{group_name}' "
        f"(batch_id={batch_id}, requestor={requestor.username})"
    )

    response = BatchUpdateResponse(
        updated_count=len(pending_ids),
        batch_id=batch_id,
        pending_change_ids=pending_ids,
    )
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=response.model_dump(mode="json"),
    )


@router.get(
    "/text_generation/groups",
    response_model=GroupListResponse,
    summary="List all text model group names",
    tags=["text_utils"],
)
def list_groups(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupListResponse:
    """Return sorted distinct ``text_model_group`` values across all text models."""
    all_models = _get_all_text_models(manager)
    groups: set[str] = set()
    for data in all_models.values():
        group = data.get("text_model_group")
        if isinstance(group, str) and group:
            groups.add(group)
    return GroupListResponse(groups=sorted(groups))


def _collect_group_health_issues(
    group_name: str,
    members: list[tuple[str, dict[str, Any]]],
) -> list[GroupHealthIssue]:
    """Check a group's canonical members for common problems.

    Returns a list of issues found. Called by both the summary and health
    endpoints to avoid duplicating diagnostic logic.
    """
    issues: list[GroupHealthIssue] = []

    if len(members) == 1:
        issues.append(
            GroupHealthIssue(
                group_name=group_name,
                issue_type="singleton_group",
                message="Group has only 1 canonical model — may not need a group",
            )
        )

    baselines: set[str] = set()
    nsfw_values: set[bool] = set()
    missing_desc_count = 0
    missing_baseline_count = 0

    for _, data in members:
        bl = data.get("baseline")
        if isinstance(bl, str) and bl:
            baselines.add(bl)
        else:
            missing_baseline_count += 1
        nsfw = data.get("nsfw")
        if isinstance(nsfw, bool):
            nsfw_values.add(nsfw)
        if not data.get("description"):
            missing_desc_count += 1

    if len(baselines) > 1:
        issues.append(
            GroupHealthIssue(
                group_name=group_name,
                issue_type="inconsistent_baseline",
                message=f"Members have different baselines: {', '.join(sorted(baselines))}",
            )
        )

    if len(nsfw_values) > 1:
        issues.append(
            GroupHealthIssue(
                group_name=group_name,
                issue_type="inconsistent_nsfw",
                message="Members have different NSFW flags",
            )
        )

    if missing_baseline_count > 0:
        issues.append(
            GroupHealthIssue(
                group_name=group_name,
                issue_type="missing_baseline",
                message=f"{missing_baseline_count} member(s) missing baseline",
            )
        )

    if missing_desc_count > 0:
        issues.append(
            GroupHealthIssue(
                group_name=group_name,
                issue_type="missing_description",
                message=f"{missing_desc_count} member(s) missing description",
                severity="info",
            )
        )

    return issues


@router.get(
    "/text_generation/groups/summary",
    response_model=GroupsSummaryResponse,
    summary="Enriched overview of all text model groups",
    tags=["text_utils"],
)
def list_groups_summary(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupsSummaryResponse:
    """Return per-group metadata including member counts, family/alias info, and health flags.

    Designed to power a group management overview UI in a single request.
    """
    all_models = _get_all_text_models(manager)

    # Bucket models by group
    groups_map: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, data in all_models.items():
        group = data.get("text_model_group")
        if isinstance(group, str) and group:
            groups_map.setdefault(group, []).append((key, data))

    schema_store = manager.group_schema_store
    alias_store = manager.group_alias_store
    family_store = manager.group_family_store

    total_models = 0
    groups_with_families = 0
    groups_with_aliases = 0
    groups_with_issues = 0
    entries: list[GroupSummaryEntry] = []

    for group_name in sorted(groups_map):
        raw_members = groups_map[group_name]
        canonical = [(k, d) for k, d in raw_members if not has_legacy_text_backend_prefix(k)]
        dups = len(raw_members) - len(canonical)
        total_models += len(raw_members)

        # Schema
        has_custom = bool(schema_store and schema_store.get(group_name))

        # Family
        family_name: str | None = None
        if family_store:
            fam = family_store.get_family_for_group(group_name)
            if fam:
                family_name = fam.family_name
                groups_with_families += 1

        # Aliases
        alias_canonical: str | None = None
        aliases: list[str] = []
        if alias_store:
            resolved = alias_store.resolve(group_name)
            if resolved != group_name:
                alias_canonical = resolved
                groups_with_aliases += 1
            else:
                alias_entry = alias_store.get(group_name)
                if alias_entry and alias_entry.aliases:
                    aliases = alias_entry.aliases
                    groups_with_aliases += 1

        # Sizes
        sizes: set[str] = set()
        for key, _ in canonical:
            parse_target = key
            for prefix in ("aphrodite/", "koboldcpp/"):
                if key.startswith(prefix):
                    parse_target = key[len(prefix) :]
                    break
            parsed = parse_text_model_name(parse_target)
            if parsed.size:
                sizes.add(parsed.size)

        # Health
        health_issues = _collect_group_health_issues(group_name, canonical)
        if health_issues:
            groups_with_issues += 1

        entries.append(
            GroupSummaryEntry(
                group_name=group_name,
                canonical_count=len(canonical),
                backend_duplicate_count=dups,
                has_custom_schema=has_custom,
                family_name=family_name,
                alias_canonical=alias_canonical,
                aliases=aliases,
                available_sizes=sorted(sizes),
                health_issues=health_issues,
            )
        )

    return GroupsSummaryResponse(
        groups=entries,
        total_groups=len(entries),
        total_models=total_models,
        groups_with_families=groups_with_families,
        groups_with_aliases=groups_with_aliases,
        groups_with_issues=groups_with_issues,
    )


@router.get(
    "/text_generation/groups/health",
    response_model=GroupHealthResponse,
    summary="Health check across all text model groups",
    tags=["text_utils"],
)
def check_groups_health(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupHealthResponse:
    """Scan all text model groups for common problems.

    Returns an aggregate list of issues sorted by severity, useful for
    admin triage dashboards.
    """
    all_models = _get_all_text_models(manager)

    groups_map: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, data in all_models.items():
        group = data.get("text_model_group")
        if isinstance(group, str) and group:
            groups_map.setdefault(group, []).append((key, data))

    all_issues: list[GroupHealthIssue] = []
    groups_with_issues_set: set[str] = set()
    issue_counts: dict[str, int] = {}

    for group_name in sorted(groups_map):
        raw_members = groups_map[group_name]
        canonical = [(k, d) for k, d in raw_members if not has_legacy_text_backend_prefix(k)]

        issues = _collect_group_health_issues(group_name, canonical)
        if issues:
            groups_with_issues_set.add(group_name)
            all_issues.extend(issues)
            for issue in issues:
                issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1

    return GroupHealthResponse(
        issues=all_issues,
        total_groups_checked=len(groups_map),
        groups_with_issues=len(groups_with_issues_set),
        issue_counts_by_type=issue_counts,
    )


def _require_group_schema_store(manager: ModelReferenceManager) -> GroupSchemaStore:
    """Return the group schema store or raise 503 if unavailable."""
    store = manager.group_schema_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Group schema storage is not available (backend may be in REPLICA mode).",
        )
    return store


@router.get(
    "/text_generation/group/{group_name}/name_schema",
    response_model=GroupNameSchemaResponse,
    summary="Get the naming schema for a text model group",
    tags=["text_utils"],
)
def get_group_name_schema(
    group_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupNameSchemaResponse:
    """Return the persisted naming schema if one exists, otherwise infer from member names."""
    store = manager.group_schema_store
    persisted = store.get(group_name) if store else None
    if persisted:
        return GroupNameSchemaResponse(
            group_name=group_name,
            name_schema=persisted,
            is_custom=True,
        )

    # Fall back to inferred schema
    all_models = _get_all_text_models(manager)
    raw_members = _get_group_members(all_models, group_name)
    canonical_names = [k for k, d in raw_members if not has_legacy_text_backend_prefix(k)]

    if not canonical_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No models found in group '{group_name}'",
        )

    inferred = infer_name_format(canonical_names)
    return GroupNameSchemaResponse(
        group_name=group_name,
        name_schema=TextModelGroupNameSchema(
            separator=inferred.separator,
            part_order=inferred.part_order,
            author_included=inferred.author_included,
            common_author=inferred.common_author,
            template=inferred.template,
        ),
        is_custom=False,
    )


@router.put(
    "/text_generation/group/{group_name}/name_schema",
    response_model=GroupNameSchemaResponse,
    summary="Save a custom naming schema for a text model group",
    tags=["text_utils"],
)
async def update_group_name_schema(
    group_name: str,
    request: GroupNameSchemaUpdateRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupNameSchemaResponse:
    """Persist a custom naming schema for a text model group."""
    requestor = await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_schema_store(manager)

    # Merge with existing persisted schema (or defaults) for partial updates
    existing = store.get(group_name) or TextModelGroupNameSchema()
    updates = request.model_dump(exclude_none=True)
    merged = existing.model_copy(update=updates)
    store.set(group_name, merged)

    logger.info(f"Updated naming schema for group '{group_name}' (requestor={requestor.username})")

    return GroupNameSchemaResponse(
        group_name=group_name,
        name_schema=merged,
        is_custom=True,
    )


@router.delete(
    "/text_generation/group/{group_name}/name_schema",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a custom naming schema (revert to inferred)",
    tags=["text_utils"],
)
async def delete_group_name_schema(
    group_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> None:
    """Remove the persisted naming schema so the group reverts to inference."""
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_schema_store(manager)
    deleted = store.delete(group_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No custom naming schema found for group '{group_name}'",
        )


@router.put(
    "/text_generation/{model_name:path}/name_exception",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Set or clear the naming schema exception flag on a model",
    tags=["text_utils"],
)
async def set_name_exception(
    model_name: str,
    request: NameExceptionRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Set or clear the ``name_schema_exception`` field on a text model.

    Pass ``{"reason": "some reason"}`` to flag the model or ``{"reason": null}`` to clear.
    Changes go through the pending queue.
    """
    requestor = await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    all_models = _get_all_text_models(manager)
    if model_name not in all_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in text_generation category",
        )

    existing_data = dict(all_models[model_name])
    if request.reason is not None:
        existing_data["name_schema_exception"] = request.reason
    else:
        existing_data.pop("name_schema_exception", None)

    queue_service = require_pending_queue_service(manager)
    change = queue_service.enqueue_change(
        category=_CATEGORY,
        model_name=model_name,
        operation=AuditOperation.UPDATE,
        payload=existing_data,
        requestor_id=requestor.user_id,
        requestor_username=requestor.username,
        notes=f"Name exception {'set' if request.reason else 'cleared'} for '{model_name}'",
        request_metadata={"route": "set_name_exception"},
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"pending_change_id": change.change_id},
    )


class GroupAliasResponse(BaseModel):
    """A single group alias entry."""

    canonical: str
    aliases: list[str]


class GroupAliasListResponse(BaseModel):
    """Response listing all group alias entries."""

    entries: list[GroupAliasResponse]


class SetAliasesRequest(BaseModel):
    """Request body for setting the full alias list for a canonical group."""

    aliases: list[str]


class AddAliasRequest(BaseModel):
    """Request body for adding a single alias to a canonical group."""

    alias: str


class RemoveAliasRequest(BaseModel):
    """Request body for removing a single alias from a canonical group."""

    alias: str


def _require_group_alias_store(manager: ModelReferenceManager) -> GroupAliasStore:
    """Return the group alias store or raise 503 if unavailable."""
    store = manager.group_alias_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Group alias storage is not available (backend may be in REPLICA mode).",
        )
    return store


@router.get(
    "/text_generation/aliases",
    response_model=GroupAliasListResponse,
    summary="List all group alias entries",
    tags=["text_utils"],
)
def list_aliases(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupAliasListResponse:
    """Return all configured group alias entries."""
    store = _require_group_alias_store(manager)
    all_entries = store.list_all()
    return GroupAliasListResponse(
        entries=[
            GroupAliasResponse(canonical=canonical, aliases=entry.aliases)
            for canonical, entry in sorted(all_entries.items())
        ],
    )


@router.get(
    "/text_generation/aliases/{canonical}",
    response_model=GroupAliasResponse,
    summary="Get aliases for a canonical group",
    tags=["text_utils"],
)
def get_alias(
    canonical: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupAliasResponse:
    """Return the alias entry for a specific canonical group name."""
    store = _require_group_alias_store(manager)
    entry = store.get(canonical)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No alias entry for canonical group '{canonical}'",
        )
    return GroupAliasResponse(canonical=entry.canonical, aliases=entry.aliases)


@router.put(
    "/text_generation/aliases/{canonical}",
    response_model=GroupAliasResponse,
    summary="Set the full alias list for a canonical group",
    tags=["text_utils"],
)
async def set_aliases(
    canonical: str,
    request: SetAliasesRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupAliasResponse:
    """Replace all aliases for a canonical group name.

    Raises 409 if any alias is already claimed by a different canonical group.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_alias_store(manager)
    try:
        store.set_aliases(canonical, request.aliases)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    logger.info(f"Set {len(request.aliases)} aliases for canonical group '{canonical}'")
    return GroupAliasResponse(canonical=canonical, aliases=request.aliases)


@router.post(
    "/text_generation/aliases/{canonical}/add",
    response_model=GroupAliasResponse,
    summary="Add a single alias to a canonical group",
    tags=["text_utils"],
)
async def add_alias(
    canonical: str,
    request: AddAliasRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupAliasResponse:
    """Add one alias to a canonical group. Creates the entry if needed.

    Raises 409 if the alias is already claimed by a different canonical group.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_alias_store(manager)
    try:
        store.add_alias(canonical, request.alias)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    entry = store.get(canonical)
    assert entry is not None
    return GroupAliasResponse(canonical=canonical, aliases=entry.aliases)


@router.post(
    "/text_generation/aliases/{canonical}/remove",
    response_model=GroupAliasResponse,
    summary="Remove a single alias from a canonical group",
    tags=["text_utils"],
)
async def remove_alias(
    canonical: str,
    request: RemoveAliasRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupAliasResponse:
    """Remove one alias from a canonical group.

    Returns the updated entry. Raises 404 if the alias was not found.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_alias_store(manager)
    removed = store.remove_alias(canonical, request.alias)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alias '{request.alias}' not found under canonical group '{canonical}'",
        )

    entry = store.get(canonical)
    if entry is None:
        return GroupAliasResponse(canonical=canonical, aliases=[])
    return GroupAliasResponse(canonical=canonical, aliases=entry.aliases)


@router.delete(
    "/text_generation/aliases/{canonical}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete all aliases for a canonical group",
    tags=["text_utils"],
)
async def delete_aliases(
    canonical: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> None:
    """Remove the entire alias entry for a canonical group."""
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_alias_store(manager)
    deleted = store.delete(canonical)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No alias entry for canonical group '{canonical}'",
        )


class GroupFamilyResponse(BaseModel):
    """A single related-group family."""

    family_name: str
    members: list[str]


class GroupFamilyListResponse(BaseModel):
    """Response listing all related-group families."""

    families: list[GroupFamilyResponse]


class SetFamilyRequest(BaseModel):
    """Request body for creating or replacing a family."""

    members: list[str]


class AddFamilyMemberRequest(BaseModel):
    """Request body for adding a single group to a family."""

    group_name: str


class RemoveFamilyMemberRequest(BaseModel):
    """Request body for removing a single group from a family."""

    group_name: str


class DetectFamiliesResponse(BaseModel):
    """Response from auto-detection of family suggestions."""

    suggestions: list[GroupFamilyResponse]
    total_groups_analyzed: int
    groups_in_families: int
    standalone_groups: int


def _require_group_family_store(manager: ModelReferenceManager) -> GroupFamilyStore:
    """Return the group family store or raise 503 if unavailable."""
    store = manager.group_family_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Group family storage is not available (backend may be in REPLICA mode).",
        )
    return store


@router.get(
    "/text_generation/families",
    response_model=GroupFamilyListResponse,
    summary="List all related-group families",
    tags=["text_utils"],
)
def list_families(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupFamilyListResponse:
    """Return all configured related-group families."""
    store = _require_group_family_store(manager)
    all_families = store.list_all()
    return GroupFamilyListResponse(
        families=[
            GroupFamilyResponse(family_name=name, members=family.members)
            for name, family in sorted(all_families.items())
        ],
    )


@router.get(
    "/text_generation/families/detect",
    response_model=DetectFamiliesResponse,
    summary="Auto-detect family suggestions from current model groups",
    tags=["text_utils"],
)
def detect_family_suggestions(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    min_prefix_length: Annotated[int, Query(ge=2, le=20)] = 3,
    min_family_size: Annotated[int, Query(ge=2, le=50)] = 2,
) -> DetectFamiliesResponse:
    """Run prefix-based heuristics over current group names to suggest families.

    Results are suggestions only — they are not persisted automatically.
    """
    all_models = _get_all_text_models(manager)
    if not all_models:
        return DetectFamiliesResponse(
            suggestions=[],
            total_groups_analyzed=0,
            groups_in_families=0,
            standalone_groups=0,
        )

    alias_store = manager.group_alias_store
    group_names: set[str] = set()
    for model_name in all_models:
        base = get_base_model_name(model_name)
        if alias_store is not None:
            base = alias_store.resolve(base)
        group_names.add(base)

    sorted_names = sorted(group_names)
    suggestions = detect_families(
        sorted_names,
        min_prefix_length=min_prefix_length,
        min_family_size=min_family_size,
    )

    groups_in_families = sum(len(members) for members in suggestions.values())

    return DetectFamiliesResponse(
        suggestions=[
            GroupFamilyResponse(family_name=prefix, members=members) for prefix, members in suggestions.items()
        ],
        total_groups_analyzed=len(sorted_names),
        groups_in_families=groups_in_families,
        standalone_groups=len(sorted_names) - groups_in_families,
    )


@router.get(
    "/text_generation/families/{family_name}",
    response_model=GroupFamilyResponse,
    summary="Get a specific related-group family",
    tags=["text_utils"],
)
def get_family(
    family_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> GroupFamilyResponse:
    """Return the family with the given name."""
    store = _require_group_family_store(manager)
    family = store.get_family(family_name)
    if family is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Family '{family_name}' not found",
        )
    return GroupFamilyResponse(family_name=family.family_name, members=family.members)


@router.put(
    "/text_generation/families/{family_name}",
    response_model=GroupFamilyResponse,
    summary="Create or replace a related-group family",
    tags=["text_utils"],
)
async def set_family(
    family_name: str,
    request: SetFamilyRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupFamilyResponse:
    """Create or replace a family with the given members.

    Raises 409 if any member already belongs to a different family.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_family_store(manager)
    try:
        store.set_family(family_name, request.members)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    logger.info(f"Set family '{family_name}' with {len(request.members)} members")
    return GroupFamilyResponse(family_name=family_name, members=sorted(set(request.members)))


@router.post(
    "/text_generation/families/{family_name}/add",
    response_model=GroupFamilyResponse,
    summary="Add a group to a family",
    tags=["text_utils"],
)
async def add_family_member(
    family_name: str,
    request: AddFamilyMemberRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupFamilyResponse:
    """Add a single group to an existing family.

    Raises 404 if the family does not exist, 409 if the group belongs to another family.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_family_store(manager)
    try:
        store.add_member(family_name, request.group_name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    family = store.get_family(family_name)
    assert family is not None
    return GroupFamilyResponse(family_name=family_name, members=family.members)


@router.post(
    "/text_generation/families/{family_name}/remove",
    response_model=GroupFamilyResponse,
    summary="Remove a group from a family",
    tags=["text_utils"],
)
async def remove_family_member(
    family_name: str,
    request: RemoveFamilyMemberRequest,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> GroupFamilyResponse:
    """Remove a single group from a family.

    If the family becomes empty, it is deleted entirely.
    Raises 404 if the group was not found in the family.
    """
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_family_store(manager)
    removed = store.remove_member(family_name, request.group_name)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Group '{request.group_name}' not found in family '{family_name}'",
        )

    family = store.get_family(family_name)
    if family is None:
        return GroupFamilyResponse(family_name=family_name, members=[])
    return GroupFamilyResponse(family_name=family_name, members=family.members)


@router.delete(
    "/text_generation/families/{family_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a related-group family",
    tags=["text_utils"],
)
async def delete_family(
    family_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> None:
    """Delete an entire family, releasing all its members."""
    await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)

    store = _require_group_family_store(manager)
    deleted = store.delete(family_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Family '{family_name}' not found",
        )
