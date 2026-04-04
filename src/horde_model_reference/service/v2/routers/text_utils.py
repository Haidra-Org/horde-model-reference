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
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
)
from horde_model_reference.service.v2.routers.write_validations import assert_v2_write_enabled
from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

router = APIRouter()

_CATEGORY = MODEL_REFERENCE_CATEGORY.text_generation

# Fields that can be shared/edited at group level
_COMMON_FIELD_KEYS = ("baseline", "description", "url", "nsfw", "tags", "style", "instruct_format")


class ParsedNameResponse(BaseModel):
    """Structured result of parsing a text model name."""

    original_name: str
    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    version: str | None = None
    suggested_group: str


class ParsedNameInfo(BaseModel):
    """Parsed name components for a single group member."""

    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    version: str | None = None


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


class ComposeNameResponse(BaseModel):
    """Response from the name composition endpoint."""

    composed_name: str
    already_exists: bool
    suggested_group: str


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
    pending_change_ids: list[str]


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
) -> str:
    """Compose a model name from structured parts.

    When part_order is provided, parts are arranged in that order, using separator.
    Otherwise uses default: [author/]base{sep}size[{sep}variant][{sep}version][{sep}quant]
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

    if part_order:
        ordered = [available_parts[p] for p in part_order if p in available_parts]
    else:
        ordered = [base_name, size]
        if variant:
            ordered.append(variant)
        if version:
            ordered.append(version)
        if quant:
            ordered.append(quant)

    model_name = separator.join(ordered)

    if author:
        return f"{author}/{model_name}"
    return model_name


@router.get(
    "/text_generation/parse_name",
    response_model=ParsedNameResponse,
    summary="Parse a text model name into structured components",
    tags=["text_utils"],
)
def parse_name(
    name: Annotated[str, Query(description="The model name to parse")],
) -> ParsedNameResponse:
    """Parse a text model name into base name, size, variant, and quantization components."""
    parsed = parse_text_model_name(name)
    suggested_group = get_base_model_name(name)

    return ParsedNameResponse(
        original_name=name,
        base_name=parsed.base_name,
        size=parsed.size,
        variant=parsed.variant,
        quant=parsed.quant,
        version=parsed.version,
        suggested_group=suggested_group,
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

    # Infer naming convention from canonical member names
    canonical_names = [k for k, _ in canonical_members]
    schema = infer_name_format(canonical_names)
    name_format = NameFormatInfo(
        separator=schema.separator,
        part_order=schema.part_order,
        author_included=schema.author_included,
        common_author=schema.common_author,
        template=schema.template,
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
    composed = _compose_name_from_parts(
        base_name=request.base_name,
        size=request.size,
        variant=request.variant,
        version=request.version,
        quant=request.quant,
        author=request.author,
        separator=request.separator or "-",
        part_order=request.part_order,
    )

    all_models = _get_all_text_models(manager)
    already_exists = composed in all_models

    suggested_group = get_base_model_name(composed)

    return ComposeNameResponse(
        composed_name=composed,
        already_exists=already_exists,
        suggested_group=suggested_group,
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
    assert_v2_write_enabled(manager)

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
    pending_ids: list[str] = []

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
        pending_ids.append(change.id)

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
