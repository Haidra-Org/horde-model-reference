"""Search and popularity endpoints for the v2 API."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord
from horde_model_reference.service.shared import get_model_reference_manager

router = APIRouter()

MAX_SEARCH_LIMIT = 500
DEFAULT_SEARCH_LIMIT = 50


class SearchResponse(BaseModel):
    """Paginated search response."""

    results: list[dict[str, Any]]
    """Serialized model records matching the query."""

    total: int
    """Total number of matches before limit/offset (for pagination)."""

    offset: int
    """Offset applied."""

    limit: int
    """Limit applied."""

    has_more: bool
    """Whether more results exist beyond the current page."""


def _validate_category(category_name: str) -> MODEL_REFERENCE_CATEGORY:
    try:
        return MODEL_REFERENCE_CATEGORY(category_name)
    except ValueError:
        valid = [c.value for c in MODEL_REFERENCE_CATEGORY]
        raise HTTPException(status_code=422, detail=f"Unknown category '{category_name}'. Valid: {valid}") from None


def _serialize_record(record: GenericModelRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _apply_generic_filters(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    nsfw: bool | None,
    baseline: str | None,
    inpainting: bool | None,
    tags_any: list[str] | None,
    tags_all: list[str] | None,
    tags_none: list[str] | None,
    name_contains: str | None,
    sort_by: str | None,
    sort_desc: bool,
    limit: int,
    offset: int,
    backend: str | None,
    exclude_backend_variations: bool,
    quantized: bool | None,
) -> SearchResponse:
    """Build a query from parameters, execute, and return a SearchResponse."""
    q = manager.query(category)

    try:
        if nsfw is not None:
            q = q.where(nsfw=nsfw)

        if baseline is not None:
            q = q.where(baseline=baseline)

        if inpainting is not None:
            q = q.where(inpainting=inpainting)

        if tags_any is not None:
            q = q.tags_any(tags_any)

        if tags_all is not None:
            q = q.tags_all(tags_all)

        if tags_none is not None:
            q = q.tags_none(tags_none)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Filter not supported for this category: {exc}") from None

    if name_contains is not None:
        lower_q = name_contains.lower()
        q = q.filter(lambda r: lower_q in r.name.lower())

    # Text-generation-specific filters
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.query import TextModelQuery

        if isinstance(q, TextModelQuery):
            if backend is not None:
                q = q.for_backend(backend)  # type: ignore[assignment]
            if exclude_backend_variations:
                q = q.exclude_backend_variations()  # type: ignore[assignment]
            if quantized is True:
                q = q.only_quantized()  # type: ignore[assignment]
            elif quantized is False:
                q = q.exclude_quantized()  # type: ignore[assignment]

    if sort_by is not None:
        try:
            q = q.order_by(sort_by, descending=sort_desc)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid sort_by field: {exc}") from None

    total = q.count()

    q = q.offset(offset).limit(limit)
    matched = q.to_list()

    return SearchResponse(
        results=[_serialize_record(r) for r in matched],
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
    )


@router.get(
    "/{model_category_name}/search",
    response_model=SearchResponse,
    summary="Search models in a category",
)
def search_category(
    model_category_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    nsfw: Annotated[bool | None, Query(description="Filter by NSFW status")] = None,
    baseline: Annotated[str | None, Query(description="Filter by baseline")] = None,
    inpainting: Annotated[bool | None, Query(description="Filter by inpainting (image only)")] = None,
    tags_any: Annotated[list[str] | None, Query(description="Models with any of these tags")] = None,
    tags_all: Annotated[list[str] | None, Query(description="Models with all of these tags")] = None,
    tags_none: Annotated[list[str] | None, Query(description="Models with none of these tags")] = None,
    name_contains: Annotated[str | None, Query(description="Case-insensitive name substring match")] = None,
    sort_by: Annotated[str | None, Query(description="Field name to sort by")] = None,
    sort_desc: Annotated[bool, Query(description="Sort descending")] = False,
    limit: Annotated[
        int, Query(ge=1, le=MAX_SEARCH_LIMIT, description="Max results to return")
    ] = DEFAULT_SEARCH_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Number of results to skip")] = 0,
    backend: Annotated[str | None, Query(description="Text model backend filter")] = None,
    exclude_backend_variations: Annotated[bool, Query(description="Exclude text model backend variations")] = False,
    quantized: Annotated[bool | None, Query(description="Filter by quantization (text only)")] = None,
) -> SearchResponse:
    """Search models within a specific category with filtering, sorting, and pagination."""
    category = _validate_category(model_category_name)
    return _apply_generic_filters(
        manager,
        category,
        nsfw=nsfw,
        baseline=baseline,
        inpainting=inpainting,
        tags_any=tags_any,
        tags_all=tags_all,
        tags_none=tags_none,
        name_contains=name_contains,
        sort_by=sort_by,
        sort_desc=sort_desc,
        limit=limit,
        offset=offset,
        backend=backend,
        exclude_backend_variations=exclude_backend_variations,
        quantized=quantized,
    )


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search models across all categories",
)
def search_all(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    nsfw: Annotated[bool | None, Query(description="Filter by NSFW status")] = None,
    name_contains: Annotated[str | None, Query(description="Case-insensitive name substring match")] = None,
    tags_any: Annotated[list[str] | None, Query(description="Models with any of these tags")] = None,
    tags_all: Annotated[list[str] | None, Query(description="Models with all of these tags")] = None,
    tags_none: Annotated[list[str] | None, Query(description="Models with none of these tags")] = None,
    sort_by: Annotated[str | None, Query(description="Field name to sort by")] = None,
    sort_desc: Annotated[bool, Query(description="Sort descending")] = False,
    limit: Annotated[
        int, Query(ge=1, le=MAX_SEARCH_LIMIT, description="Max results to return")
    ] = DEFAULT_SEARCH_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Number of results to skip")] = 0,
) -> SearchResponse:
    """Search models across all categories with generic filters only."""
    q = manager.query_all()

    if nsfw is not None:
        nsfw_val = nsfw
        q = q.filter(lambda r: getattr(r, "nsfw", None) == nsfw_val)

    if name_contains is not None:
        lower_q = name_contains.lower()
        q = q.filter(lambda r: lower_q in r.name.lower())

    if tags_any is not None:
        tag_set_any = set(tags_any)
        q = q.filter(lambda r: bool(tag_set_any & set(getattr(r, "tags", None) or [])))

    if tags_all is not None:
        tag_set_all = set(tags_all)
        q = q.filter(lambda r: tag_set_all <= set(getattr(r, "tags", None) or []))

    if tags_none is not None:
        tag_set_none = set(tags_none)
        q = q.filter(
            lambda r: not bool(tag_set_none & set(getattr(r, "tags", None) or [])),
        )

    if sort_by is not None:
        try:
            q = q.order_by(sort_by, descending=sort_desc)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid sort_by field: {exc}") from None

    total = q.count()

    q = q.offset(offset).limit(limit)
    all_results = q.to_list()

    return SearchResponse(
        results=[_serialize_record(r) for r in all_results],
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + limit < total,
    )


@router.get(
    "/{model_category_name}/popular",
    summary="Get popular models ranked by live Horde usage",
)
async def popular_models(
    model_category_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    limit: Annotated[int, Query(ge=1, le=100, description="Max results")] = 10,
    sort_by: Annotated[
        Literal["worker_count", "usage_day", "usage_month", "usage_total"],
        Query(description="Metric to rank by"),
    ] = "worker_count",
    include_workers: Annotated[bool, Query(description="Include per-worker details")] = False,
) -> list[dict[str, Any]]:
    """Return models ranked by live Horde popularity metrics.

    Only ``image_generation`` and ``text_generation`` have Horde API data.
    Other categories return an empty list.
    """
    category = _validate_category(model_category_name)
    results = await manager.get_popular_models(
        category,
        limit=limit,
        sort_by=sort_by,
        include_workers=include_workers,
    )
    return [r.model_dump(mode="json", exclude_none=True) for r in results]
