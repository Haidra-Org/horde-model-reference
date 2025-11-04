"""Statistics endpoints for the v2 model reference API.

Provides endpoints to retrieve category-level statistics with caching support.
"""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.analytics.statistics import CategoryStatistics, calculate_category_statistics
from horde_model_reference.analytics.statistics_cache import StatisticsCache
from horde_model_reference.analytics.text_model_parser import get_base_model_name
from horde_model_reference.integrations import HordeAPIIntegration
from horde_model_reference.integrations.data_merger import (
    CombinedModelStatistics,
    merge_category_with_horde_data,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import (
    ErrorResponse,
    PathVariables,
    RouteNames,
    route_registry,
    statistics_prefix,
)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()


def get_statistics_cache() -> StatisticsCache:
    """Dependency to get the StatisticsCache singleton."""
    return StatisticsCache()


statistics_route_subpath = f"/{{{PathVariables.model_category_name}}}"
"""/{model_category_name}/statistics"""
route_registry.register_route(
    statistics_prefix,
    RouteNames.get_category_statistics,
    statistics_route_subpath,
)


@router.get(
    statistics_route_subpath,
    summary="Get statistics for a model category",
    operation_id="read_v2_category_statistics",
    response_model=CategoryStatistics,
    responses={
        200: {
            "description": "Category statistics retrieved successfully",
            "model": CategoryStatistics,
        },
        404: {
            "description": "Category not found",
        },
        500: {
            "description": "Internal server error computing statistics",
        },
    },
)
async def get_category_statistics(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    stats_cache: Annotated[StatisticsCache, Depends(get_statistics_cache)],
    group_text_models: bool = Query(default=False, description="Group text models by base name (strips quantization)"),
    limit: int | None = Query(default=None, ge=1, description="Maximum number of models to return (None = all)"),
    offset: int = Query(default=0, ge=0, description="Number of models to skip (for pagination)"),
) -> CategoryStatistics:
    """Get comprehensive statistics for a model reference category.

    Returns aggregate metrics including:
    - Total model counts (overall, NSFW, SFW)
    - Baseline distribution
    - Download statistics
    - Tag and style distributions
    - Category-specific metrics (trigger words, inpainting, etc.)

    Statistics are cached with TTL (default 300s) and automatically
    invalidated when model data changes. Caching is skipped when
    grouping is enabled.

    Args:
        model_category_name: The model reference category to get statistics for.
        manager: The model reference manager (injected).
        stats_cache: The statistics cache (injected).
        group_text_models: Group text models by base name (strips quantization info).
        limit: Maximum number of models to return (for pagination).
        offset: Number of models to skip (for pagination).

    Returns:
        CategoryStatistics containing all computed metrics.

    Raises:
        HTTPException: 404 if category not found, 500 if computation fails.
    """
    logger.debug(
        f"Statistics request for category: {model_category_name}, "
        f"group_text_models={group_text_models}, limit={limit}, offset={offset}"
    )

    # Try cache first (uses grouped parameter for cache key)
    cached_stats = stats_cache.get(model_category_name, grouped=group_text_models)
    if cached_stats:
        logger.debug(f"Returning cached statistics for {model_category_name} (grouped={group_text_models})")
        # Apply pagination to cached results if requested
        if limit is not None or offset > 0:
            # Note: Statistics doesn't contain individual models, so pagination doesn't apply
            # Just update the pagination fields to reflect the request
            cached_stats.offset = offset
            cached_stats.limit = limit
        return cached_stats

    # Get model reference data
    try:
        raw_models = manager.get_raw_model_reference_json(model_category_name)
    except Exception as e:
        logger.exception(f"Error fetching models for category {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model data: {e!s}",
        ) from e

    if not raw_models:
        logger.warning(f"Category not found: {model_category_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{model_category_name}' not found or has no models",
        )

    # Apply text model grouping if requested
    if group_text_models and model_category_name == MODEL_REFERENCE_CATEGORY.text_generation:
        logger.debug(f"Grouping {len(raw_models)} text models by base name")
        grouped_models: dict[str, dict[str, Any]] = {}
        for model_name, model_data in raw_models.items():
            base_name = get_base_model_name(model_name)
            if base_name not in grouped_models:
                grouped_models[base_name] = model_data.copy()
                grouped_models[base_name]["name"] = base_name
            else:
                # Merge download entries, sum sizes, etc.
                existing = grouped_models[base_name]
                if "config" in model_data and "download" in model_data["config"]:
                    if "config" not in existing:
                        existing["config"] = {}
                    if "download" not in existing["config"]:
                        existing["config"]["download"] = []
                    existing["config"]["download"].extend(model_data["config"].get("download", []))

        raw_models = grouped_models
        logger.debug(f"Grouped into {len(raw_models)} base models")

    # Compute statistics
    logger.debug(f"Computing statistics for {model_category_name} ({len(raw_models)} models)")
    try:
        stats = calculate_category_statistics(raw_models, model_category_name)
    except Exception as e:
        logger.exception(f"Error computing statistics for {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute statistics: {e!s}",
        ) from e

    # Store in cache with grouped parameter
    stats_cache.set(model_category_name, stats, grouped=group_text_models)

    logger.info(
        f"Computed statistics for {model_category_name}: "
        f"{stats.total_models} models, {len(stats.baseline_distribution)} baselines"
    )

    return stats


def get_horde_api_integration() -> HordeAPIIntegration:
    """Dependency to get the HordeAPIIntegration singleton."""
    return HordeAPIIntegration()


with_stats_legacy_route_subpath = f"/{{{PathVariables.model_category_name}}}/with-stats"
"""/{model_category_name}/with-stats"""
route_registry.register_route(
    statistics_prefix,
    RouteNames.get_models_with_stats,
    with_stats_legacy_route_subpath,
)


@router.get(
    with_stats_legacy_route_subpath,
    response_model=dict[str, Any],
    responses={
        200: {
            "description": "Models with runtime statistics from AI Horde",
            "links": {
                "GetModelCategory": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Get raw model reference data without statistics.",
                },
            },
        },
        404: {"description": "Model category not found", "model": ErrorResponse},
        500: {"description": "Failed to fetch Horde API data", "model": ErrorResponse},
    },
    summary="Get models merged with AI Horde runtime statistics",
    operation_id="read_models_with_stats",
)
async def read_models_with_stats(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    horde_api: Annotated[HordeAPIIntegration, Depends(get_horde_api_integration)],
    include_workers: bool = False,
    min_worker_count: int | None = None,
    sort_by: str | None = None,
    sort_desc: bool = True,
) -> dict[str, CombinedModelStatistics]:
    """Get AI Horde statistics data for models in a given category.

    Combines live runtime statistics from the AI Horde API:
    - Worker count, queued jobs, performance metrics, ETA
    - Usage statistics (day, month, total)
    - Optional worker details

    **Caching:**
    - Model reference data: cached by ModelReferenceManager (60s TTL)
    - Horde API data: cached by HordeAPIIntegration (60s TTL, Redis if available)
    - Merged results: computed on-demand (no caching)

    Args:
        model_category_name: The model category (image_generation or text_generation).
        manager: Model reference manager dependency.
        horde_api: Horde API integration dependency.
        include_workers: Include detailed worker information for each model.
        min_worker_count: Filter to models with at least this many workers.
        sort_by: Sort by field (worker_count, usage_total, usage_month, name).
        sort_desc: Sort in descending order (default: True).

    Returns:
        JSONResponse: Dict of model_name -> enriched_model_data.

    Raises:
        HTTPException: 404 if category not found, 500 if Horde API fails.
    """
    # 1. Get reference data (from ModelReferenceManager cache)
    model_names = manager.get_model_names(model_category_name)
    if not model_names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{model_category_name}' not found",
        )

    # 2. Get Horde data (from HordeAPIIntegration cache)
    try:
        model_type: Literal["image", "text"]

        if model_category_name == MODEL_REFERENCE_CATEGORY.image_generation:
            model_type = "image"
        elif model_category_name == MODEL_REFERENCE_CATEGORY.text_generation:
            model_type = "text"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Category '{model_category_name}' does not support Horde statistics. "
                "Only image_generation and text_generation are supported.",
            )

        status_data = await horde_api.get_model_status_indexed(model_type)
        stats_data = await horde_api.get_model_stats_indexed(model_type)
        workers_data = await horde_api.get_workers_indexed(model_type) if include_workers else None
    except Exception as e:
        logger.exception(f"Failed to fetch Horde API data for {model_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Horde API data: {e!s}",
        ) from e

    # 3. Merge data (pure computation, no caching)

    models_statistics = merge_category_with_horde_data(
        model_names=model_names,
        horde_status=status_data,
        horde_stats=stats_data,
        workers=workers_data,
    )

    # 4. Apply server-side filtering and sorting
    # Filter by min_worker_count
    if min_worker_count is not None:
        models_statistics = {
            name: data
            for name, data in models_statistics.items()
            if data.worker_summaries and len(data.worker_summaries) >= min_worker_count
        }

    # Sort by requested field
    if sort_by is not None:
        reverse = sort_desc

        def sort_key(item: tuple[str, CombinedModelStatistics]) -> float | int | str:
            name, data = item
            if sort_by == "worker_count":
                return len(data.worker_summaries) if data.worker_summaries else 0
            if sort_by == "usage_total":
                return data.usage_stats.total if data.usage_stats else 0
            if sort_by == "usage_month":
                return data.usage_stats.month if data.usage_stats else 0
            if sort_by == "name":
                return name.lower()
            return 0  # Unknown sort field

        models_statistics = dict(sorted(models_statistics.items(), key=sort_key, reverse=reverse))

    return models_statistics
