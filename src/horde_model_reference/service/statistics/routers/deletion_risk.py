"""Deletion risk analysis endpoints for the v2 model reference API.

Provides endpoints to retrieve model deletion risk information.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.analytics.deletion_risk_analysis import (
    CategoryDeletionRiskResponse,
    CategoryDeletionRiskSummary,
    ModelDeletionRiskInfoFactory,
)
from horde_model_reference.analytics.deletion_risk_cache import DeletionRiskCache
from horde_model_reference.analytics.filter_presets import apply_preset_filter
from horde_model_reference.analytics.text_model_grouping import apply_text_model_grouping_to_risk_response
from horde_model_reference.integrations.data_merger import merge_category_with_horde_data
from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import (
    PathVariables,
    RouteNames,
    get_model_reference_manager,
    route_registry,
    v2_prefix,
)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


def get_horde_api_integration() -> HordeAPIIntegration:
    """Dependency to get the HordeAPIIntegration singleton."""
    from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration

    return HordeAPIIntegration()


def get_deletion_risk_cache() -> DeletionRiskCache:
    """Dependency to get the DeletionRiskCache singleton."""
    return DeletionRiskCache()


deletion_risk_route_subpath = f"/{{{PathVariables.model_category_name}}}/deletion-risk"
"""/{model_category_name}/deletion-risk"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_category_deletion_risk,
    deletion_risk_route_subpath,
)


@router.get(
    deletion_risk_route_subpath,
    summary="Get deletion risk analysis for a model category",
    operation_id="read_v2_category_deletion_risk",
    response_model=CategoryDeletionRiskResponse,
    responses={
        200: {
            "description": "Category deletion risk analysis retrieved successfully",
            "model": CategoryDeletionRiskResponse,
        },
        400: {
            "description": "Invalid category or unsupported for deletion risk analysis",
        },
        404: {
            "description": "Category not found",
        },
        500: {
            "description": "Internal server error fetching Horde API data or computing deletion risk",
        },
    },
)
async def get_category_deletion_risk(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    horde_api: Annotated[HordeAPIIntegration, Depends(get_horde_api_integration)],
    risk_cache: Annotated[DeletionRiskCache, Depends(get_deletion_risk_cache)],
    group_text_models: bool = Query(default=False, description="Group text models by base name (strips quantization)"),
    include_backend_variations: bool = Query(
        default=False,
        description=(
            "Include per-backend breakdown (aphrodite, koboldcpp) for text models. "
            "Only applies when group_text_models=False."
        ),
    ),
    preset: str | None = Query(
        default=None,
        description=(
            "Apply preset filter to results. "
            "Valid presets: deletion_candidates, zero_usage, no_workers, missing_data, "
            "host_issues, critical, low_usage"
        ),
    ),
    limit: int | None = Query(default=None, ge=1, description="Maximum number of models to return (None = all)"),
    offset: int = Query(default=0, ge=0, description="Number of models to skip (for pagination)"),
) -> CategoryDeletionRiskResponse:
    """Get comprehensive deletion risk analysis for a model reference category.

    Analyzes all models in the category to identify deletion risks including:
    - Missing or invalid download URLs
    - Non-preferred file hosts
    - Missing required fields (description, baseline)
    - Zero active workers
    - Low or no recent usage

    Returns both per-model risk information and aggregate summary statistics.
    Results are cached (default 300s TTL) and automatically invalidated
    when model data changes.

    Args:
        model_category_name: The model reference category to analyze.
        manager: The model reference manager (injected).
        horde_api: The Horde API integration (injected).
        risk_cache: The deletion risk cache (injected).
        group_text_models: Group text models by base name (strips quantization info).
        include_backend_variations: Include per-backend breakdown for text models (ungrouped view).
        preset: Optional preset filter to apply (deletion_candidates, zero_usage, etc.).
        limit: Maximum number of models to return (None = all).
        offset: Number of models to skip (for pagination).

    Returns:
        CategoryDeletionRiskResponse with per-model risk info and summary.

    Raises:
        HTTPException: 400 for unsupported categories or invalid preset, 404 if not found, 500 for errors.

    """
    # Determine effective backend variations flag
    # Only include backend variations for text models in ungrouped mode
    is_text_category = model_category_name == MODEL_REFERENCE_CATEGORY.text_generation
    effective_include_backend_variations = include_backend_variations and is_text_category and not group_text_models

    logger.debug(
        f"Deletion risk request for category: {model_category_name}, "
        f"group_text_models={group_text_models}, include_backend_variations={effective_include_backend_variations}, "
        f"preset={preset}, limit={limit}, offset={offset}"
    )

    # Try cache first (uses grouped parameter and backend_variations, but not with preset filter)
    if not preset:
        cached_response = risk_cache.get(
            model_category_name,
            grouped=group_text_models,
            include_backend_variations=effective_include_backend_variations,
        )
        if cached_response:
            logger.debug(
                f"Returning cached deletion risk for {model_category_name} "
                f"(grouped={group_text_models}, backend_variations={effective_include_backend_variations})"
            )
            # Apply pagination to cached results if requested
            if limit is not None or offset > 0:
                total_models = len(cached_response.models)
                end_index = offset + limit if limit is not None else None
                paginated_models = cached_response.models[offset:end_index]

                # Create new response with paginated models
                cached_response = CategoryDeletionRiskResponse(
                    category=cached_response.category,
                    category_total_month_usage=cached_response.category_total_month_usage,
                    total_count=total_models,
                    returned_count=len(paginated_models),
                    offset=offset,
                    limit=limit,
                    models=paginated_models,
                    summary=cached_response.summary,  # Summary reflects all models, not just page
                )
            return cached_response

    # Only support categories that have Horde API data
    if model_category_name not in [
        MODEL_REFERENCE_CATEGORY.image_generation,
        MODEL_REFERENCE_CATEGORY.text_generation,
    ]:
        logger.warning(f"Deletion risk analysis not supported for category: {model_category_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion risk analysis is only supported for image_generation and text_generation categories",
        )

    # Get model names from reference
    try:
        model_names = manager.get_model_names(model_category_name)
    except Exception as e:
        logger.exception(f"Error fetching models for category {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model data: {e!s}",
        ) from e

    if not model_names:
        logger.warning(f"Category not found: {model_category_name}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{model_category_name}' not found or has no models",
        )

    # Get typed model reference records
    try:
        model_records = manager.get_model_reference(model_category_name)
    except Exception as e:
        logger.exception(f"Error fetching model records for category {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model records: {e!s}",
        ) from e

    # Determine model type for Horde API
    model_type: Literal["image", "text"] = (
        "image" if model_category_name == MODEL_REFERENCE_CATEGORY.image_generation else "text"
    )

    # Fetch Horde API data
    logger.debug(f"Fetching Horde API data for {model_type} models")
    try:
        status_data = await horde_api.get_model_status_indexed(model_type)
        stats_data = await horde_api.get_model_stats_indexed(model_type)
        # Don't fetch workers for deletion risk analysis (not needed)
    except Exception as e:
        logger.exception(f"Error fetching Horde API data for {model_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Horde API data: {e!s}",
        ) from e

    # Get statistics for models (separate from model reference data)
    logger.debug(f"Merging {len(model_names)} models with Horde API data")
    try:
        model_statistics = merge_category_with_horde_data(
            model_names=model_names,
            horde_status=status_data,
            horde_stats=stats_data,
            workers=None,  # Not needed for deletion risk analysis
            include_backend_variations=effective_include_backend_variations,
        )
    except Exception as e:
        logger.exception(f"Error merging Horde API data for {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge Horde API data: {e!s}",
        ) from e

    # Calculate total category usage for percentage calculations
    category_total_month_usage = sum(
        stats.usage_stats.month for stats in model_statistics.values() if stats.usage_stats
    )

    # Analyze models and create response using factory method
    logger.debug(f"Analyzing {len(model_records)} models for deletion risk")
    try:
        factory = ModelDeletionRiskInfoFactory.create_default()
        risk_response = factory.create_deletion_risk_response(
            model_records,
            model_statistics,
            category_total_month_usage,
            model_category_name,
            include_backend_variations=effective_include_backend_variations,
        )
    except Exception as e:
        logger.exception(f"Error analyzing models for deletion risk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze models: {e!s}",
        ) from e

    # Cache the base response (before preset filtering, before grouping)
    if not preset:
        risk_cache.set(
            model_category_name,
            risk_response,
            grouped=group_text_models,
            include_backend_variations=effective_include_backend_variations,
        )
        logger.debug(
            f"Cached deletion risk results for {model_category_name} "
            f"(grouped={group_text_models}, backend_variations={effective_include_backend_variations})"
        )

    # Apply text model grouping if requested
    if group_text_models:
        logger.debug(f"Applying text model grouping for {model_category_name}")
        risk_response = apply_text_model_grouping_to_risk_response(risk_response)

    # Apply preset filter if requested
    if preset:
        try:
            logger.debug(f"Applying preset filter '{preset}' to {len(risk_response.models)} models")
            filtered_models = apply_preset_filter(risk_response.models, preset)

            risk_response = CategoryDeletionRiskResponse(
                category=risk_response.category,
                category_total_month_usage=risk_response.category_total_month_usage,
                total_count=risk_response.total_count,  # Preserve original total
                returned_count=len(filtered_models),
                offset=0,
                limit=None,
                models=filtered_models,
                summary=CategoryDeletionRiskSummary.from_risk_models(filtered_models),
            )
            logger.debug(f"Preset filter reduced to {len(filtered_models)} models")
        except ValueError as e:
            logger.warning(f"Invalid preset '{preset}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid preset: {e!s}",
            ) from e

    # Apply pagination if requested
    if limit is not None or offset > 0:
        total_models = len(risk_response.models)
        end_index = offset + limit if limit is not None else None
        paginated_models = risk_response.models[offset:end_index]

        risk_response = CategoryDeletionRiskResponse(
            category=risk_response.category,
            category_total_month_usage=risk_response.category_total_month_usage,
            total_count=total_models,
            returned_count=len(paginated_models),
            offset=offset,
            limit=limit,
            models=paginated_models,
            summary=risk_response.summary,  # Summary reflects all models, not just page
        )

    logger.info(
        f"Deletion risk analysis completed for {model_category_name}: "
        f"{risk_response.returned_count} of {risk_response.total_count} models returned, "
        f"{risk_response.summary.models_at_risk} at risk, avg risk score: {risk_response.summary.average_risk_score}"
    )

    return risk_response
