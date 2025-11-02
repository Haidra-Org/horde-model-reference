"""Audit endpoints for the v2 model reference API.

Provides endpoints to retrieve model audit information including deletion risk analysis.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.analytics.audit_analysis import (
    CategoryAuditResponse,
    analyze_models_for_audit,
    calculate_audit_summary,
)
from horde_model_reference.analytics.audit_cache import AuditCache
from horde_model_reference.analytics.filter_presets import AuditFilterPreset, apply_preset_filter
from horde_model_reference.analytics.text_model_grouping import apply_text_model_grouping_to_audit
from horde_model_reference.integrations.data_merger import merge_category_with_horde_data
from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v2_prefix

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()


def get_horde_api_integration() -> HordeAPIIntegration:
    """Dependency to get the HordeAPIIntegration singleton."""
    from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration

    return HordeAPIIntegration()


def get_audit_cache() -> AuditCache:
    """Dependency to get the AuditCache singleton."""
    return AuditCache()


audit_route_subpath = f"/{{{PathVariables.model_category_name}}}/audit"
"""/{model_category_name}/audit"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_category_audit,
    audit_route_subpath,
)


@router.get(
    audit_route_subpath,
    summary="Get audit analysis for a model category",
    operation_id="read_v2_category_audit",
    response_model=CategoryAuditResponse,
    responses={
        200: {
            "description": "Category audit analysis retrieved successfully",
            "model": CategoryAuditResponse,
        },
        400: {
            "description": "Invalid category or unsupported category for audit",
        },
        404: {
            "description": "Category not found",
        },
        500: {
            "description": "Internal server error fetching Horde API data or computing audit",
        },
    },
)
async def get_category_audit(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    horde_api: Annotated[HordeAPIIntegration, Depends(get_horde_api_integration)],
    audit_cache: Annotated[AuditCache, Depends(get_audit_cache)],
    group_text_models: bool = Query(default=False, description="Group text models by base name (strips quantization)"),
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
) -> CategoryAuditResponse:
    """Get comprehensive audit analysis for a model reference category.

    Analyzes all models in the category to identify deletion risks including:
    - Missing or invalid download URLs
    - Non-preferred file hosts
    - Missing required fields (description, baseline)
    - Zero active workers
    - Low or no recent usage

    Returns both per-model audit information and aggregate summary statistics.
    Audit results are cached (default 300s TTL) and automatically invalidated
    when model data changes.

    Args:
        model_category_name: The model reference category to audit.
        manager: The model reference manager (injected).
        horde_api: The Horde API integration (injected).
        audit_cache: The audit cache (injected).
        group_text_models: Group text models by base name (strips quantization info).
        preset: Optional preset filter to apply (deletion_candidates, zero_usage, etc.).

    Returns:
        CategoryAuditResponse with per-model audit info and summary.

    Raises:
        HTTPException: 400 for unsupported categories or invalid preset, 404 if not found, 500 for errors.
    """
    logger.debug(
        f"Audit request for category: {model_category_name}, "
        f"group_text_models={group_text_models}, preset={preset}, limit={limit}, offset={offset}"
    )

    # Try cache first (uses grouped parameter, but not with preset filter)
    if not preset:
        cached_audit = audit_cache.get(model_category_name, grouped=group_text_models)
        if cached_audit:
            logger.debug(f"Returning cached audit for {model_category_name} (grouped={group_text_models})")
            # Apply pagination to cached results if requested
            if limit is not None or offset > 0:
                total_models = len(cached_audit.models)
                end_index = offset + limit if limit is not None else None
                paginated_models = cached_audit.models[offset:end_index]

                # Create new response with paginated models
                cached_audit = CategoryAuditResponse(
                    category=cached_audit.category,
                    category_total_month_usage=cached_audit.category_total_month_usage,
                    total_count=total_models,
                    returned_count=len(paginated_models),
                    offset=offset,
                    limit=limit,
                    models=paginated_models,
                    summary=cached_audit.summary,  # Summary reflects all models, not just page
                )
            return cached_audit

    # Only support categories that have Horde API data
    if model_category_name not in [
        MODEL_REFERENCE_CATEGORY.image_generation,
        MODEL_REFERENCE_CATEGORY.text_generation,
    ]:
        logger.warning(f"Audit not supported for category: {model_category_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audit analysis is only supported for image_generation and text_generation categories",
        )

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

    # Determine model type for Horde API
    model_type: Literal["image", "text"] = (
        "image" if model_category_name == MODEL_REFERENCE_CATEGORY.image_generation else "text"
    )

    # Fetch Horde API data
    logger.debug(f"Fetching Horde API data for {model_type} models")
    try:
        status_data, stats_data, workers_data = await horde_api.get_combined_data(model_type)
    except Exception as e:
        logger.exception(f"Error fetching Horde API data for {model_type}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Horde API data: {e!s}",
        ) from e

    # Merge model reference data with Horde API data
    logger.debug(f"Merging model reference data with Horde API data for {len(raw_models)} models")
    try:
        enriched_models_pydantic = merge_category_with_horde_data(
            raw_models,
            status_data,
            stats_data,
            workers_data,
        )
        # Convert Pydantic models to dictionaries for audit analysis
        enriched_models = {name: model.model_dump() for name, model in enriched_models_pydantic.items()}
    except Exception as e:
        logger.exception(f"Error merging data for {model_category_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge model data: {e!s}",
        ) from e

    # Calculate total category usage for percentage calculations
    category_total_month_usage = 0
    for model_data in enriched_models.values():
        usage_stats = model_data.get("usage_stats", {})
        if isinstance(usage_stats, dict):
            usage_month = usage_stats.get("month", 0)
            if isinstance(usage_month, int):
                category_total_month_usage += usage_month

    # Analyze models for audit
    logger.debug(f"Analyzing {len(enriched_models)} models for audit")
    try:
        audit_models = analyze_models_for_audit(
            enriched_models,
            category_total_month_usage,
            model_category_name,
        )
    except Exception as e:
        logger.exception(f"Error analyzing models for audit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze models: {e!s}",
        ) from e

    # Calculate summary
    summary = calculate_audit_summary(audit_models)

    # Create base audit response
    audit_response = CategoryAuditResponse(
        category=model_category_name,
        category_total_month_usage=category_total_month_usage,
        total_count=len(audit_models),
        returned_count=len(audit_models),
        offset=0,
        limit=None,
        models=audit_models,
        summary=summary,
    )

    # Cache the base response (before preset filtering, after grouping)
    if not preset:
        audit_cache.set(model_category_name, audit_response, grouped=group_text_models)
        logger.debug(f"Cached audit results for {model_category_name} (grouped={group_text_models})")

    # Apply text model grouping if requested
    if group_text_models:
        logger.debug(f"Applying text model grouping for {model_category_name}")
        audit_response = apply_text_model_grouping_to_audit(audit_response)

    # Apply preset filter if requested
    if preset:
        try:
            logger.debug(f"Applying preset filter '{preset}' to {len(audit_response.models)} models")
            filtered_models = apply_preset_filter(audit_response.models, preset)

            audit_response = CategoryAuditResponse(
                category=audit_response.category,
                category_total_month_usage=audit_response.category_total_month_usage,
                total_count=audit_response.total_count,  # Preserve original total
                returned_count=len(filtered_models),
                offset=0,
                limit=None,
                models=filtered_models,
                summary=calculate_audit_summary(filtered_models),
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
        total_models = len(audit_response.models)
        end_index = offset + limit if limit is not None else None
        paginated_models = audit_response.models[offset:end_index]

        audit_response = CategoryAuditResponse(
            category=audit_response.category,
            category_total_month_usage=audit_response.category_total_month_usage,
            total_count=total_models,
            returned_count=len(paginated_models),
            offset=offset,
            limit=limit,
            models=paginated_models,
            summary=audit_response.summary,  # Summary reflects all models, not just page
        )

    logger.info(
        f"Audit completed for {model_category_name}: "
        f"{audit_response.returned_count} of {audit_response.total_count} models returned, "
        f"{audit_response.summary.models_at_risk} at risk, avg risk score: {audit_response.summary.average_risk_score}"
    )

    return audit_response
