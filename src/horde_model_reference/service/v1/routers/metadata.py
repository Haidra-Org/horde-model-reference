"""V1 metadata API endpoints for legacy format operations."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from horde_model_reference import ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_metadata import CategoryMetadata
from horde_model_reference.service.shared import (
    RouteNames,
    route_registry,
    v1_prefix,
)
from horde_model_reference.service.v1.routers.shared import get_model_reference_manager

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


class LastUpdatedResponse(BaseModel):
    """Response for /last_updated endpoint."""

    last_updated: int | None


class CategoryLastUpdatedResponse(BaseModel):
    """Response for /{category}/last_updated endpoint."""

    category: str
    last_updated: int | None


# /last_updated endpoint
last_updated_route_subpath = "/last_updated"
route_registry.register_route(
    v1_prefix + "/metadata",
    RouteNames.get_legacy_last_updated,
    last_updated_route_subpath,
)


@router.get(
    last_updated_route_subpath,
    response_model=LastUpdatedResponse,
    summary="Get last update timestamp for canonical format (legacy)",
    operation_id="read_legacy_last_updated",
    responses={
        200: {
            "description": "Last updated timestamp",
            "links": {
                "GetAllLegacyMetadata": {
                    "operationId": "read_all_legacy_metadata",
                    "description": "Get all legacy format metadata for all categories.",
                },
            },
        },
        503: {"description": "Service unavailable (metadata not supported or non-canonical format)"},
    },
    tags=["v1", "metadata"],
)
async def read_legacy_last_updated(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LastUpdatedResponse:
    """Get the last update timestamp for the canonical format.

    This endpoint returns the maximum last_updated timestamp across all categories
    for legacy format operations. Only available when canonical_format='legacy'.

    Returns:
        LastUpdatedResponse with the maximum timestamp, or None if no metadata exists.

    Raises:
        HTTPException: 503 if metadata is not supported or canonical_format != 'legacy'.
    """
    # Check if backend supports metadata
    if not manager.backend.supports_metadata():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadata tracking is not supported in REPLICA mode",
        )

    # Check if canonical format is 'legacy'
    if horde_model_reference_settings.canonical_format != "legacy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"This endpoint is only available when canonical_format='legacy'. "
            f"Current setting: canonical_format='{horde_model_reference_settings.canonical_format}'",
        )

    # Get all legacy metadata and find max timestamp
    all_metadata = manager.backend.get_all_legacy_metadata()

    if not all_metadata:
        return LastUpdatedResponse(last_updated=None)

    max_timestamp = max(
        (metadata.last_updated for metadata in all_metadata.values()),
        default=None,
    )

    return LastUpdatedResponse(last_updated=max_timestamp)


# /{model_category_name}/last_updated endpoint
category_last_updated_route_subpath = "/{model_category_name}/last_updated"
route_registry.register_route(
    v1_prefix + "/metadata",
    RouteNames.get_legacy_category_last_updated,
    category_last_updated_route_subpath,
)


@router.get(
    category_last_updated_route_subpath,
    response_model=CategoryLastUpdatedResponse,
    summary="Get last update timestamp for a specific category (legacy)",
    operation_id="read_legacy_category_last_updated",
    responses={
        200: {
            "description": "Category last updated timestamp",
            "links": {
                "GetCategoryMetadata": {
                    "operationId": "read_legacy_category_metadata",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Get full metadata for this category.",
                },
            },
        },
        404: {"description": "Category not found or no metadata available"},
        503: {"description": "Service unavailable (metadata not supported)"},
    },
    tags=["v1", "metadata"],
)
async def read_legacy_category_last_updated(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> CategoryLastUpdatedResponse:
    """Get the last update timestamp for a specific category (legacy format).

    Args:
        model_category_name: The model reference category to get metadata for.
        manager: The model reference manager dependency.

    Returns:
        CategoryLastUpdatedResponse with the timestamp, or None if no metadata exists.

    Raises:
        HTTPException: 503 if metadata is not supported.
        HTTPException: 404 if category has no metadata.
    """
    # Check if backend supports metadata
    if not manager.backend.supports_metadata():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadata tracking is not supported in REPLICA mode",
        )

    metadata = manager.backend.get_legacy_metadata(model_category_name)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No legacy metadata found for category {model_category_name.value}",
        )

    return CategoryLastUpdatedResponse(
        category=model_category_name.value,
        last_updated=metadata.last_updated,
    )


# /metadata endpoint
metadata_route_subpath = "/metadata"
route_registry.register_route(
    v1_prefix + "/metadata",
    RouteNames.get_all_legacy_metadata,
    metadata_route_subpath,
)


@router.get(
    metadata_route_subpath,
    response_model=dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata],
    summary="Get all legacy format metadata",
    operation_id="read_all_legacy_metadata",
    responses={
        200: {
            "description": "All legacy format metadata",
            "links": {
                "GetCategoryMetadata": {
                    "operationId": "read_legacy_category_metadata",
                    "description": "Get metadata for a specific category.",
                },
                "GetLastUpdated": {
                    "operationId": "read_legacy_last_updated",
                    "description": "Get the most recent update timestamp across all categories.",
                },
            },
        },
        503: {"description": "Service unavailable (metadata not supported)"},
    },
    tags=["v1", "metadata"],
)
async def read_all_legacy_metadata(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
    """Get all legacy format metadata.

    Returns a dictionary mapping each category to its legacy format metadata.

    Args:
        manager: The model reference manager dependency.

    Returns:
        Dict of category to CategoryMetadata.

    Raises:
        HTTPException: 503 if metadata is not supported.
    """
    # Check if backend supports metadata
    if not manager.backend.supports_metadata():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadata tracking is not supported in REPLICA mode",
        )

    return manager.backend.get_all_legacy_metadata()


# /{model_category_name} endpoint
category_metadata_route_subpath = "/{model_category_name}"
route_registry.register_route(
    v1_prefix + "/metadata",
    RouteNames.get_legacy_category_metadata,
    category_metadata_route_subpath,
)


@router.get(
    category_metadata_route_subpath,
    response_model=CategoryMetadata,
    summary="Get legacy format metadata for a specific category",
    operation_id="read_legacy_category_metadata",
    responses={
        200: {
            "description": "Category metadata",
            "links": {
                "GetCategoryLastUpdated": {
                    "operationId": "read_legacy_category_last_updated",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Get just the last updated timestamp for this category.",
                },
                "GetAllMetadata": {
                    "operationId": "read_all_legacy_metadata",
                    "description": "Get metadata for all categories.",
                },
            },
        },
        404: {"description": "Category not found or no metadata available"},
        503: {"description": "Service unavailable (metadata not supported)"},
    },
    tags=["v1", "metadata"],
)
async def read_legacy_category_metadata(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> CategoryMetadata:
    """Get legacy format metadata for a specific category.

    Args:
        model_category_name: The model reference category to get metadata for.
        manager: The model reference manager dependency.

    Returns:
        CategoryMetadata for the category.

    Raises:
        HTTPException: 503 if metadata is not supported.
        HTTPException: 404 if category has no metadata.
    """
    # Check if backend supports metadata
    if not manager.backend.supports_metadata():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metadata tracking is not supported in REPLICA mode",
        )

    metadata = manager.backend.get_legacy_metadata(model_category_name)

    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No legacy metadata found for category {model_category_name.value}",
        )

    return metadata
