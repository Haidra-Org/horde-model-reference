import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from haidra_core.service_base import ContainsMessage
from loguru import logger
from pydantic import ValidationError

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    GenericModelRecord,
)
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v2_prefix
from horde_model_reference.service.v2.models import ErrorResponse, ModelRecordUnion, ModelRecordUnionType

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


info_route_subpath = "/info"
"""/info"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_reference_info,
    info_route_subpath,
)


@router.get(info_route_subpath)
async def get_reference_info() -> ContainsMessage:
    """Info about the legacy model reference API, as follows.

    This is the  model reference API, which uses the new format established by horde_model_reference.
    """
    info = get_reference_info.__doc__ or "No information available."
    return ContainsMessage(message=info.replace("\n\n", " ").replace("\n", " ").strip())


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()


read_reference_route_subpath = "/model_categories"
"""/model_categories"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_reference_names,
    read_reference_route_subpath,
)


@router.get(read_reference_route_subpath, response_model=list[MODEL_REFERENCE_CATEGORY | str])
async def get_reference_names(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)]
) -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Get all legacy model reference names."""
    return list(manager.get_all_model_references().keys())


get_reference_by_category_route_subpath = f"/{{{PathVariables.model_category_name}}}"
"""/{model_category_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_reference_by_category,
    get_reference_by_category_route_subpath,
)


@router.get(
    get_reference_by_category_route_subpath,
    response_model=dict[
        str,
        ModelRecordUnion,
    ],
    responses={
        404: {"description": "Model category not found"},
    },
)
async def get_reference_by_category(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Get a specific model reference by category name."""
    raw_json = manager.get_raw_model_reference_json(model_category_name)

    if raw_json is None:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found")

    return JSONResponse(content=raw_json, media_type="application/json")


single_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_single_model,
    single_model_route_subpath,
)


@router.get(
    "/{model_category_name}/{model_name}",
    response_model=dict[str, Any],
    responses={
        404: {"description": "Model category or model not found", "model": ErrorResponse},
    },
)
async def get_single_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Get a specific model by category and name.

    Args:
        model_category_name: The model reference category (e.g., image_generation).
        model_name: The name of the model within the category.
        manager: The model reference manager dependency.

    Returns:
        JSONResponse: The model record data.

    Raises:
        HTTPException: 404 if category or model not found.
    """
    raw_json = manager.get_raw_model_reference_json(model_category_name)

    if raw_json is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model category '{model_category_name}' not found",
        )

    if model_name not in raw_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'",
        )

    return JSONResponse(content=raw_json[model_name], media_type="application/json")


add_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/add"
"""/{model_category_name}/add"""
route_registry.register_route(
    v2_prefix,
    RouteNames.create_model,
    add_model_route_subpath,
)


@router.post(
    "/{model_category_name}/add",
    response_model=ModelRecordUnion,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Model created successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        409: {"description": "Model already exists", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable (REPLICA mode)", "model": ErrorResponse},
    },
)
async def create_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    request_body: ModelRecordUnion,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new model in the specified category.

    This endpoint is only available in PRIMARY mode. REPLICA instances will return 503.

    Args:
        model_category_name (str): The model reference category.
        model_name (str): The name of the model to create. Must match the 'name' field in request body.
        request_body (ModelRecordUnion): The model record data conforming to the category's schema.
        manager (ModelReferenceManager): The model reference manager dependency.

    Returns:
        JSONResponse: The created model record data.

    Raises:
        HTTPException: 400 for invalid requests, 409 if model exists, 422 for validation errors,
            503 if backend doesn't support writes (REPLICA mode).
    """
    from horde_model_reference import horde_model_reference_settings

    if not manager.backend.supports_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance is in REPLICA mode and does not support write operations. "
            "Only PRIMARY instances can create models.",
        )

    if horde_model_reference_settings.canonical_format != "v2":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance uses legacy format as canonical. "
            "Write operations are only available via v1 API when canonical_format='legacy'. "
            "To use v2 CRUD, set canonical_format='v2'.",
        )

    existing_models = manager.get_raw_model_reference_json(model_category_name)
    if existing_models is not None and request_body.name in existing_models:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{request_body.name}' already exists in category '{model_category_name}'. "
            "Use PUT to update existing models.",
        )

    record_type = MODEL_RECORD_TYPE_LOOKUP.get(model_category_name, GenericModelRecord)

    try:
        model_record = record_type.model_validate(request_body)
    except ValidationError as e:
        logger.warning(f"Validation error creating model '{request_body.name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e

    try:
        manager.update_model(model_category_name, request_body.name, model_record)
        logger.info(f"Created model '{request_body.name}' in category '{model_category_name}'")
    except Exception as e:
        logger.exception(f"Error creating model '{request_body.name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {e!s}",
        ) from e

    created_data = model_record.model_dump(exclude_unset=True)
    return JSONResponse(content=created_data, status_code=status.HTTP_201_CREATED, media_type="application/json")


update_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.update_model,
    update_model_route_subpath,
)


@router.put(
    "/{model_category_name}/{model_name}",
    response_model=dict[str, Any],
    responses={
        200: {"description": "Model updated successfully"},
        201: {"description": "Model created successfully"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable (REPLICA mode)", "model": ErrorResponse},
    },
)
async def update_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    request_body: dict[str, Any],
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Update an existing model or create if it doesn't exist (upsert).

    This endpoint is only available in PRIMARY mode. REPLICA instances will return 503.

    If the model exists:
    - Preserves original `created_at` and `created_by` metadata
    - Updates `updated_at` timestamp

    If the model doesn't exist:
    - Sets `created_at` timestamp
    - Returns 201 Created status

    Args:
        model_category_name: The model reference category.
        model_name: The name of the model to update. Must match the 'name' field in request body.
        request_body: The model record data conforming to the category's schema.
        manager: The model reference manager dependency.

    Returns:
        JSONResponse: The updated model record data.

    Raises:
        HTTPException: 400 for invalid requests, 422 for validation errors,
            503 if backend doesn't support writes (REPLICA mode).
    """
    from horde_model_reference import horde_model_reference_settings

    if not manager.backend.supports_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance is in REPLICA mode and does not support write operations. "
            "Only PRIMARY instances can update models.",
        )

    if horde_model_reference_settings.canonical_format != "v2":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance uses legacy format as canonical. "
            "Write operations are only available via v1 API when canonical_format='legacy'. "
            "To use v2 CRUD, set canonical_format='v2'.",
        )

    if "name" not in request_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must include 'name' field",
        )

    if request_body["name"] != model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model name in URL ('{model_name}') must match name in body ('{request_body['name']}')",
        )

    existing_models = manager.get_raw_model_reference_json(model_category_name)
    is_new = existing_models is None or model_name not in existing_models

    record_type = MODEL_RECORD_TYPE_LOOKUP.get(model_category_name, GenericModelRecord)

    if "metadata" not in request_body:
        request_body["metadata"] = {}

    if is_new:
        request_body["metadata"]["created_at"] = int(time.time())
    else:
        if existing_models is not None and model_name in existing_models:
            existing_metadata = existing_models[model_name].get("metadata", {})
            if "created_at" in existing_metadata:
                request_body["metadata"]["created_at"] = existing_metadata["created_at"]
            if "created_by" in existing_metadata:
                request_body["metadata"]["created_by"] = existing_metadata["created_by"]

        request_body["metadata"]["updated_at"] = int(time.time())

    request_body["metadata"]["schema_version"] = "2.0.0"

    try:
        model_record = record_type.model_validate(request_body)
    except ValidationError as e:
        logger.warning(f"Validation error updating model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e

    try:
        manager.update_model(model_category_name, model_name, model_record)
        action = "Created" if is_new else "Updated"
        logger.info(f"{action} model '{model_name}' in category '{model_category_name}'")
    except Exception as e:
        logger.exception(f"Error updating model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {e!s}",
        ) from e

    updated_data = model_record.model_dump(exclude_unset=True)
    response_status = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return JSONResponse(content=updated_data, status_code=response_status, media_type="application/json")


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.delete_model,
    delete_model_route_subpath,
)


@router.delete(
    "/{model_category_name}/{model_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Model deleted successfully"},
        404: {"description": "Model not found", "model": ErrorResponse},
        503: {"description": "Service unavailable (REPLICA mode)", "model": ErrorResponse},
    },
)
async def delete_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Delete a model from the specified category.

    This endpoint is only available in PRIMARY mode. REPLICA instances will return 503.

    Args:
        model_category_name: The model reference category.
        model_name: The name of the model to delete.
        manager: The model reference manager dependency.

    Returns:
        Response: 204 No Content on success.

    Raises:
        HTTPException: 404 if model not found, 503 if backend doesn't support writes (REPLICA mode).
    """
    from horde_model_reference import horde_model_reference_settings

    if not manager.backend.supports_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance is in REPLICA mode and does not support write operations. "
            "Only PRIMARY instances can delete models.",
        )

    if horde_model_reference_settings.canonical_format != "v2":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance uses legacy format as canonical. "
            "Write operations are only available via v1 API when canonical_format='legacy'. "
            "To use v2 CRUD, set canonical_format='v2'.",
        )

    existing_models = manager.get_raw_model_reference_json(model_category_name)
    if existing_models is None or model_name not in existing_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'",
        )

    try:
        manager.delete_model(model_category_name, model_name)
        logger.info(f"Deleted model '{model_name}' from category '{model_category_name}'")
    except KeyError as e:
        logger.warning(f"Model '{model_name}' not found during deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'",
        ) from e
    except Exception as e:
        logger.exception(f"Error deleting model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {e!s}",
        ) from e

    return Response(status_code=status.HTTP_204_NO_CONTENT)
