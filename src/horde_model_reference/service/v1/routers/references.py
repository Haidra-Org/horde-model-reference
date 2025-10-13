from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, Response
from haidra_core.service_base import ContainsMessage
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.legacy.classes.legacy_models import get_legacy_model_type
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v1_prefix

router = APIRouter(
    # prefix="/references",
    responses={404: {"description": "Not found"}},
)


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()


info_route_subpath = "/info"
"""/info"""
route_registry.register_route(
    v1_prefix,
    RouteNames.get_reference_info,
    info_route_subpath,
)


@router.get(info_route_subpath)
async def read_legacy_reference_info() -> ContainsMessage:
    """Info about the legacy model reference API, as follows.

    This is the legacy model reference API, which uses the format originally found at the
    github repositories, https://github.com/Haidra-Org/AI-Horde-image-model-reference and
    https://github.com/Haidra-Org/AI-Horde-text-model-reference.
    """
    info = read_legacy_reference_info.__doc__ or "No information available."
    return ContainsMessage(message=info.replace("\n\n", " ").replace("\n", " ").strip())


read_reference_route_subpath = "/model_categories"
"""/model_categories"""
route_registry.register_route(
    v1_prefix,
    RouteNames.get_reference_names,
    read_reference_route_subpath,
)


@router.get(read_reference_route_subpath, response_model=list[MODEL_REFERENCE_CATEGORY | str])
async def read_legacy_reference_names(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Get all legacy model reference names."""
    return list(manager.backend.get_all_category_file_paths().keys())


get_reference_by_category_route_subpath = "/{model_category_name}"
"""/{model_category_name}"""
route_registry.register_route(
    v1_prefix,
    RouteNames.get_reference_by_category,
    get_reference_by_category_route_subpath,
)


@router.get(
    get_reference_by_category_route_subpath,
    responses={
        404: {"description": "Model category not found or empty"},
        422: {"description": "Invalid model category"},
    },
)
async def read_legacy_reference(
    model_category_name: MODEL_REFERENCE_CATEGORY | Literal["stable_diffusion"] | str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Get a specific legacy model reference by category name."""
    if model_category_name == "db.json":
        model_category_name = MODEL_REFERENCE_CATEGORY.text_generation

    if isinstance(model_category_name, str) and model_category_name.endswith(".json"):
        model_category_name = model_category_name[:-5]

    if model_category_name == "stable_diffusion":
        model_category_name = MODEL_REFERENCE_CATEGORY.image_generation

    try:
        model_reference_category = MODEL_REFERENCE_CATEGORY(model_category_name)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model category: '{model_category_name}'. {e!s}",
        ) from e

    raw_json_string = manager.backend.get_legacy_json_string(model_reference_category)

    if not raw_json_string or raw_json_string.strip() in ("", "{}", "null"):
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found or is empty")

    return Response(content=raw_json_string, media_type="application/json")


create_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_model,
    create_model_route_subpath,
)


@router.post(
    "/{model_category_name}/{model_name}",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Model created successfully"},
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
)
async def create_legacy_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    request_body: dict[str, Any],
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new model in legacy format.

    This endpoint is only available when canonical_format='legacy' in PRIMARY mode.

    Args:
        model_category_name: The model reference category.
        model_name: The name of the model to create. Must match the 'name' field in request body.
        request_body: The model record data in legacy format.
        manager: The model reference manager dependency.

    Returns:
        JSONResponse: The created model record data.

    Raises:
        HTTPException: 400 for invalid requests, 409 if model exists, 503 if not in legacy mode.
    """
    if not manager.backend.supports_legacy_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance does not support legacy format write operations. "
            "Legacy CRUD is only available when canonical_format='legacy' in PRIMARY mode. "
            "Use v2 API for write operations when canonical_format='v2'.",
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

    existing_models = manager.backend.get_legacy_json(model_category_name)
    if existing_models is not None and model_name in existing_models:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{model_category_name}'. "
            "Use PUT to update existing models.",
        )

    model_type = get_legacy_model_type(model_category_name)
    try:
        validated_model = model_type.model_validate(request_body)
        validated_data = validated_model.model_dump(exclude_none=True, mode="json")
    except Exception as e:
        logger.warning(f"Validation error for model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request body validation failed: {e!s}",
        ) from e

    try:
        manager.update_model_legacy(model_category_name, model_name, validated_data)
        logger.info(f"Created legacy model '{model_name}' in category '{model_category_name}'")
    except Exception as e:
        logger.exception(f"Error creating legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {e!s}",
        ) from e

    return JSONResponse(content=validated_data, status_code=status.HTTP_201_CREATED, media_type="application/json")


update_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v1_prefix,
    RouteNames.update_model,
    update_model_route_subpath,
)


@router.put(
    "/{model_category_name}/{model_name}",
    responses={
        200: {"description": "Model updated successfully"},
        201: {"description": "Model created successfully"},
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
)
async def update_legacy_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    request_body: dict[str, Any],
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Update an existing model or create if it doesn't exist (upsert) in legacy format.

    This endpoint is only available when canonical_format='legacy' in PRIMARY mode.

    Args:
        model_category_name: The model reference category.
        model_name: The name of the model to update. Must match the 'name' field in request body.
        request_body: The model record data in legacy format.
        manager: The model reference manager dependency.

    Returns:
        JSONResponse: The updated model record data.

    Raises:
        HTTPException: 400 for invalid requests, 503 if not in legacy mode.
    """
    if not manager.backend.supports_legacy_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance does not support legacy format write operations. "
            "Legacy CRUD is only available when canonical_format='legacy' in PRIMARY mode. "
            "Use v2 API for write operations when canonical_format='v2'.",
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

    existing_models = manager.backend.get_legacy_json(model_category_name)
    is_new = existing_models is None or model_name not in existing_models

    model_type = get_legacy_model_type(model_category_name)
    try:
        validated_model = model_type.model_validate(request_body)
        validated_data = validated_model.model_dump(exclude_none=True, mode="json")
    except Exception as e:
        logger.warning(f"Validation error for model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request body validation failed: {e!s}",
        ) from e

    try:
        manager.update_model_legacy(model_category_name, model_name, validated_data)
        action = "Created" if is_new else "Updated"
        logger.info(f"{action} legacy model '{model_name}' in category '{model_category_name}'")
    except Exception as e:
        logger.exception(f"Error updating legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {e!s}",
        ) from e

    response_status = status.HTTP_201_CREATED if is_new else status.HTTP_200_OK
    return JSONResponse(content=validated_data, status_code=response_status, media_type="application/json")


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v1_prefix,
    RouteNames.delete_model,
    delete_model_route_subpath,
)


@router.delete(
    "/{model_category_name}/{model_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Model deleted successfully"},
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
)
async def delete_legacy_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Delete a model from legacy format files.

    This endpoint is only available when canonical_format='legacy' in PRIMARY mode.

    Args:
        model_category_name: The model reference category.
        model_name: The name of the model to delete.
        manager: The model reference manager dependency.

    Returns:
        Response: 204 No Content on success.

    Raises:
        HTTPException: 404 if model not found, 503 if not in legacy mode.
    """
    if not manager.backend.supports_legacy_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance does not support legacy format write operations. "
            "Legacy CRUD is only available when canonical_format='legacy' in PRIMARY mode. "
            "Use v2 API for write operations when canonical_format='v2'.",
        )

    existing_models = manager.backend.get_legacy_json(model_category_name)
    if existing_models is None or model_name not in existing_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'",
        )

    try:
        manager.delete_model_legacy(model_category_name, model_name)
        logger.info(f"Deleted legacy model '{model_name}' from category '{model_category_name}'")
    except KeyError as e:
        logger.warning(f"Model '{model_name}' not found during deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'",
        ) from e
    except Exception as e:
        logger.exception(f"Error deleting legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {e!s}",
        ) from e

    return Response(status_code=status.HTTP_204_NO_CONTENT)
