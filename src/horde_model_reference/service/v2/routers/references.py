from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from haidra_core.service_base import ContainsMessage
from loguru import logger
from strenum import StrEnum

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    ControlNetModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)
from horde_model_reference.service.shared import PathVariables, RouteNames, route_registry, v2_prefix
from horde_model_reference.service.v2.models import ErrorResponse, ModelRecordUnion, ModelRecordUnionType

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


class Operation(StrEnum):
    """CRUD operation types."""

    create = "create"
    update = "update"
    delete = "delete"


def _check_model_exists(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
) -> bool:
    """Check if a model exists in the given category."""
    existing_models = manager.get_raw_model_reference_json(category)
    return existing_models is not None and model_name in existing_models


def _create_or_update_v2_model(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    model_record: ModelRecordUnionType,
    operation: Operation,
) -> None:
    """Create or update a v2 model record.

    Args:
        manager: The model reference manager.
        category: The model reference category.
        model_name: The name of the model.
        model_record: The model record data.
        operation: Description of operation for logging (e.g., "create", "update").

    Raises:
        HTTPException: On failure to create/update the model.
    """
    try:
        manager.backend.update_model_from_base_model(category, model_name, model_record)
        logger.info(f"{operation.capitalize()} v2 model '{model_name}' in category '{category}'")
    except Exception as e:
        logger.exception(f"Error {operation}ing v2 model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {operation} model: {e!s}",
        ) from e


info_route_subpath = "/info"
"""/info"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_reference_info,
    info_route_subpath,
)


@router.get(
    info_route_subpath,
    summary="Get info about the v2 model reference API",
    operation_id="read_v2_reference_api_info",
    responses={
        200: {
            "description": "API information",
            "links": {
                "GetAllCategories": {
                    "operationId": "read_v2_references_names",
                    "description": "Get all available model categories.",
                }
            },
        }
    },
)
async def read_v2_reference_info() -> ContainsMessage:
    """Get information about the v2 model reference API.

    This is the v2 model reference API, which uses the new format established by horde_model_reference.
    """
    info = read_v2_reference_info.__doc__ or "No information available."
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


@router.get(
    read_reference_route_subpath,
    response_model=list[MODEL_REFERENCE_CATEGORY],
    summary="Get all v2 model reference names",
    operation_id="read_v2_references_names",
    responses={
        200: {
            "description": "List of all model categories",
            "links": {
                "GetModelsByCategory": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$response.body#/*",
                    },
                    "description": "Each category name can be used to retrieve all models in that category.",
                }
            },
        }
    },
)
async def read_v2_reference_names(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> list[MODEL_REFERENCE_CATEGORY]:
    """Get all available v2 model reference category names.

    Returns a list of all model categories that have v2 format references available.
    """
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
        200: {
            "description": "All models in the category",
            "links": {
                "GetSingleModel": {
                    "operationId": "read_v2_single_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Get a specific model from this category.",
                },
                "CreateModelInCategory": {
                    "operationId": "create_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create a new model in this category.",
                },
            },
        },
        404: {"description": "Model category not found or empty"},
        422: {"description": "Invalid model category"},
    },
    summary="Get a specific v2 model reference by category name",
    operation_id="read_v2_reference",
)
async def read_v2_reference(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Get all models in a specific v2 model reference category.

    Returns the complete v2 format JSON for the requested category.
    """
    raw_json = manager.get_raw_model_reference_json(model_category_name)

    if raw_json is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model category '{model_category_name}' not found",
        )

    return JSONResponse(content=raw_json, media_type="application/json")


single_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.get_single_model,
    single_model_route_subpath,
)


@router.get(
    single_model_route_subpath,
    response_model=dict[str, Any],
    responses={
        200: {
            "description": "Model record data",
            "links": {
                "UpdateThisModel": {
                    "operationId": "update_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.path.model_name",
                    },
                    "description": "Update this model.",
                },
                "DeleteThisModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.path.model_name",
                    },
                    "description": "Delete this model.",
                },
                "GetAllModelsInCategory": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Get all models in this category.",
                },
            },
        },
        404: {"description": "Model category or model not found", "model": ErrorResponse},
    },
    summary="Get a specific model by category and name",
    operation_id="read_v2_single_model",
)
async def read_v2_single_model(
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


create_model_image_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.image_generation}/create_model"
"""/image_generation/create_model"""
route_registry.register_route(
    v2_prefix,
    RouteNames.image_generation_model,
    create_model_image_generation_route_subpath,
)


@router.post(
    create_model_image_generation_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedImageModel": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                    },
                    "description": "Retrieve all image generation models including the newly created one.",
                },
                "UpdateCreatedImageModel": {
                    "operationId": "update_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Update the newly created image generation model.",
                },
                "DeleteCreatedImageModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created image generation model.",
                },
            },
        },
        400: {"description": "Invalid request", "model": ErrorResponse},
        409: {"description": "Model already exists (use PUT to update)", "model": ErrorResponse},
        422: {"description": "Validation error in request body", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Create a new image generation model in v2 format",
    response_model=ImageGenerationModelRecord,
    operation_id="create_v2_image_generation_model",
)
async def create_v2_image_generation_model(
    new_model_record: ImageGenerationModelRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new image generation model in v2 format.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model name in the request body must not already exist in the image generation category.
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

    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.image_generation

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. Use PUT to update existing models.",
        )

    _create_or_update_v2_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_text_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.text_generation}/create_model"
"""/text_generation/create_model"""
route_registry.register_route(
    v2_prefix,
    RouteNames.text_generation_model,
    create_model_text_generation_route_subpath,
)


@router.post(
    create_model_text_generation_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedTextModel": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                    },
                    "description": "Retrieve all text generation models including the newly created one.",
                },
                "UpdateCreatedTextModel": {
                    "operationId": "update_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Update the newly created text generation model.",
                },
                "DeleteCreatedTextModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created text generation model.",
                },
            },
        },
        400: {"description": "Invalid request", "model": ErrorResponse},
        409: {"description": "Model already exists (use PUT to update)", "model": ErrorResponse},
        422: {"description": "Validation error in request body", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Create a new text generation model in v2 format",
    response_model=TextGenerationModelRecord,
    operation_id="create_v2_text_generation_model",
)
async def create_v2_text_generation_model(
    new_model_record: TextGenerationModelRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new text generation model in v2 format.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model name in the request body must not already exist in the text generation category.
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

    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.text_generation

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. Use PUT to update existing models.",
        )

    _create_or_update_v2_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_controlnet_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.controlnet}/create_model"
"""/controlnet/create_model"""
route_registry.register_route(
    v2_prefix,
    RouteNames.controlnet_model,
    create_model_controlnet_route_subpath,
)


@router.post(
    create_model_controlnet_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedControlNetModel": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                    },
                    "description": "Retrieve all ControlNet models including the newly created one.",
                },
                "UpdateCreatedControlNetModel": {
                    "operationId": "update_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Update the newly created ControlNet model.",
                },
                "DeleteCreatedControlNetModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created ControlNet model.",
                },
            },
        },
        400: {"description": "Invalid request", "model": ErrorResponse},
        409: {"description": "Model already exists (use PUT to update)", "model": ErrorResponse},
        422: {"description": "Validation error in request body", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Create a new ControlNet model in v2 format",
    response_model=ControlNetModelRecord,
    operation_id="create_v2_controlnet_model",
)
async def create_v2_controlnet_model(
    new_model_record: ControlNetModelRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new ControlNet model in v2 format.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model name in the request body must not already exist in the ControlNet category.
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

    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.controlnet

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. Use PUT to update existing models.",
        )

    _create_or_update_v2_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


add_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/add"
"""/{model_category_name}/add"""
route_registry.register_route(
    v2_prefix,
    RouteNames.create_model,
    add_model_route_subpath,
)


@router.post(
    add_model_route_subpath,
    response_model=ModelRecordUnion,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedModel": {
                    "operationId": "read_v2_single_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.body#/name",
                    },
                    "description": "Retrieve the newly created model.",
                },
                "GetAllModelsInCategory": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Retrieve all models in the category including the newly created one.",
                },
                "UpdateCreatedModel": {
                    "operationId": "update_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.body#/name",
                    },
                    "description": "Update the newly created model.",
                },
                "DeleteCreatedModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created model.",
                },
            },
        },
        400: {"description": "Invalid request", "model": ErrorResponse},
        409: {"description": "Model already exists (use PUT to update)", "model": ErrorResponse},
        422: {"description": "Validation error in request body", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Create a new model in v2 format",
    operation_id="create_v2_model",
)
async def create_v2_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    new_model_record: ModelRecordUnion,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new model in the specified category.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model name in the request body must not already exist in the specified category.
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

    model_name = new_model_record.name

    if _check_model_exists(manager, model_category_name, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{model_category_name}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_v2_model(manager, model_category_name, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


update_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.update_model,
    update_model_route_subpath,
)


@router.put(
    update_model_route_subpath,
    response_model=ModelRecordUnion,
    responses={
        200: {
            "description": "Model updated successfully",
            "links": {
                "GetUpdatedModel": {
                    "operationId": "read_v2_single_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.path.model_name",
                    },
                    "description": "Retrieve the updated model.",
                },
                "GetAllModelsInCategory": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Retrieve all models in the category including the updated one.",
                },
                "CreateModelInCategory": {
                    "operationId": "create_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create another model in the same category.",
                },
                "DeleteUpdatedModel": {
                    "operationId": "delete_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.path.model_name",
                    },
                    "description": "Delete the updated model.",
                },
            },
        },
        400: {"description": "Invalid request", "model": ErrorResponse},
        404: {"description": "Model not found (use POST to create)", "model": ErrorResponse},
        422: {"description": "Validation error in request body", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Update an existing model in v2 format",
    operation_id="update_v2_model",
)
async def update_v2_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    new_model_record: ModelRecordUnion,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ModelRecordUnion:
    """Update an existing model in v2 format.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model must already exist in the specified category. Use POST to create new models.

    - Preserves original `created_at` and `created_by` metadata
    - Updates `updated_at` timestamp
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

    if not _check_model_exists(manager, model_category_name, model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'. "
            "Use POST to create new models.",
        )

    _create_or_update_v2_model(manager, model_category_name, model_name, new_model_record, Operation.update)
    return manager.get_model(model_category_name, model_name)


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.delete_model,
    delete_model_route_subpath,
)


@router.delete(
    delete_model_route_subpath,
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {
            "description": "Model deleted successfully",
            "links": {
                "GetRemainingModels": {
                    "operationId": "read_v2_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Retrieve all remaining models in the category after deletion.",
                },
                "CreateModelInCategory": {
                    "operationId": "create_v2_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create a new model in the same category.",
                },
            },
        },
        404: {"description": "Model not found in the specified category", "model": ErrorResponse},
        503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
    },
    summary="Delete a v2 model entry",
    operation_id="delete_v2_model",
)
async def delete_v2_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Delete a model from a v2 model reference category.

    ⚠️ **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    Permanently removes the specified model from the category.
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
        manager.backend.delete_model(model_category_name, model_name)
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
