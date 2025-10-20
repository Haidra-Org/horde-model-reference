from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse, Response
from haidra_core.service_base import ContainsMessage
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyBlipRecord,
    LegacyClipRecord,
    LegacyCodeformerRecord,
    LegacyControlnetRecord,
    LegacyEsrganRecord,
    LegacyGenericRecord,
    LegacyGfpganRecord,
    LegacyMiscellaneousRecord,
    LegacySafetyCheckerRecord,
    LegacyStableDiffusionRecord,
    LegacyTextGenerationRecord,
    get_legacy_model_type,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import Operation, PathVariables, RouteNames, route_registry, v1_prefix

router = APIRouter(
    # prefix="/references",
    responses={404: {"description": "Not found"}},
)


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()


def _check_model_exists(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
) -> bool:
    """Check if a model exists in the given category."""
    existing_models = manager.backend.get_legacy_json(category)
    return existing_models is not None and model_name in existing_models


def _create_or_update_legacy_model(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    model_record: (
        LegacyGenericRecord
        | LegacyStableDiffusionRecord
        | LegacyTextGenerationRecord
        | LegacyBlipRecord
        | LegacyClipRecord
        | LegacyCodeformerRecord
        | LegacyControlnetRecord
        | LegacyEsrganRecord
        | LegacyGfpganRecord
        | LegacySafetyCheckerRecord
        | LegacyMiscellaneousRecord
    ),
    operation: Operation,
) -> None:
    """Create or update a legacy model record.

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
        manager.backend.update_model_legacy_from_base_model(category, model_name, model_record)
        logger.info(f"{operation.capitalize()} legacy model '{model_name}' in category '{category}'")
    except Exception as e:
        logger.exception(f"Error {operation}ing legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {operation} model: {e!s}",
        ) from e


info_route_subpath = "/info"
"""/info"""
route_registry.register_route(
    v1_prefix,
    RouteNames.get_reference_info,
    info_route_subpath,
)


@router.get(
    info_route_subpath,
    summary="Get info about the legacy model reference API",
    operation_id="read_legacy_reference_api_info",
    responses={
        200: {
            "description": "API information",
            "links": {
                "GetAllCategories": {
                    "operationId": "read_legacy_references_names",
                    "description": "Get all available model categories.",
                }
            },
        }
    },
)
async def read_legacy_reference_info() -> ContainsMessage:
    """Get information about the legacy model reference API.

    This is the legacy model reference API, which uses the format originally found at the
    github repositories:
    - https://github.com/Haidra-Org/AI-Horde-image-model-reference
    - https://github.com/Haidra-Org/AI-Horde-text-model-reference
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


@router.get(
    read_reference_route_subpath,
    response_model=list[MODEL_REFERENCE_CATEGORY],
    summary="Get all legacy model reference names",
    operation_id="read_legacy_references_names",
    responses={
        200: {
            "description": "List of all model categories",
            "links": {
                "GetModelsByCategory": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": "$response.body#/*",
                    },
                    "description": "Each category name can be used to retrieve all models in that category.",
                }
            },
        }
    },
)
async def read_legacy_reference_names(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> list[MODEL_REFERENCE_CATEGORY]:
    """Get all available legacy model reference category names.

    Returns a list of all model categories that have legacy format references available.
    """
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
        200: {
            "description": "All models in the category",
            "links": {
                "CreateModelInCategory": {
                    "operationId": "create_legacy_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create a new model in this category.",
                },
                "UpdateModelInCategory": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Update a model in this category.",
                },
            },
        },
        404: {"description": "Model category not found or empty"},
        422: {"description": "Invalid model category"},
    },
    summary="Get a specific legacy model reference by category name",
    operation_id="read_legacy_reference",
)
async def read_legacy_reference(
    model_category_name: MODEL_REFERENCE_CATEGORY | Literal["stable_diffusion"] | str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Get all models in a specific legacy model reference category.

    Returns the complete legacy format JSON for the requested category.

    **Note:** `stable_diffusion` is an alias for `image_generation`.
    """
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


create_model_image_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.image_generation}/create_model"
"""/image_generation/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_image_generation_model,
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
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                    },
                    "description": "Retrieve all image generation models including the newly created one.",
                },
                "UpdateCreatedImageModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                    },
                    "description": "Update the newly created image generation model.",
                },
                "DeleteCreatedImageModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.image_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created image generation model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists"},
        422: {"description": "Validation error"},
    },
    summary="Create a new image generation model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.image_generation),
    operation_id="create_legacy_image_generation_model",
)
async def create_legacy_image_generation_model(
    new_model_record: LegacyStableDiffusionRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new image generation model in legacy format.

    The model name in the request body must not already exist in the image generation category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.image_generation

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_text_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.text_generation}/create_model"
"""/text_generation/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_text_generation_model,
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
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                    },
                    "description": "Retrieve all text generation models including the newly created one.",
                },
                "UpdateCreatedTextModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                    },
                    "description": "Update the newly created text generation model.",
                },
                "DeleteCreatedTextModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.text_generation,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created text generation model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists"},
        422: {"description": "Validation error"},
    },
    summary="Create a new text generation model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.text_generation),
    operation_id="create_legacy_text_generation_model",
)
async def create_legacy_text_generation_model(
    new_model_record: LegacyTextGenerationRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new text generation model in legacy format.

    The model name in the request body must not already exist in the text generation category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.text_generation

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_clip_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.clip}/create_model"
"""/clip/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_clip_model,
    create_model_clip_route_subpath,
)


@router.post(
    create_model_clip_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedClipModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.clip,
                    },
                    "description": "Retrieve all CLIP models including the newly created one.",
                },
                "UpdateCreatedClipModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.clip,
                    },
                    "description": "Update the newly created CLIP model.",
                },
                "DeleteCreatedClipModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.clip,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created CLIP model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new clip model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.clip),
    operation_id="create_legacy_clip_model",
)
async def create_legacy_clip_model(
    new_model_record: LegacyClipRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new CLIP model in legacy format.

    The model name in the request body must not already exist in the clip category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.clip

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_blip_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.blip}/create_model"
"""/blip/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_blip_model,
    create_model_blip_route_subpath,
)


@router.post(
    create_model_blip_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedBlipModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.blip,
                    },
                    "description": "Retrieve all BLIP models including the newly created one.",
                },
                "UpdateCreatedBlipModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.blip,
                    },
                    "description": "Update the newly created BLIP model.",
                },
                "DeleteCreatedBlipModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.blip,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created BLIP model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new BLIP model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.blip),
    operation_id="create_legacy_blip_model",
)
async def create_legacy_blip_model(
    new_model_record: LegacyBlipRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new BLIP model in legacy format.

    The model name in the request body must not already exist in the blip category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.blip

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_codeformer_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.codeformer}/create_model"
"""/codeformer/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_codeformer_model,
    create_model_codeformer_route_subpath,
)


@router.post(
    create_model_codeformer_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedCodeformerModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.codeformer,
                    },
                    "description": "Retrieve all Codeformer models including the newly created one.",
                },
                "UpdateCreatedCodeformerModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.codeformer,
                    },
                    "description": "Update the newly created Codeformer model.",
                },
                "DeleteCreatedCodeformerModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.codeformer,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created Codeformer model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new Codeformer model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.codeformer),
    operation_id="create_legacy_codeformer_model",
)
async def create_legacy_codeformer_model(
    new_model_record: LegacyCodeformerRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new Codeformer model in legacy format.

    The model name in the request body must not already exist in the codeformer category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.codeformer

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_controlnet_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.controlnet}/create_model"
"""/controlnet/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_controlnet_model,
    create_model_controlnet_route_subpath,
)


@router.post(
    create_model_controlnet_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedControlnetModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                    },
                    "description": "Retrieve all ControlNet models including the newly created one.",
                },
                "UpdateCreatedControlnetModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                    },
                    "description": "Update the newly created ControlNet model.",
                },
                "DeleteCreatedControlnetModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.controlnet,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created ControlNet model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new ControlNet model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.controlnet),
    operation_id="create_legacy_controlnet_model",
)
async def create_legacy_controlnet_model(
    new_model_record: LegacyControlnetRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new ControlNet model in legacy format.

    The model name in the request body must not already exist in the controlnet category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.controlnet

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_esrgan_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.esrgan}/create_model"
"""/esrgan/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_esrgan_model,
    create_model_esrgan_route_subpath,
)


@router.post(
    create_model_esrgan_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedEsrganModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.esrgan,
                    },
                    "description": "Retrieve all ESRGAN models including the newly created one.",
                },
                "UpdateCreatedEsrganModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.esrgan,
                    },
                    "description": "Update the newly created ESRGAN model.",
                },
                "DeleteCreatedEsrganModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.esrgan,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created ESRGAN model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new ESRGAN model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.esrgan),
    operation_id="create_legacy_esrgan_model",
)
async def create_legacy_esrgan_model(
    new_model_record: LegacyEsrganRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new ESRGAN model in legacy format.

    The model name in the request body must not already exist in the esrgan category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.esrgan

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_gfpgan_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.gfpgan}/create_model"
"""/gfpgan/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_gfpgan_model,
    create_model_gfpgan_route_subpath,
)


@router.post(
    create_model_gfpgan_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedGfpganModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.gfpgan,
                    },
                    "description": "Retrieve all GFPGAN models including the newly created one.",
                },
                "UpdateCreatedGfpganModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.gfpgan,
                    },
                    "description": "Update the newly created GFPGAN model.",
                },
                "DeleteCreatedGfpganModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.gfpgan,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created GFPGAN model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new GFPGAN model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.gfpgan),
    operation_id="create_legacy_gfpgan_model",
)
async def create_legacy_gfpgan_model(
    new_model_record: LegacyGfpganRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new GFPGAN model in legacy format.

    The model name in the request body must not already exist in the gfpgan category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.gfpgan

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_safety_checker_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.safety_checker}/create_model"
"""/safety_checker/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_safety_checker_model,
    create_model_safety_checker_route_subpath,
)


@router.post(
    create_model_safety_checker_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedSafetyCheckerModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.safety_checker,
                    },
                    "description": "Retrieve all safety checker models including the newly created one.",
                },
                "UpdateCreatedSafetyCheckerModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.safety_checker,
                    },
                    "description": "Update the newly created safety checker model.",
                },
                "DeleteCreatedSafetyCheckerModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.safety_checker,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created safety checker model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new safety checker model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.safety_checker),
    operation_id="create_legacy_safety_checker_model",
)
async def create_legacy_safety_checker_model(
    new_model_record: LegacySafetyCheckerRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new safety checker model in legacy format.

    The model name in the request body must not already exist in the safety_checker category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.safety_checker

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


create_model_miscellaneous_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.miscellaneous}/create_model"
"""/miscellaneous/create_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.create_miscellaneous_model,
    create_model_miscellaneous_route_subpath,
)


@router.post(
    create_model_miscellaneous_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
            "links": {
                "GetCreatedMiscellaneousModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.miscellaneous,
                    },
                    "description": "Retrieve all miscellaneous models including the newly created one.",
                },
                "UpdateCreatedMiscellaneousModel": {
                    "operationId": "update_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.miscellaneous,
                    },
                    "description": "Update the newly created miscellaneous model.",
                },
                "DeleteCreatedMiscellaneousModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": MODEL_REFERENCE_CATEGORY.miscellaneous,
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the newly created miscellaneous model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        409: {"description": "Model already exists (use PUT to update)"},
        422: {"description": "Validation error in request body"},
    },
    summary="Create a new miscellaneous model in legacy format",
    response_model=get_legacy_model_type(MODEL_REFERENCE_CATEGORY.miscellaneous),
    operation_id="create_legacy_miscellaneous_model",
)
async def create_legacy_miscellaneous_model(
    new_model_record: LegacyMiscellaneousRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Create a new miscellaneous model in legacy format.

    The model name in the request body must not already exist in the miscellaneous category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.miscellaneous

    if _check_model_exists(manager, category, model_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )

    _create_or_update_legacy_model(manager, category, model_name, new_model_record, Operation.create)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


update_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/update_model"
"""/{model_category_name}/update_model"""
route_registry.register_route(
    v1_prefix,
    RouteNames.update_model,
    update_model_route_subpath,
)


@router.put(
    "/{model_category_name}/update_model",
    responses={
        200: {
            "description": "Model updated successfully",
            "links": {
                "GetUpdatedModel": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Retrieve all models in the category including the updated one.",
                },
                "CreateModelInCategory": {
                    "operationId": "create_legacy_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create another model in the same category.",
                },
                "DeleteUpdatedModel": {
                    "operationId": "delete_legacy_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                        "model_name": "$request.body#/name",
                    },
                    "description": "Delete the updated model.",
                },
            },
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyGenericRecord,
    summary="Update an existing model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_model",
)
async def update_legacy_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    new_model_record: LegacyGenericRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> JSONResponse:
    """Update an existing model in legacy format.

    ⚠️ **This endpoint is only available when `canonical_format='legacy'` in PRIMARY mode.**

    The model must already exist in the specified category. Use POST to create new models.
    """
    if not manager.backend.supports_legacy_writes():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This instance does not support legacy format write operations. "
            "Legacy CRUD is only available when canonical_format='legacy' in PRIMARY mode. "
            "Use v2 API for write operations when canonical_format='v2'.",
        )

    model_name = new_model_record.name

    if not _check_model_exists(manager, model_category_name, model_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{model_category_name}'. "
            "Use POST to create new models.",
        )

    _create_or_update_legacy_model(manager, model_category_name, model_name, new_model_record, Operation.update)
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/delete_model/{{{PathVariables.model_name}}}"
"""/{model_category_name}/delete_model/{model_name}"""
route_registry.register_route(
    v1_prefix,
    RouteNames.delete_model,
    delete_model_route_subpath,
)


@router.delete(
    delete_model_route_subpath,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Model deleted successfully",
            "links": {
                "GetRemainingModels": {
                    "operationId": "read_legacy_reference",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Retrieve all remaining models in the category after deletion.",
                },
                "CreateModelInCategory": {
                    "operationId": "create_legacy_model",
                    "parameters": {
                        "model_category_name": "$request.path.model_category_name",
                    },
                    "description": "Create a new model in the same category.",
                },
            },
        },
        404: {"description": "Model not found"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    summary="Delete a legacy model entry.",
    operation_id="delete_legacy_model",
)
async def delete_legacy_model(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> Response:
    """Delete a model from a legacy model reference category.

    ⚠️ **This endpoint is only available when `canonical_format='legacy'` in PRIMARY mode.**

    Permanently removes the specified model from the category.
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
        manager.backend.delete_model_legacy(model_category_name, model_name)
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

    return Response(status_code=status.HTTP_200_OK)
