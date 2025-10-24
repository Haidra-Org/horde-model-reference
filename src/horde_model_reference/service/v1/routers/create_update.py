from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from loguru import logger

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyBlipRecord,
    LegacyClipRecord,
    LegacyCodeformerRecord,
    LegacyControlnetRecord,
    LegacyEsrganRecord,
    LegacyGfpganRecord,
    LegacyMiscellaneousRecord,
    LegacySafetyCheckerRecord,
    LegacyStableDiffusionRecord,
    LegacyTextGenerationRecord,
    get_legacy_model_type,
)
from horde_model_reference.service.shared import (
    APIKeyInvalidException,
    Operation,
    PathVariables,
    RouteNames,
    auth_against_horde,
    header_auth_scheme,
    httpx_client,
    route_registry,
    v1_prefix,
)
from horde_model_reference.service.v1.routers.shared import (
    _create_or_update_legacy_model,
    get_model_reference_manager,
)

router = APIRouter(responses={404: {"description": "Not Found"}}, tags=["v1_create_update"])


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/{{{PathVariables.model_name}}}"
"""/{model_category_name}/{model_name}"""
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> Response:
    """Delete a model from a legacy model reference category.

    Permanently removes the specified model from the category.
    """
    authenticated = await auth_against_horde(apikey, httpx_client)

    if not authenticated:
        raise APIKeyInvalidException()

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


# region Image Generation

image_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.image_generation}"
"""/image_generation"""
route_registry.register_route(
    v1_prefix,
    RouteNames.image_generation_model,
    image_generation_route_subpath,
)


@router.post(
    image_generation_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new image generation model in legacy format.

    The model name in the request body must not already exist in the image generation category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.image_generation

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    image_generation_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyStableDiffusionRecord,
    summary="Update an existing model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_model",
)
async def update_legacy_image_generation_model(
    new_model_record: LegacyStableDiffusionRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing image generation model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_category_name = MODEL_REFERENCE_CATEGORY.image_generation
    model_name = new_model_record.name

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion Image Generation

# region Text Generation

text_generation_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.text_generation}"
"""/text_generation"""
route_registry.register_route(
    v1_prefix,
    RouteNames.text_generation_model,
    text_generation_route_subpath,
)


@router.post(
    text_generation_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new text generation model in legacy format.

    The model name in the request body must not already exist in the text generation category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.text_generation

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    text_generation_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyTextGenerationRecord,
    summary="Update an existing model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_text_generation_model",
)
async def update_legacy_text_generation_model(
    new_model_record: LegacyTextGenerationRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing text generation model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.text_generation

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion Text Generation

# region CLIP

clip_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.clip}"
"""/clip"""
route_registry.register_route(
    v1_prefix,
    RouteNames.clip_model,
    clip_route_subpath,
)


@router.post(
    clip_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new CLIP model in legacy format.

    The model name in the request body must not already exist in the clip category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.clip

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    clip_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyClipRecord,
    summary="Update an existing CLIP model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_clip_model",
)
async def update_legacy_clip_model(
    new_model_record: LegacyClipRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing CLIP model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.clip

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion CLIP

# region BLIP

create_model_blip_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.blip}"
"""/blip"""
route_registry.register_route(
    v1_prefix,
    RouteNames.blip_model,
    create_model_blip_route_subpath,
)


@router.post(
    create_model_blip_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new BLIP model in legacy format.

    The model name in the request body must not already exist in the blip category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.blip

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    create_model_blip_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyBlipRecord,
    summary="Update an existing BLIP model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_blip_model",
)
async def update_legacy_blip_model(
    new_model_record: LegacyBlipRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing BLIP model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.blip

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion BLIP

# region Codeformer

codeformer_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.codeformer}"
"""/codeformer"""
route_registry.register_route(
    v1_prefix,
    RouteNames.codeformer_model,
    codeformer_route_subpath,
)


@router.post(
    codeformer_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new Codeformer model in legacy format.

    The model name in the request body must not already exist in the codeformer category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.codeformer

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    codeformer_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyCodeformerRecord,
    summary="Update an existing Codeformer model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_codeformer_model",
)
async def update_legacy_codeformer_model(
    new_model_record: LegacyCodeformerRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing Codeformer model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.codeformer

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion Codeformer

# region ControlNet

controlnet_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.controlnet}"
"""/controlnet"""
route_registry.register_route(
    v1_prefix,
    RouteNames.controlnet_model,
    controlnet_route_subpath,
)


@router.post(
    controlnet_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new ControlNet model in legacy format.

    The model name in the request body must not already exist in the controlnet category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.controlnet

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    controlnet_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyControlnetRecord,
    summary="Update an existing ControlNet model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_controlnet_model",
)
async def update_legacy_controlnet_model(
    new_model_record: LegacyControlnetRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing ControlNet model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.controlnet

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion ControlNet

# region ESRGAN

esrgan_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.esrgan}"
"""/esrgan"""
route_registry.register_route(
    v1_prefix,
    RouteNames.esrgan_model,
    esrgan_route_subpath,
)


@router.post(
    esrgan_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new ESRGAN model in legacy format.

    The model name in the request body must not already exist in the esrgan category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.esrgan

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    esrgan_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyEsrganRecord,
    summary="Update an existing ESRGAN model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_esrgan_model",
)
async def update_legacy_esrgan_model(
    new_model_record: LegacyEsrganRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing ESRGAN model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.esrgan

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion ESRGAN

# region GFPGAN

gfpgan_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.gfpgan}"
"""/gfpgan"""
route_registry.register_route(
    v1_prefix,
    RouteNames.gfpgan_model,
    gfpgan_route_subpath,
)


@router.post(
    gfpgan_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new GFPGAN model in legacy format.

    The model name in the request body must not already exist in the gfpgan category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.gfpgan

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    gfpgan_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyGfpganRecord,
    summary="Update an existing GFPGAN model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_gfpgan_model",
)
async def update_legacy_gfpgan_model(
    new_model_record: LegacyGfpganRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing GFPGAN model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.gfpgan

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion GFPGAN

# region Safety Checker

safety_checker_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.safety_checker}"
"""/safety_checker"""
route_registry.register_route(
    v1_prefix,
    RouteNames.safety_checker_model,
    safety_checker_route_subpath,
)


@router.post(
    safety_checker_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new safety checker model in legacy format.

    The model name in the request body must not already exist in the safety_checker category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.safety_checker

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    safety_checker_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacySafetyCheckerRecord,
    summary="Update an existing safety checker model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_safety_checker_model",
)
async def update_legacy_safety_checker_model(
    new_model_record: LegacySafetyCheckerRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing safety checker model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.safety_checker

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion Safety Checker

# region Miscellaneous

miscellaneous_route_subpath = f"/{MODEL_REFERENCE_CATEGORY.miscellaneous}"
"""/miscellaneous"""
route_registry.register_route(
    v1_prefix,
    RouteNames.miscellaneous_model,
    miscellaneous_route_subpath,
)


@router.post(
    miscellaneous_route_subpath,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Model created successfully",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new miscellaneous model in legacy format.

    The model name in the request body must not already exist in the miscellaneous category.
    """
    model_name = new_model_record.name
    category = MODEL_REFERENCE_CATEGORY.miscellaneous

    await _create_or_update_legacy_model(
        manager,
        category,
        model_name,
        new_model_record,
        Operation.create,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=new_model_record.model_dump())


@router.put(
    miscellaneous_route_subpath,
    responses={
        200: {
            "description": "Model updated successfully",
        },
        400: {"description": "Invalid request"},
        422: {"description": "Validation error"},
        503: {"description": "Service unavailable (not in legacy canonical mode)"},
    },
    response_model=LegacyMiscellaneousRecord,
    summary="Update an existing miscellaneous model in legacy format",
    description=(
        "Update an existing model or create if it doesn't exist (upsert) in legacy format.\n\n"
        "This endpoint is only available when canonical_format='legacy' in PRIMARY mode."
    ),
    operation_id="update_legacy_miscellaneous_model",
)
async def update_legacy_miscellaneous_model(
    new_model_record: LegacyMiscellaneousRecord,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing miscellaneous model in legacy format.

    The model must already exist in the specified category. Use POST to create new models.
    """
    model_name = new_model_record.name
    model_category_name = MODEL_REFERENCE_CATEGORY.miscellaneous

    await _create_or_update_legacy_model(
        manager,
        model_category_name,
        model_name,
        new_model_record,
        Operation.update,
        apikey,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=new_model_record.model_dump())


# endregion Miscellaneous
