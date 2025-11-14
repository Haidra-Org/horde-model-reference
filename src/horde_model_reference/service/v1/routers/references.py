import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from haidra_core.service_base import ContainsMessage

import horde_model_reference.service.v1.routers.create_update as v1_router_create_update
from horde_model_reference import ModelReferenceManager
from horde_model_reference.analytics.text_model_parser import get_base_model_name
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import (
    RouteNames,
    route_registry,
    v1_prefix,
)
from horde_model_reference.service.v1.routers.shared import get_model_reference_manager

router = APIRouter(
    # prefix="/references",
    responses={404: {"description": "Not found"}},
)


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
        }
    },
    tags=["v1"],
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
        }
    },
    tags=["v1"],
)
async def read_legacy_reference_names(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> list[MODEL_REFERENCE_CATEGORY]:
    """Get all available legacy model reference category names.

    Returns a list of all model categories that have legacy format references available.
    """
    return list(manager.backend.get_all_category_file_paths().keys())


get_text_generation_reference_route_subpath = "/text_generation"
"""/text_generation"""
route_registry.register_route(
    v1_prefix,
    RouteNames.get_text_generation_reference,
    get_text_generation_reference_route_subpath,
)


@router.get(
    get_text_generation_reference_route_subpath,
    responses={
        200: {
            "description": "All text generation models",
        },
        404: {"description": "Text generation models not found or empty"},
    },
    summary="Get text generation models with optional grouping field",
    operation_id="read_legacy_text_generation_reference",
    tags=["v1"],
)
async def read_legacy_text_generation_reference(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    include_group: bool = Query(
        default=False,
        description="Include text_model_group field for grouping model variants together",
    ),
) -> Response:
    """Get all text generation models with optional text_model_group field.

    Returns the complete legacy format JSON for text generation models.

    By default, the text_model_group field is not included (legacy format).
    Set include_group=true to dynamically compute and include the text_model_group
    field, which contains the base model group name for grouping model variants
    together (e.g., models with different quantization levels).
    """
    raw_json_string = manager.backend.get_legacy_json_string(MODEL_REFERENCE_CATEGORY.text_generation)

    if not raw_json_string or raw_json_string.strip() in ("", "{}", "null"):
        raise HTTPException(
            status_code=404,
            detail="Text generation models not found or empty",
        )

    # If include_group is True, compute and add text_model_group to each model
    if include_group:
        try:
            models_dict = json.loads(raw_json_string)
            # Add text_model_group to each model entry by computing base name
            for model_name, model_data in models_dict.items():
                base_name = get_base_model_name(model_name)
                model_data["text_model_group"] = base_name
            raw_json_string = json.dumps(models_dict)
        except (json.JSONDecodeError, AttributeError) as e:
            # If parsing fails, log and return original response
            from loguru import logger

            logger.warning(f"Failed to parse JSON for text_model_group computation: {e}")

    return Response(content=raw_json_string, media_type="application/json")


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
        },
        404: {"description": "Model category not found or empty"},
        422: {"description": "Invalid model category"},
    },
    summary="Get a specific legacy model reference by category name",
    operation_id="read_legacy_reference",
    tags=["v1"],
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

    if (
        model_reference_category == MODEL_REFERENCE_CATEGORY.lora
        or model_reference_category == MODEL_REFERENCE_CATEGORY.ti
    ):
        return Response(content="{}", media_type="application/json")

    raw_json_string = manager.backend.get_legacy_json_string(model_reference_category)

    if not raw_json_string or raw_json_string.strip() in ("", "{}", "null"):
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found or is empty")

    return Response(content=raw_json_string, media_type="application/json")


if ModelReferenceManager().backend.supports_legacy_writes():
    router.include_router(v1_router_create_update.router)
