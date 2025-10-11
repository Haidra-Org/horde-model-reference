from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from haidra_core.service_base import ContainsMessage

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import RouteNames, route_registry, v1_prefix

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
