from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from haidra_core.service_base import ContainsMessage

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

router = APIRouter(
    # prefix="/references",
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def read_legacy_reference_info() -> ContainsMessage:
    """Info about the legacy model reference API, as follows.

    This is the legacy model reference API, which uses the format originally found at the
    github repositories, https://github.com/Haidra-Org/AI-Horde-image-model-reference and
    https://github.com/Haidra-Org/AI-Horde-text-model-reference.
    """
    info = read_legacy_reference_info.__doc__ or "No information available."
    return ContainsMessage(message=info.replace("\n\n", " ").replace("\n", " ").strip())


model_reference_manager = ModelReferenceManager()


@router.get("/model_categories", response_model=list[MODEL_REFERENCE_CATEGORY | str])
async def read_legacy_reference_names() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Get all legacy model reference names."""
    return list(model_reference_manager.backend.get_all_category_file_paths().keys())


@router.get(
    "/{model_category_name}",
    responses={
        404: {"description": "Model category not found or empty"},
    },
)
async def read_legacy_reference(
    model_category_name: MODEL_REFERENCE_CATEGORY | Literal["stable_diffusion"] | str,
) -> Response:
    """Get a specific legacy model reference by category name."""
    if model_category_name == "db.json":
        model_category_name = MODEL_REFERENCE_CATEGORY.text_generation

    if model_category_name.endswith(".json"):
        model_category_name = model_category_name[:-5]

    if model_category_name == "stable_diffusion":
        model_category_name = MODEL_REFERENCE_CATEGORY.image_generation

    model_reference_category = MODEL_REFERENCE_CATEGORY(model_category_name)

    raw_json_string = model_reference_manager.backend.get_legacy_json_string(model_reference_category)

    if not raw_json_string:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found or is empty")

    return Response(content=raw_json_string, media_type="application/json")
