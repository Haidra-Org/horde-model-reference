from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from haidra_core.service_base import ContainsMessage

from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    CLIPModelRecord,
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)

router = APIRouter(
    # prefix="/references",
    responses={404: {"description": "Not found"}},
)


@router.get("/info")
async def read_reference_info() -> ContainsMessage:
    """Info about the legacy model reference API, as follows.

    This is the  model reference API, which uses the new format established by horde_model_reference.
    """
    info = read_reference_info.__doc__ or "No information available."
    return ContainsMessage(message=info.replace("\n\n", " ").replace("\n", " ").strip())


model_reference_manager = ModelReferenceManager()


@router.get("/model_categories", response_model=list[MODEL_REFERENCE_CATEGORY | str])
async def read_reference_names() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Get all legacy model reference names."""
    return list(model_reference_manager.get_all_model_references().keys())


@router.get(
    "/{model_category_name}",
    response_model=dict[
        str,
        GenericModelRecord | ImageGenerationModelRecord | TextGenerationModelRecord | CLIPModelRecord,
    ],
    responses={
        404: {"description": "Model category not found"},
    },
)
async def read_reference(
    model_category_name: MODEL_REFERENCE_CATEGORY,
) -> JSONResponse:
    """Get a specific model reference by category name."""
    # Get raw cached JSON directly - no pydantic overhead
    raw_json = model_reference_manager.get_raw_model_reference_json(model_category_name)

    if raw_json is None:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found")

    # Return JSONResponse directly with cached JSON data
    return JSONResponse(content=raw_json)
