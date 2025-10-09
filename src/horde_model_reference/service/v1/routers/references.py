from typing import Any, Literal

import ujson
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from haidra_core.service_base import ContainsMessage

from horde_model_reference.legacy import LegacyReferenceDownloadManager
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyClipRecord,
    LegacyGenericRecord,
    LegacyStableDiffusionRecord,
    LegacyTextGenerationRecord,
)
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


legacy_reference_download_manager = LegacyReferenceDownloadManager()


@router.get("/model_categories", response_model=list[MODEL_REFERENCE_CATEGORY | str])
async def read_legacy_reference_names() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Get all legacy model reference names."""
    return list(legacy_reference_download_manager.get_all_legacy_model_references_paths().keys())


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
    if model_category_name.endswith(".json"):
        model_category_name = model_category_name[:-5]

    if model_category_name == "stable_diffusion":
        model_category_name = MODEL_REFERENCE_CATEGORY.image_generation

    model_reference_category = MODEL_REFERENCE_CATEGORY(model_category_name)

    # Get raw legacy JSON directly - no pydantic overhead
    raw_json_string = legacy_reference_download_manager.get_legacy_model_reference_json_string(
        model_reference_category
    )

    if not raw_json_string:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found or is empty")

    # Return JSONResponse directly with cached JSON data
    return Response(content=raw_json_string, media_type="application/json")


@router.post("/{model_category_name}/add")
async def add_legacy_reference_record(
    model_category_name: MODEL_REFERENCE_CATEGORY | Literal["stable_diffusion"] | str,
    record: LegacyStableDiffusionRecord | dict[str, Any],
) -> ContainsMessage:
    """Add a new record to a specific legacy model reference by category name.

    Note: This does not save the record to disk. It only adds it to the in-memory representation.
    """
    if model_category_name.endswith(".json"):
        model_category_name = model_category_name[:-5]

    if model_category_name == "stable_diffusion":
        model_category_name = MODEL_REFERENCE_CATEGORY.image_generation

    all_legacy_references = legacy_reference_download_manager.get_all_legacy_model_references()

    try:
        model_reference_category = MODEL_REFERENCE_CATEGORY(model_category_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found") from e

    if model_reference_category not in all_legacy_references:
        raise HTTPException(status_code=404, detail=f"Model category '{model_category_name}' not found")

    legacy_record: LegacyGenericRecord | LegacyStableDiffusionRecord | LegacyTextGenerationRecord | LegacyClipRecord

    if model_reference_category == MODEL_REFERENCE_CATEGORY.image_generation:
        if not isinstance(record, LegacyStableDiffusionRecord):
            try:
                record = LegacyStableDiffusionRecord.model_validate(record)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid record format: {e}") from e
        legacy_record = record
    elif model_reference_category == MODEL_REFERENCE_CATEGORY.text_generation:
        if not isinstance(record, dict):
            raise HTTPException(status_code=400, detail="Invalid record format: must be a dict")
        legacy_record = LegacyTextGenerationRecord.model_validate(record)
    elif model_reference_category == MODEL_REFERENCE_CATEGORY.clip:
        if not isinstance(record, dict):
            raise HTTPException(status_code=400, detail="Invalid record format: must be a dict")
        legacy_record = LegacyClipRecord.model_validate(record)
    else:
        if not isinstance(record, dict):
            raise HTTPException(status_code=400, detail="Invalid record format: must be a dict")
        legacy_record = LegacyGenericRecord.model_validate(record)

    # Add or update the record in the in-memory representation
    category_dict = all_legacy_references[model_reference_category]
    if category_dict is None:
        category_dict = {}
        all_legacy_references[model_reference_category] = category_dict

    category_dict[legacy_record.name] = legacy_record

    return ContainsMessage(message=f"Record '{legacy_record.name}' added/updated in category '{model_category_name}'")
