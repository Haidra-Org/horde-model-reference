from fastapi import HTTPException, status
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
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import APIKeyInvalidException, Operation, auth_against_horde, httpx_client


def _check_legacy_model_exists(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
) -> bool:
    """Check if a model exists in the given category."""
    existing_models = manager.backend.get_legacy_json(category)
    return existing_models is not None and model_name in existing_models


async def _create_or_update_legacy_model(
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
    apikey: str,
) -> None:
    """Create or update a legacy model record.

    Args:
        manager: The model reference manager.
        category: The model reference category.
        model_name: The name of the model.
        model_record: The model record data.
        operation: Description of operation for logging (e.g., "create", "update").
        apikey: The API key for authentication.

    Raises:
        HTTPException: On failure to create/update the model.
    """
    authenticated = await auth_against_horde(
        apikey,
        httpx_client,
    )

    if not authenticated:
        raise APIKeyInvalidException()

    model_exists = _check_legacy_model_exists(manager, category, model_name)

    if operation == Operation.create and model_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. "
            "Use PUT to update existing models.",
        )
    if operation == Operation.update and not model_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' does not exist in category '{category}'. " "Use POST to create new models.",
        )

    try:
        manager.backend.update_model_legacy_from_base_model(category, model_name, model_record)
        logger.info(f"{operation.capitalize()} legacy model '{model_name}' in category '{category}'")
    except Exception as e:
        logger.exception(f"Error {operation}ing legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {operation} model: {e!s}",
        ) from e


def get_model_reference_manager() -> ModelReferenceManager:
    """Dependency to get the model reference manager singleton."""
    return ModelReferenceManager()
