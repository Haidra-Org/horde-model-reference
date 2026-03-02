from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from horde_model_reference import ModelReferenceManager
from horde_model_reference.audit.events import AuditOperation
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
from horde_model_reference.service.shared import (
    APIKeyInvalidException,
    Operation,
    allowed_users,
    auth_against_horde,
    authenticate_queue_requestor,
    httpx_client,
)


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
    route_name: str,
) -> JSONResponse:
    """Create or update a legacy model record.

    When pending queue is enabled, this enqueues the change and returns HTTP 202.
    When pending queue is disabled, this writes directly to backend and returns HTTP 200/201.

    Args:
        manager: The model reference manager.
        category: The model reference category.
        model_name: The name of the model.
        model_record: The model record data.
        operation: Description of operation for logging (e.g., "create", "update").
        apikey: The API key for authentication.
        route_name: The route name for audit metadata.

    Returns:
        JSONResponse with either PendingChangeRecord (202) or the model record (200/201).

    Raises:
        HTTPException: On validation failure or backend error.
    """
    # Reject backend-prefixed names for text_generation: server auto-generates duplicates
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

        if has_legacy_text_backend_prefix(model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model name '{model_name}' contains a backend prefix (aphrodite/, koboldcpp/). "
                    "Submit only the base model name — backend duplicates are generated automatically."
                ),
            )

    # Check if pending queue is enabled
    queue_service = manager.pending_queue_service

    # Validate model existence before proceeding
    model_exists = _check_legacy_model_exists(manager, category, model_name)

    if operation == Operation.create and model_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. Use PUT to update existing models.",
        )
    if operation == Operation.update and not model_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' does not exist in category '{category}'. Use POST to create new models.",
        )

    # Route to queue or direct write based on queue availability
    if queue_service is not None:
        # Pending queue is enabled - enqueue the change
        requestor = await authenticate_queue_requestor(apikey)

        # Convert operation to AuditOperation
        audit_operation = AuditOperation.CREATE if operation == Operation.create else AuditOperation.UPDATE

        # Compute related_models for text_generation so UI can display affected variants
        related_models: list[str] | None = None
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            from horde_model_reference.text_model_duplicates import TextModelDuplicateManager

            related_models = TextModelDuplicateManager.get_variant_names(model_name)

        # Enqueue the change
        change_record = queue_service.enqueue_change(
            category=category,
            model_name=model_name,
            operation=audit_operation,
            payload=model_record.model_dump(mode="json"),
            requestor_id=requestor.user_id,
            requestor_username=requestor.username,
            notes=None,
            request_metadata={"route": route_name},
            related_models=related_models,
        )

        # Return 202 with the pending change record
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=change_record.model_dump(mode="json", exclude_none=True),
        )

    # Pending queue is disabled - write directly to backend
    auth_context = await auth_against_horde(
        apikey,
        httpx_client,
        allowed_user_ids=allowed_users,
    )

    if auth_context is None:
        raise APIKeyInvalidException()

    try:
        manager.backend.update_model_legacy_from_base_model(
            category,
            model_name,
            model_record,
            logical_user_id=auth_context.user_id,
        )
        logger.info(f"{operation.capitalize()} legacy model '{model_name}' in category '{category}'")
    except Exception as e:
        logger.exception(f"Error {operation}ing legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {operation} model: {e!s}",
        ) from e

    # Return appropriate success status
    response_status = status.HTTP_201_CREATED if operation == Operation.create else status.HTTP_200_OK
    return JSONResponse(status_code=response_status, content=model_record.model_dump())


async def _delete_legacy_model(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    apikey: str,
    route_name: str,
) -> JSONResponse:
    """Delete a legacy model record.

    When pending queue is enabled, this enqueues the deletion and returns HTTP 202.
    When pending queue is disabled, this deletes directly from backend and returns HTTP 200.

    Args:
        manager: The model reference manager.
        category: The model reference category.
        model_name: The name of the model to delete.
        apikey: The API key for authentication.
        route_name: The route name for audit metadata.

    Returns:
        JSONResponse with either PendingChangeRecord (202) or empty response (200).

    Raises:
        HTTPException: On validation failure or backend error.
    """
    # Reject backend-prefixed names for text_generation: server auto-generates duplicates
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

        if has_legacy_text_backend_prefix(model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model name '{model_name}' contains a backend prefix (aphrodite/, koboldcpp/). "
                    "Submit only the base model name — backend duplicates are deleted automatically."
                ),
            )

    # Check if pending queue is enabled
    queue_service = manager.pending_queue_service

    # Validate model exists
    existing_models = manager.backend.get_legacy_json(category)
    if existing_models is None or model_name not in existing_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{category}'",
        )

    # Route to queue or direct delete based on queue availability
    if queue_service is not None:
        # Pending queue is enabled - enqueue the deletion
        requestor = await authenticate_queue_requestor(apikey)

        # Compute related_models for text_generation so UI can display affected variants
        related_models: list[str] | None = None
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            from horde_model_reference.text_model_duplicates import TextModelDuplicateManager

            related_models = TextModelDuplicateManager.get_variant_names(model_name)

        # Enqueue the deletion with the existing model data as payload
        change_record = queue_service.enqueue_change(
            category=category,
            model_name=model_name,
            operation=AuditOperation.DELETE,
            payload=existing_models[model_name],
            requestor_id=requestor.user_id,
            requestor_username=requestor.username,
            notes=None,
            request_metadata={"route": route_name},
            related_models=related_models,
        )

        # Return 202 with the pending change record
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=change_record.model_dump(mode="json", exclude_none=True),
        )

    # Pending queue is disabled - delete directly from backend
    auth_context = await auth_against_horde(
        apikey,
        httpx_client,
        allowed_user_ids=allowed_users,
    )

    if auth_context is None:
        raise APIKeyInvalidException()

    try:
        manager.backend.delete_model_legacy(
            category,
            model_name,
            logical_user_id=auth_context.user_id,
        )
        logger.info(f"Deleted legacy model '{model_name}' from category '{category}'")
    except KeyError as e:
        logger.warning(f"Model '{model_name}' not found during deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{category}'",
        ) from e
    except Exception as e:
        logger.exception(f"Error deleting legacy model '{model_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {e!s}",
        ) from e

    return JSONResponse(status_code=status.HTTP_200_OK, content={})
