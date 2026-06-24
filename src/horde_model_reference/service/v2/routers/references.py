from types import GenericAlias
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from haidra_core.service_base import ContainsMessage

from horde_model_reference import ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    GenericModelRecord,
    TextGenerationModelRecord,
)
from horde_model_reference.pending_queue import (
    PendingChangeRecord,
    PendingChangeStatus,
    PendingQueueFilter,
    PendingQueueService,
)
from horde_model_reference.pending_queue.materialize import materialize_pending_records
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    READ_ERROR_RESPONSES,
    ErrorResponse,
    PathVariables,
    RouteNames,
    authenticate_queue_reader,
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
    route_registry,
    v2_prefix,
    validate_model_name,
)
from horde_model_reference.service.v2.models import ModelRecordUnion
from horde_model_reference.service.v2.routers.write_validations import assert_v2_write_enabled

router = APIRouter(
    responses={**READ_ERROR_RESPONSES},
)


def _check_model_exists(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
) -> bool:
    """Check if a model exists in the given category."""
    existing_models = manager.get_raw_model_reference_json(category)
    return existing_models is not None and model_name in existing_models


def _model_payload(record: GenericModelRecord) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)


def _enqueue_pending_change(
    *,
    queue_service: PendingQueueService,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    operation: AuditOperation,
    payload: dict[str, Any] | None,
    requestor_id: str,
    requestor_username: str,
    request_metadata: dict[str, Any] | None = None,
    related_models: list[str] | None = None,
) -> PendingChangeRecord:
    return queue_service.enqueue_change(
        category=category,
        model_name=model_name,
        operation=operation,
        payload=payload,
        requestor_id=requestor_id,
        requestor_username=requestor_username,
        notes=None,
        request_metadata=request_metadata,
        related_models=related_models,
    )


def _queue_response(record: PendingChangeRecord, *, status_code: int = status.HTTP_202_ACCEPTED) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=record.model_dump(mode="json", exclude_none=True))


def _preserve_created_metadata(
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    model_record: GenericModelRecord,
) -> GenericModelRecord:
    """Copy created_* metadata fields from the stored record into the new payload."""
    existing_models = manager.get_raw_model_reference_json(category)
    if not existing_models:
        return model_record

    existing_model = existing_models.get(model_name)
    if not isinstance(existing_model, dict):
        return model_record

    metadata = existing_model.get("metadata")
    if not isinstance(metadata, dict):
        return model_record

    preserved_fields: dict[str, Any] = {}
    for field in ("created_at", "created_by"):
        value = metadata.get(field)
        if value is not None:
            preserved_fields[field] = value

    if not preserved_fields:
        return model_record

    new_metadata = model_record.metadata.model_copy(update=preserved_fields)
    return model_record.model_copy(update={"metadata": new_metadata})


def _queue_change(
    *,
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    operation: AuditOperation,
    payload: dict[str, Any] | None,
    requestor_id: str,
    requestor_username: str,
    request_metadata: dict[str, Any],
    related_models: list[str] | None = None,
) -> PendingChangeRecord:
    queue_service = require_pending_queue_service(manager)
    return _enqueue_pending_change(
        queue_service=queue_service,
        category=category,
        model_name=model_name,
        operation=operation,
        payload=payload,
        requestor_id=requestor_id,
        requestor_username=requestor_username,
        request_metadata=request_metadata,
        related_models=related_models,
    )


async def _queue_model_record_request(
    *,
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_record: GenericModelRecord,
    apikey: str,
    operation: AuditOperation,
    route_name: str,
) -> JSONResponse:
    requestor = await authenticate_queue_requestor(apikey)
    model_name = model_record.name
    assert_v2_write_enabled(manager, category)
    validate_model_name(model_name)

    # Reject backend-prefixed names for text_generation: server auto-generates duplicates
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

        if has_legacy_text_backend_prefix(model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model name '{model_name}' contains a backend prefix (aphrodite/, koboldcpp/). "
                    "Submit only the base model name \u2014 backend duplicates are generated automatically."
                ),
            )

    # Auto-set text_model_group if not provided for text_generation models
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.analytics.text_model_parser import get_base_model_name

        if isinstance(model_record, TextGenerationModelRecord) and model_record.text_model_group is None:
            computed_group = get_base_model_name(model_name)
            model_record = model_record.model_copy(update={"text_model_group": computed_group})

    model_exists = _check_model_exists(manager, category, model_name)

    if operation is AuditOperation.CREATE and model_exists:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model '{model_name}' already exists in category '{category}'. Use PUT to update existing models.",
        )

    if operation is AuditOperation.UPDATE and not model_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{category}'. Use POST to create new models.",
        )

    if operation is AuditOperation.UPDATE:
        model_record = _preserve_created_metadata(manager, category, model_name, model_record)

    # Compute related_models for text_generation so UI can display affected variants
    related_models: list[str] | None = None
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_model_duplicates import TextModelDuplicateManager

        related_models = TextModelDuplicateManager.get_variant_names(model_name)

    change = _queue_change(
        manager=manager,
        category=category,
        model_name=model_name,
        operation=operation,
        payload=_model_payload(model_record),
        requestor_id=requestor.user_id,
        requestor_username=requestor.username,
        request_metadata={"route": route_name},
        related_models=related_models,
    )
    return _queue_response(change)


async def _queue_delete_request(
    *,
    manager: ModelReferenceManager,
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    apikey: str,
    route_name: str,
) -> JSONResponse:
    requestor = await authenticate_queue_requestor(apikey)
    assert_v2_write_enabled(manager, category)
    validate_model_name(model_name)

    # Reject backend-prefixed names for text_generation: server auto-generates duplicates
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix

        if has_legacy_text_backend_prefix(model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model name '{model_name}' contains a backend prefix (aphrodite/, koboldcpp/). "
                    "Submit only the base model name \u2014 backend duplicates are deleted automatically."
                ),
            )

    existing_models = manager.get_raw_model_reference_json(category)
    if existing_models is None or model_name not in existing_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found in category '{category}'",
        )

    # Compute related_models for text_generation so UI can display affected variants
    related_models: list[str] | None = None
    if category == MODEL_REFERENCE_CATEGORY.text_generation:
        from horde_model_reference.text_model_duplicates import TextModelDuplicateManager

        related_models = TextModelDuplicateManager.get_variant_names(model_name)

    change = _queue_change(
        manager=manager,
        category=category,
        model_name=model_name,
        operation=AuditOperation.DELETE,
        payload=existing_models[model_name],
        requestor_id=requestor.user_id,
        requestor_username=requestor.username,
        request_metadata={"route": route_name},
        related_models=related_models,
    )
    return _queue_response(change)


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


single_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/model/{{{PathVariables.model_name}}}"
"""/{model_category_name}/model/{model_name}"""
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


pending_route_subpath = f"/{{{PathVariables.model_category_name}}}/pending"
"""/{model_category_name}/pending"""


@router.get(
    pending_route_subpath,
    response_model=dict[str, ModelRecordUnion],
    responses={
        200: {"description": "Pending (beta) models in the category, materialized as v2 records"},
        401: {"description": "Invalid API key", "model": ErrorResponse},
        503: {"description": "Pending queue not enabled on this deployment", "model": ErrorResponse},
    },
    summary="Get pending (beta) models for a category",
    operation_id="read_v2_reference_pending",
)
async def read_v2_reference_pending(
    model_category_name: MODEL_REFERENCE_CATEGORY,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Return ``PENDING``/``APPROVED`` create/update changes as ready-to-use v2 records.

    These are "beta" models awaiting (or pending) approval into the canonical reference.
    The payload is keyed by model name and shaped exactly like the canonical category
    endpoint, so a REPLICA client can overlay it onto canonical data without any format
    handling. ``DELETE`` changes are never surfaced (a beta cannot remove a live model),
    and when several changes target one name the most recent wins.

    Reader authentication is required (any valid Horde API key), matching the other
    pending-queue read surfaces.
    """
    await authenticate_queue_reader(apikey)
    queue_service = require_pending_queue_service(manager)

    page = queue_service.list_changes(
        queue_filter=PendingQueueFilter(
            categories={model_category_name},
            statuses={PendingChangeStatus.PENDING, PendingChangeStatus.APPROVED},
        ),
    )
    records = materialize_pending_records(
        model_category_name,
        page.items,
        domain=horde_model_reference_settings.canonical_format,
    )
    return JSONResponse(content=records, media_type="application/json")


# ---------------------------------------------------------------------------
# Typed per-category write/read routes
#
# Every category has a concrete record type registered in ``MODEL_RECORD_TYPE_LOOKUP``.
# We generate one POST (create), one PUT (update), and two GET (read) operations per
# category so the OpenAPI schema and generated SDK clients see the concrete record
# type for each category instead of a discriminated union. Reads return the raw stored
# JSON via passthrough (no ``response_model`` filtering); ``response_model`` is declared
# purely for documentation. The generic enum-bound endpoints defined below remain as a
# uniform fallback and carry the stable ``operation_id``s referenced by OpenAPI links.
# ---------------------------------------------------------------------------

_CATEGORY_CREATE_ROUTE_NAMES: dict[MODEL_REFERENCE_CATEGORY, RouteNames] = {
    MODEL_REFERENCE_CATEGORY.image_generation: RouteNames.image_generation_model,
    MODEL_REFERENCE_CATEGORY.text_generation: RouteNames.text_generation_model,
    MODEL_REFERENCE_CATEGORY.blip: RouteNames.blip_model,
    MODEL_REFERENCE_CATEGORY.clip: RouteNames.clip_model,
    MODEL_REFERENCE_CATEGORY.codeformer: RouteNames.codeformer_model,
    MODEL_REFERENCE_CATEGORY.controlnet: RouteNames.controlnet_model,
    MODEL_REFERENCE_CATEGORY.esrgan: RouteNames.esrgan_model,
    MODEL_REFERENCE_CATEGORY.gfpgan: RouteNames.gfpgan_model,
    MODEL_REFERENCE_CATEGORY.safety_checker: RouteNames.safety_checker_model,
    MODEL_REFERENCE_CATEGORY.miscellaneous: RouteNames.miscellaneous_model,
}

_CATEGORY_UPDATE_ROUTE_NAMES: dict[MODEL_REFERENCE_CATEGORY, RouteNames] = {
    MODEL_REFERENCE_CATEGORY.image_generation: RouteNames.update_image_generation_model,
    MODEL_REFERENCE_CATEGORY.text_generation: RouteNames.update_text_generation_model,
    MODEL_REFERENCE_CATEGORY.controlnet: RouteNames.update_controlnet_model,
}

_TYPED_WRITE_RESPONSES: dict[int | str, dict[str, Any]] = {
    202: {"description": "Model change queued for approval"},
    400: {"description": "Invalid request", "model": ErrorResponse},
    404: {"description": "Model not found", "model": ErrorResponse},
    409: {"description": "Model already exists (use PUT to update)", "model": ErrorResponse},
    422: {"description": "Validation error in request body", "model": ErrorResponse},
    503: {"description": "Service unavailable (v2 canonical mode required)", "model": ErrorResponse},
}


def _register_typed_category_routes(
    category: MODEL_REFERENCE_CATEGORY,
    record_type: type[GenericModelRecord],
) -> None:
    """Register concrete-schema POST/PUT/GET routes for a single category.

    The request body is annotated as ``GenericModelRecord`` for static analysis, then the
    concrete ``record_type`` is injected into ``__annotations__`` so FastAPI documents and
    validates against the category's specific record schema.
    """
    cat = category.value
    create_path = f"/{cat}"
    item_path = f"/{cat}/model/{{{PathVariables.model_name}}}"

    async def create_handler(
        new_model_record: GenericModelRecord,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> JSONResponse:
        return await _queue_model_record_request(
            manager=manager,
            category=category,
            model_record=new_model_record,
            apikey=apikey,
            operation=AuditOperation.CREATE,
            route_name=f"create_v2_{cat}",
        )

    async def update_handler(
        model_name: str,
        new_model_record: GenericModelRecord,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
        apikey: Annotated[str, Depends(header_auth_scheme)],
    ) -> JSONResponse:
        if new_model_record.name != model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model name in the path must match the payload when queuing updates.",
            )
        return await _queue_model_record_request(
            manager=manager,
            category=category,
            model_record=new_model_record,
            apikey=apikey,
            operation=AuditOperation.UPDATE,
            route_name=f"update_v2_{cat}",
        )

    async def get_all_handler(
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    ) -> JSONResponse:
        raw_json = manager.get_raw_model_reference_json(category)
        if raw_json is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model category '{category}' not found",
            )
        return JSONResponse(content=raw_json, media_type="application/json")

    async def get_one_handler(
        model_name: str,
        manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    ) -> JSONResponse:
        raw_json = manager.get_raw_model_reference_json(category)
        if raw_json is None or model_name not in raw_json:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found in category '{category}'",
            )
        return JSONResponse(content=raw_json[model_name], media_type="application/json")

    # Swap the documented/validated body type to the category's concrete record.
    create_handler.__annotations__["new_model_record"] = record_type
    update_handler.__annotations__["new_model_record"] = record_type

    if category in _CATEGORY_CREATE_ROUTE_NAMES:
        route_registry.register_route(v2_prefix, _CATEGORY_CREATE_ROUTE_NAMES[category], create_path)
    if category in _CATEGORY_UPDATE_ROUTE_NAMES:
        route_registry.register_route(v2_prefix, _CATEGORY_UPDATE_ROUTE_NAMES[category], item_path)

    router.add_api_route(
        create_path,
        create_handler,
        methods=["POST"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=PendingChangeRecord,
        responses=_TYPED_WRITE_RESPONSES,
        summary=f"Create a new {cat} model",
        operation_id=f"create_v2_{cat}",
        tags=["v2"],
    )
    router.add_api_route(
        item_path,
        update_handler,
        methods=["PUT"],
        status_code=status.HTTP_202_ACCEPTED,
        response_model=PendingChangeRecord,
        responses=_TYPED_WRITE_RESPONSES,
        summary=f"Update an existing {cat} model",
        operation_id=f"update_v2_{cat}",
        tags=["v2"],
    )
    router.add_api_route(
        create_path,
        get_all_handler,
        methods=["GET"],
        response_model=GenericAlias(dict, (str, record_type)),
        responses={404: {"description": "Category not found or empty", "model": ErrorResponse}},
        summary=f"Get all {cat} models",
        operation_id=f"read_v2_{cat}_all",
        tags=["v2"],
    )
    router.add_api_route(
        item_path,
        get_one_handler,
        methods=["GET"],
        response_model=record_type,
        responses={404: {"description": "Model not found", "model": ErrorResponse}},
        summary=f"Get a specific {cat} model",
        operation_id=f"read_v2_{cat}_one",
        tags=["v2"],
    )


for _category, _record_type in MODEL_RECORD_TYPE_LOOKUP.items():
    if isinstance(_category, MODEL_REFERENCE_CATEGORY):
        _register_typed_category_routes(_category, _record_type)


add_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/add"
"""/{model_category_name}/add"""
route_registry.register_route(
    v2_prefix,
    RouteNames.create_model,
    add_model_route_subpath,
)


@router.post(
    add_model_route_subpath,
    response_model=PendingChangeRecord,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {
            "description": "Model change queued for approval",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Create a new model in the specified category.

    **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model name in the request body must not already exist in the specified category.
    """
    return await _queue_model_record_request(
        manager=manager,
        category=model_category_name,
        model_record=new_model_record,
        apikey=apikey,
        operation=AuditOperation.CREATE,
        route_name="create_v2_model",
    )


# Typed per-category update routes are generated by _register_typed_category_routes above.


update_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/model/{{{PathVariables.model_name}}}"
"""/{model_category_name}/model/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.update_model,
    update_model_route_subpath,
)


@router.put(
    update_model_route_subpath,
    response_model=PendingChangeRecord,
    responses={
        202: {
            "description": "Model change queued for approval",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Update an existing model in v2 format.

    **This endpoint is only available when `canonical_format='v2'` in PRIMARY mode.**

    The model must already exist in the specified category. Use POST to create new models.

    - Preserves original `created_at` and `created_by` metadata
    - Updates `updated_at` timestamp
    """
    if new_model_record.name != model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name in the path must match the payload when queuing updates.",
        )

    return await _queue_model_record_request(
        manager=manager,
        category=model_category_name,
        model_record=new_model_record,
        apikey=apikey,
        operation=AuditOperation.UPDATE,
        route_name="update_v2_model",
    )


delete_model_route_subpath = f"/{{{PathVariables.model_category_name}}}/model/{{{PathVariables.model_name}}}"
"""/{model_category_name}/model/{model_name}"""
route_registry.register_route(
    v2_prefix,
    RouteNames.delete_model,
    delete_model_route_subpath,
)


@router.delete(
    delete_model_route_subpath,
    response_model=PendingChangeRecord,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {
            "description": "Model change queued for approval",
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
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> JSONResponse:
    """Queue deletion of a model from a v2 model reference category."""
    return await _queue_delete_request(
        manager=manager,
        category=model_category_name,
        model_name=model_name,
        apikey=apikey,
        route_name="delete_v2_model",
    )
