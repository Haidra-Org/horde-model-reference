"""Public and queued-management routes for the served image-generation baseline catalog."""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from horde_model_reference import ModelReferenceManager
from horde_model_reference.audit.events import AuditOperation
from horde_model_reference.image_baseline import (
    ImageBaselineCatalog,
    ImageBaselineCatalogMetadata,
    ImageBaselineChangeSet,
    ImageBaselineRecord,
)
from horde_model_reference.image_baseline_store import BaselineConflictError
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.pending_queue import PendingChangeRecord, PendingResourceKind
from horde_model_reference.service.pending_queue.dependencies import require_pending_queue_service
from horde_model_reference.service.shared import (
    authenticate_queue_requestor,
    get_model_reference_manager,
    header_auth_scheme,
)
from horde_model_reference.service.v2.routers.write_validations import assert_primary_write_enabled

router = APIRouter(prefix="/image_generation/baselines")


class ImageBaselinePage(BaseModel):
    """Published baseline index and catalog metadata."""

    items: list[ImageBaselineRecord]
    total: int
    metadata: ImageBaselineCatalogMetadata


def _referenced_baselines(manager: ModelReferenceManager) -> set[str]:
    """Return every baseline name the canonical image generation reference still points at."""
    records = manager.get_raw_model_reference_json(MODEL_REFERENCE_CATEGORY.image_generation) or {}
    return {
        record["baseline"]
        for record in records.values()
        if isinstance(record, dict) and isinstance(record.get("baseline"), str)
    }


def _set_catalog_etag(response: Response, metadata: ImageBaselineCatalogMetadata) -> None:
    """Expose a weak revision ETag for inexpensive consumer freshness checks."""
    response.headers["ETag"] = f'W/"image-baselines-{metadata.revision}"'


@router.get("/", response_model=ImageBaselinePage, tags=["image_baselines"])
def list_baselines(
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ImageBaselinePage:
    """List every published image-generation baseline."""
    baselines = manager.image_baseline_store.list()
    metadata = manager.image_baseline_store.metadata()
    _set_catalog_etag(response, metadata)
    return ImageBaselinePage(items=baselines, total=len(baselines), metadata=metadata)


@router.get("/export", response_model=ImageBaselineCatalog, tags=["image_baselines"])
def export_catalog(
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ImageBaselineCatalog:
    """Export the complete validated published baseline catalog."""
    catalog = manager.image_baseline_store.export()
    _set_catalog_etag(response, catalog.metadata)
    return catalog


@router.get("/{baseline_name}", response_model=ImageBaselineRecord, tags=["image_baselines"])
def get_baseline(
    baseline_name: str,
    response: Response,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> ImageBaselineRecord:
    """Return one current published baseline."""
    record = manager.image_baseline_store.get(baseline_name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Baseline '{baseline_name}' not found.")
    _set_catalog_etag(response, manager.image_baseline_store.metadata())
    return record


@router.post(
    "/change-sets",
    response_model=PendingChangeRecord,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["image_baselines"],
)
async def submit_change_set(
    change_set: ImageBaselineChangeSet,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> PendingChangeRecord:
    """Validate and enqueue one coherent baseline proposal in the shared queue."""
    requestor = await authenticate_queue_requestor(apikey)
    assert_primary_write_enabled(manager)
    try:
        manager.image_baseline_store.preview_change_set(
            change_set,
            referenced_baselines=_referenced_baselines(manager),
        )
    except BaselineConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    resource_id = str(uuid4())
    changed_names = [change.name for change in change_set.changes]
    return require_pending_queue_service(manager).enqueue_change(
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_name=f"baseline:{resource_id}",
        operation=AuditOperation.UPDATE,
        payload=change_set.model_dump(mode="json"),
        requestor_id=requestor.user_id,
        requestor_username=requestor.username,
        notes=change_set.title,
        request_metadata={
            "route": "submit_image_baseline_change_set",
            "title": change_set.title,
            "affected_baselines": changed_names,
        },
        related_models=changed_names,
        resource_kind=PendingResourceKind.IMAGE_BASELINE,
        resource_id=resource_id,
    )
