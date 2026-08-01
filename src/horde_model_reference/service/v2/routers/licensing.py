"""Expose normalized licensing definitions, assets, summaries, and direct editor CRUD."""

from __future__ import annotations

from collections import Counter
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference import ModelReferenceManager
from horde_model_reference.licensing import (
    LICENSE_DATA_SCHEMA_VERSION,
    LicensedAsset,
    LicensedAssetKind,
    LicenseDefinition,
    LicensingRecordMetadata,
    ModelLicensing,
    PermissionStatus,
    unknown_model_licensing,
)
from horde_model_reference.licensing_store import LicensingDatasetMetadata
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.service.shared import (
    WRITE_ERROR_RESPONSES,
    authenticate_licensing_editor,
    get_model_reference_manager,
    header_auth_scheme,
)
from horde_model_reference.service.v2.routers.write_validations import assert_primary_write_enabled

__all__ = ["router"]

router = APIRouter(prefix="/licensing")

DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000


class LicenseDefinitionPage(BaseModel):
    """Represents a paginated collection of normalized license definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[LicenseDefinition]
    total: int
    offset: int
    limit: int
    metadata: LicensingDatasetMetadata


class LicensedAssetView(BaseModel):
    """Represents a unified public view of a model or non-model licensed asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_kind: str
    asset_identifier: str
    display_name: str
    category: MODEL_REFERENCE_CATEGORY | None = None
    source_url: str | None = None
    version: str | None = None
    locations: tuple[str, ...] = ()
    related_assets: tuple[str, ...] = ()
    licensing: ModelLicensing
    definition_urls: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
    metadata: LicensingRecordMetadata | None = None


class LicensedAssetPage(BaseModel):
    """Represents a paginated collection of unified licensed assets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[LicensedAssetView]
    total: int
    offset: int
    limit: int
    metadata: LicensingDatasetMetadata


class LicensingSummary(BaseModel):
    """Represents aggregate counts useful to policy consumers and user interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_assets: int
    commercial_use: dict[PermissionStatus, int]
    redistribution: dict[PermissionStatus, int]
    licenses: dict[str, int]


class LicensingExport(BaseModel):
    """Represents a deterministic snapshot used by documentation and offline consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = LICENSE_DATA_SCHEMA_VERSION
    metadata: LicensingDatasetMetadata
    licenses: list[LicenseDefinition]
    assets: list[LicensedAssetView]
    summary: LicensingSummary


def _definition_urls(licensing: ModelLicensing) -> dict[str, str]:
    """Return stable relative API links for definitions used by an assignment."""
    identifiers = set(licensing.license_ids)
    for file_assignment in licensing.files.values():
        identifiers.update(file_assignment.license_ids)
    return {
        license_id: f"/model_references/v2/licensing/licenses/{quote(license_id, safe='')}"
        for license_id in sorted(identifiers)
    }


def _model_asset_views(manager: ModelReferenceManager) -> list[LicensedAssetView]:
    """Return licensing views for all model records currently available to the manager."""
    model_assets: list[LicensedAssetView] = []
    references = manager.get_all_model_references_or_none()
    for category in sorted(references, key=lambda current_category: current_category.value):
        category_records = references[category] or {}
        for model_name in sorted(category_records, key=str.casefold):
            model_record = category_records[model_name]
            model_licensing = model_record.licensing or unknown_model_licensing()
            model_assets.append(
                LicensedAssetView(
                    asset_kind="model",
                    asset_identifier=f"{category.value}:{model_name}",
                    display_name=model_name,
                    category=category,
                    source_url=model_record.primary_download_url,
                    licensing=model_licensing,
                    definition_urls=_definition_urls(model_licensing),
                ),
            )
    return model_assets


def _non_model_asset_view(asset: LicensedAsset) -> LicensedAssetView:
    """Convert a stored non-model asset into the unified public representation."""
    model_licensing = ModelLicensing(**asset.licensing.model_dump(mode="python"))
    return LicensedAssetView(
        asset_kind=asset.asset_kind.value,
        asset_identifier=asset.asset_identifier,
        display_name=asset.display_name,
        source_url=asset.source_url,
        version=asset.version,
        locations=asset.locations,
        related_assets=asset.related_assets,
        licensing=model_licensing,
        definition_urls=_definition_urls(model_licensing),
        notes=asset.notes,
        metadata=asset.metadata,
    )


def _all_asset_views(manager: ModelReferenceManager) -> list[LicensedAssetView]:
    """Return all model and auxiliary assets in deterministic order."""
    model_assets = _model_asset_views(manager)
    non_model_assets = [_non_model_asset_view(asset) for asset in manager.licensing_store.list_assets()]
    return sorted(
        [*model_assets, *non_model_assets],
        key=lambda asset: (asset.asset_kind.casefold(), asset.asset_identifier.casefold()),
    )


def _summarize_assets(assets: list[LicensedAssetView]) -> LicensingSummary:
    """Create aggregate licensing counts from unified asset views."""
    commercial_counts = Counter(asset.licensing.commercial_use for asset in assets)
    redistribution_counts = Counter(asset.licensing.redistribution for asset in assets)
    license_counts: Counter[str] = Counter()
    for asset in assets:
        license_counts.update(asset.licensing.license_ids)
    return LicensingSummary(
        total_assets=len(assets),
        commercial_use={
            permission_status: commercial_counts[permission_status] for permission_status in PermissionStatus
        },
        redistribution={
            permission_status: redistribution_counts[permission_status] for permission_status in PermissionStatus
        },
        licenses=dict(sorted(license_counts.items())),
    )


def _model_licensing_assignments(manager: ModelReferenceManager) -> list[ModelLicensing]:
    """Return every stored model licensing assignment for reference checks."""
    assignments: list[ModelLicensing] = []
    for category_records in manager.get_all_model_references_or_none().values():
        for model_record in (category_records or {}).values():
            if model_record.licensing is not None:
                assignments.append(model_record.licensing)
    return assignments


@router.get("/licenses", response_model=LicenseDefinitionPage, tags=["v2", "licensing"])
def list_license_definitions(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    include_deprecated: Annotated[bool, Query(description="Include deprecated definitions")] = True,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
) -> LicenseDefinitionPage:
    """Return normalized license definitions with stable pagination."""
    definitions = manager.licensing_store.list_definitions(include_deprecated=include_deprecated)
    return LicenseDefinitionPage(
        items=definitions[offset : offset + limit],
        total=len(definitions),
        offset=offset,
        limit=limit,
        metadata=manager.licensing_store.metadata(),
    )


@router.post(
    "/licenses",
    response_model=LicenseDefinition,
    status_code=status.HTTP_201_CREATED,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def create_license_definition(
    definition: LicenseDefinition,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> LicenseDefinition:
    """Create a normalized definition as an independently allowlisted editor."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    try:
        return manager.licensing_store.create_definition(definition, editor_id=editor.user_id)
    except KeyError as exception:
        raise HTTPException(
            status_code=409,
            detail=f"License definition already exists: {definition.license_id}",
        ) from exception


@router.get("/licenses/{license_id}", response_model=LicenseDefinition, tags=["v2", "licensing"])
def get_license_definition(
    license_id: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LicenseDefinition:
    """Return one normalized license definition."""
    definition = manager.licensing_store.get_definition(license_id)
    if definition is None:
        raise HTTPException(status_code=404, detail=f"License definition not found: {license_id}")
    return definition


@router.put(
    "/licenses/{license_id}",
    response_model=LicenseDefinition,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def replace_license_definition(
    license_id: str,
    definition: LicenseDefinition,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> LicenseDefinition:
    """Replace one normalized definition as an independently allowlisted editor."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    if definition.license_id != license_id:
        raise HTTPException(status_code=400, detail="Path and body license identifiers must match.")
    try:
        return manager.licensing_store.replace_definition(definition, editor_id=editor.user_id)
    except KeyError as exception:
        raise HTTPException(status_code=404, detail=f"License definition not found: {license_id}") from exception


@router.delete(
    "/licenses/{license_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def delete_license_definition(
    license_id: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> Response:
    """Delete a definition only when no model, file, or auxiliary asset references it."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    if manager.licensing_store.definition_is_referenced(license_id, _model_licensing_assignments(manager)):
        raise HTTPException(status_code=409, detail=f"License definition is still referenced: {license_id}")
    try:
        manager.licensing_store.delete_definition(license_id, editor_id=editor.user_id)
    except KeyError as exception:
        raise HTTPException(status_code=404, detail=f"License definition not found: {license_id}") from exception
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/assets", response_model=LicensedAssetPage, tags=["v2", "licensing"])
def list_licensed_assets(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    asset_kind: Annotated[str | None, Query(description="model, custom_node, software_component, or other")] = None,
    category: MODEL_REFERENCE_CATEGORY | None = None,
    license_id: str | None = None,
    commercial_use: PermissionStatus | None = None,
    redistribution: PermissionStatus | None = None,
    name_contains: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = DEFAULT_PAGE_LIMIT,
) -> LicensedAssetPage:
    """Return model and non-model licensed assets with consumer-oriented filters."""
    assets = _all_asset_views(manager)
    if asset_kind is not None:
        assets = [asset for asset in assets if asset.asset_kind == asset_kind]
    if category is not None:
        assets = [asset for asset in assets if asset.category is category]
    if license_id is not None:
        assets = [asset for asset in assets if license_id in asset.licensing.license_ids]
    if commercial_use is not None:
        assets = [asset for asset in assets if asset.licensing.commercial_use is commercial_use]
    if redistribution is not None:
        assets = [asset for asset in assets if asset.licensing.redistribution is redistribution]
    if name_contains is not None:
        normalized_name = name_contains.casefold()
        assets = [asset for asset in assets if normalized_name in asset.display_name.casefold()]
    return LicensedAssetPage(
        items=assets[offset : offset + limit],
        total=len(assets),
        offset=offset,
        limit=limit,
        metadata=manager.licensing_store.metadata(),
    )


@router.post(
    "/assets",
    response_model=LicensedAsset,
    status_code=status.HTTP_201_CREATED,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def create_licensed_asset(
    asset: LicensedAsset,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> LicensedAsset:
    """Create a non-model asset as an independently allowlisted editor."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    try:
        return manager.licensing_store.create_asset(asset, editor_id=editor.user_id)
    except KeyError as exception:
        raise HTTPException(status_code=409, detail=f"Licensed asset already exists: {asset.asset_key}") from exception
    except ValueError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception


@router.get(
    "/assets/{asset_kind}/{asset_identifier:path}",
    response_model=LicensedAsset,
    tags=["v2", "licensing"],
)
def get_licensed_asset(
    asset_kind: LicensedAssetKind,
    asset_identifier: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LicensedAsset:
    """Return one directly managed non-model asset."""
    asset = manager.licensing_store.get_asset(asset_kind=asset_kind, asset_identifier=asset_identifier)
    if asset is None:
        raise HTTPException(status_code=404, detail=f"Licensed asset not found: {asset_kind}:{asset_identifier}")
    return asset


@router.put(
    "/assets/{asset_kind}/{asset_identifier:path}",
    response_model=LicensedAsset,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def replace_licensed_asset(
    asset_kind: LicensedAssetKind,
    asset_identifier: str,
    asset: LicensedAsset,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> LicensedAsset:
    """Replace a non-model asset as an independently allowlisted editor."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    if asset.asset_kind is not asset_kind or asset.asset_identifier != asset_identifier:
        raise HTTPException(status_code=400, detail="Path and body asset identities must match.")
    try:
        return manager.licensing_store.replace_asset(asset, editor_id=editor.user_id)
    except KeyError as exception:
        raise HTTPException(status_code=404, detail=f"Licensed asset not found: {asset.asset_key}") from exception
    except ValueError as exception:
        raise HTTPException(status_code=422, detail=str(exception)) from exception


@router.delete(
    "/assets/{asset_kind}/{asset_identifier:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=WRITE_ERROR_RESPONSES,
    tags=["v2", "licensing"],
)
async def delete_licensed_asset(
    asset_kind: LicensedAssetKind,
    asset_identifier: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
    apikey: Annotated[str, Depends(header_auth_scheme)],
) -> Response:
    """Delete a directly managed non-model asset."""
    assert_primary_write_enabled(manager)
    editor = await authenticate_licensing_editor(apikey)
    try:
        manager.licensing_store.delete_asset(
            asset_kind=asset_kind,
            asset_identifier=asset_identifier,
            editor_id=editor.user_id,
        )
    except KeyError as exception:
        raise HTTPException(
            status_code=404,
            detail=f"Licensed asset not found: {asset_kind}:{asset_identifier}",
        ) from exception
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/models/{category}/{model_name:path}",
    response_model=LicensedAssetView,
    tags=["v2", "licensing"],
)
def get_model_licensing(
    category: MODEL_REFERENCE_CATEGORY,
    model_name: str,
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LicensedAssetView:
    """Return the detailed licensing view for one canonical model."""
    model_record = manager.get_model_or_none(category, model_name)
    if model_record is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {category.value}/{model_name}")
    model_licensing = model_record.licensing or unknown_model_licensing()
    return LicensedAssetView(
        asset_kind="model",
        asset_identifier=f"{category.value}:{model_name}",
        display_name=model_name,
        category=category,
        source_url=model_record.primary_download_url,
        licensing=model_licensing,
        definition_urls=_definition_urls(model_licensing),
    )


@router.get("/summary", response_model=LicensingSummary, tags=["v2", "licensing"])
def get_licensing_summary(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LicensingSummary:
    """Return aggregate counts across model and non-model licensing data."""
    return _summarize_assets(_all_asset_views(manager))


@router.get("/export", response_model=LicensingExport, tags=["v2", "licensing"])
def export_licensing(
    manager: Annotated[ModelReferenceManager, Depends(get_model_reference_manager)],
) -> LicensingExport:
    """Return a deterministic complete snapshot for report generation."""
    assets = _all_asset_views(manager)
    return LicensingExport(
        metadata=manager.licensing_store.metadata(),
        licenses=manager.licensing_store.list_definitions(),
        assets=assets,
        summary=_summarize_assets(assets),
    )
