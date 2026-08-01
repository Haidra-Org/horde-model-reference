"""Persist normalized license definitions and non-model asset assignments.

``LicensingStore`` is the PRIMARY authority for auxiliary licensing data.  It
uses versioned JSON envelopes and atomic writes while keeping model assignments
inside their canonical v2 model records.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference.licensing import (
    LICENSE_DATA_SCHEMA_VERSION,
    LicenseAssignment,
    LicensedAsset,
    LicensedAssetKind,
    LicenseDefinition,
    LicensingRecordMetadata,
    ModelLicensing,
)
from horde_model_reference.util import atomic_write_json

__all__ = ["LicensingDatasetMetadata", "LicensingStore"]

_LICENSE_DEFINITIONS_FILENAME = "licenses.json"
_LICENSED_ASSETS_FILENAME = "assets.json"
_LICENSING_AUDIT_FILENAME = "audit.jsonl"


class LicensingDatasetMetadata(BaseModel):
    """Represents revision metadata for the two auxiliary licensing datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = LICENSE_DATA_SCHEMA_VERSION
    revision: int = Field(default=1, ge=1)
    updated_at: int | None = None


class _LicenseDefinitionsEnvelope(LicensingDatasetMetadata):
    """Represents the serialized license-definition dataset."""

    records: dict[str, LicenseDefinition] = Field(default_factory=dict)


class _LicensedAssetsEnvelope(LicensingDatasetMetadata):
    """Represents the serialized non-model asset dataset."""

    records: dict[str, LicensedAsset] = Field(default_factory=dict)


class _LicensingAuditEvent(BaseModel):
    """Represents one append-only direct licensing mutation event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: int
    editor_id: str
    operation: str
    record_kind: str
    record_key: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


class LicensingStore:
    """Provide thread-safe CRUD over normalized licensing JSON stores."""

    def __init__(self, *, root_path: Path, bootstrap_path: Path | None = None, writable: bool = True) -> None:
        """Create a licensing store and load its current datasets.

        Args:
            root_path: Directory containing mutable licensing JSON files.
            bootstrap_path: Optional packaged directory used when mutable files are absent.
            writable: Whether mutation methods and bootstrap persistence are allowed.
        """
        self._root_path = root_path
        self._bootstrap_path = bootstrap_path
        self._writable = writable
        self._lock = RLock()
        self._definitions = self._load_definitions()
        self._assets = self._load_assets()
        if writable:
            self._persist_missing_bootstrap_files()

    @property
    def writable(self) -> bool:
        """Return whether this store accepts direct mutations."""
        return self._writable

    def metadata(self) -> LicensingDatasetMetadata:
        """Return combined dataset revision metadata."""
        with self._lock:
            return LicensingDatasetMetadata(
                revision=max(self._definitions.revision, self._assets.revision),
                updated_at=max(
                    self._definitions.updated_at or 0,
                    self._assets.updated_at or 0,
                )
                or None,
            )

    def list_definitions(self, *, include_deprecated: bool = True) -> list[LicenseDefinition]:
        """Return license definitions in stable identifier order.

        Args:
            include_deprecated: Whether deprecated definitions should be included.

        Returns:
            Deep copies of the matching definitions.
        """
        with self._lock:
            definitions = self._definitions.records.values()
            return [
                definition.model_copy(deep=True)
                for definition in sorted(definitions, key=lambda record: record.license_id.casefold())
                if include_deprecated or not definition.deprecated
            ]

    def get_definition(self, license_id: str) -> LicenseDefinition | None:
        """Return one license definition, or ``None`` when it is unknown."""
        with self._lock:
            definition = self._definitions.records.get(license_id)
            return definition.model_copy(deep=True) if definition is not None else None

    def create_definition(self, definition: LicenseDefinition, *, editor_id: str) -> LicenseDefinition:
        """Create and persist a new definition.

        Args:
            definition: Definition supplied by the editor.
            editor_id: Authenticated Horde user identifier.

        Returns:
            The persisted definition with lifecycle metadata.

        Raises:
            PermissionError: If the store is read-only.
            KeyError: If the identifier already exists.
        """
        self._require_writable()
        with self._lock:
            if definition.license_id in self._definitions.records:
                raise KeyError(definition.license_id)
            timestamp = _now_timestamp()
            persisted = definition.model_copy(
                update={
                    "metadata": LicensingRecordMetadata(
                        created_at=timestamp,
                        created_by=editor_id,
                        updated_at=timestamp,
                        updated_by=editor_id,
                    ),
                },
            )
            self._definitions.records[persisted.license_id] = persisted
            self._advance_definitions(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="create",
                record_kind="license_definition",
                record_key=persisted.license_id,
                after=persisted,
            )
            return persisted.model_copy(deep=True)

    def replace_definition(self, definition: LicenseDefinition, *, editor_id: str) -> LicenseDefinition:
        """Replace and persist an existing definition while preserving creation metadata."""
        self._require_writable()
        with self._lock:
            existing = self._definitions.records.get(definition.license_id)
            if existing is None:
                raise KeyError(definition.license_id)
            timestamp = _now_timestamp()
            persisted = definition.model_copy(
                update={
                    "metadata": LicensingRecordMetadata(
                        revision=existing.metadata.revision + 1,
                        created_at=existing.metadata.created_at,
                        created_by=existing.metadata.created_by,
                        updated_at=timestamp,
                        updated_by=editor_id,
                    ),
                },
            )
            self._definitions.records[persisted.license_id] = persisted
            self._advance_definitions(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="update",
                record_kind="license_definition",
                record_key=persisted.license_id,
                before=existing,
                after=persisted,
            )
            return persisted.model_copy(deep=True)

    def delete_definition(self, license_id: str, *, editor_id: str) -> LicenseDefinition:
        """Delete an unreferenced license definition.

        Raises:
            PermissionError: If the store is read-only.
            KeyError: If the definition does not exist.
            ValueError: If a non-model asset still references the definition.
        """
        self._require_writable()
        with self._lock:
            existing = self._definitions.records.get(license_id)
            if existing is None:
                raise KeyError(license_id)
            referencing_assets = [
                asset.asset_key for asset in self._assets.records.values() if license_id in asset.licensing.license_ids
            ]
            if referencing_assets:
                raise ValueError(f"License is referenced by assets: {', '.join(sorted(referencing_assets))}")
            del self._definitions.records[license_id]
            timestamp = _now_timestamp()
            self._advance_definitions(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="delete",
                record_kind="license_definition",
                record_key=license_id,
                before=existing,
            )
            return existing.model_copy(deep=True)

    def list_assets(self) -> list[LicensedAsset]:
        """Return non-model assets in stable key order."""
        with self._lock:
            return [
                self._assets.records[asset_key].model_copy(deep=True) for asset_key in sorted(self._assets.records)
            ]

    def get_asset(self, *, asset_kind: LicensedAssetKind, asset_identifier: str) -> LicensedAsset | None:
        """Return one non-model asset, or ``None`` when it is unknown."""
        asset_key = f"{asset_kind.value}:{asset_identifier}"
        with self._lock:
            asset = self._assets.records.get(asset_key)
            return asset.model_copy(deep=True) if asset is not None else None

    def create_asset(self, asset: LicensedAsset, *, editor_id: str) -> LicensedAsset:
        """Create and persist a non-model licensed asset."""
        self._require_writable()
        with self._lock:
            if asset.asset_key in self._assets.records:
                raise KeyError(asset.asset_key)
            self.validate_assignment(asset.licensing)
            timestamp = _now_timestamp()
            persisted = asset.model_copy(
                update={
                    "metadata": LicensingRecordMetadata(
                        created_at=timestamp,
                        created_by=editor_id,
                        updated_at=timestamp,
                        updated_by=editor_id,
                    ),
                },
            )
            self._assets.records[persisted.asset_key] = persisted
            self._advance_assets(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="create",
                record_kind="licensed_asset",
                record_key=persisted.asset_key,
                after=persisted,
            )
            return persisted.model_copy(deep=True)

    def replace_asset(self, asset: LicensedAsset, *, editor_id: str) -> LicensedAsset:
        """Replace and persist an existing non-model licensed asset."""
        self._require_writable()
        with self._lock:
            existing = self._assets.records.get(asset.asset_key)
            if existing is None:
                raise KeyError(asset.asset_key)
            self.validate_assignment(asset.licensing)
            timestamp = _now_timestamp()
            persisted = asset.model_copy(
                update={
                    "metadata": LicensingRecordMetadata(
                        revision=existing.metadata.revision + 1,
                        created_at=existing.metadata.created_at,
                        created_by=existing.metadata.created_by,
                        updated_at=timestamp,
                        updated_by=editor_id,
                    ),
                },
            )
            self._assets.records[persisted.asset_key] = persisted
            self._advance_assets(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="update",
                record_kind="licensed_asset",
                record_key=persisted.asset_key,
                before=existing,
                after=persisted,
            )
            return persisted.model_copy(deep=True)

    def delete_asset(
        self,
        *,
        asset_kind: LicensedAssetKind,
        asset_identifier: str,
        editor_id: str,
    ) -> LicensedAsset:
        """Delete and return one non-model licensed asset."""
        self._require_writable()
        asset_key = f"{asset_kind.value}:{asset_identifier}"
        with self._lock:
            existing = self._assets.records.pop(asset_key, None)
            if existing is None:
                raise KeyError(asset_key)
            timestamp = _now_timestamp()
            self._advance_assets(timestamp)
            self._audit(
                editor_id=editor_id,
                operation="delete",
                record_kind="licensed_asset",
                record_key=asset_key,
                before=existing,
            )
            return existing.model_copy(deep=True)

    def validate_assignment(self, assignment: LicenseAssignment | ModelLicensing) -> None:
        """Validate that every identifier in an assignment resolves to the catalog.

        Args:
            assignment: Model or non-model assignment to validate.

        Raises:
            ValueError: If an identifier is missing or deprecated.
        """
        with self._lock:
            assignments: list[LicenseAssignment] = [assignment]
            if isinstance(assignment, ModelLicensing):
                assignments.extend(assignment.files.values())
            for scoped_assignment in assignments:
                for license_id in scoped_assignment.license_ids:
                    definition = self._definitions.records.get(license_id)
                    if definition is None:
                        raise ValueError(f"Unknown license definition: {license_id}")
                    if definition.deprecated:
                        raise ValueError(f"Deprecated license definition cannot be newly assigned: {license_id}")

    def definition_is_referenced(self, license_id: str, model_assignments: list[ModelLicensing]) -> bool:
        """Return whether a definition is referenced by model or non-model assignments."""
        with self._lock:
            if any(license_id in asset.licensing.license_ids for asset in self._assets.records.values()):
                return True
        for model_assignment in model_assignments:
            if license_id in model_assignment.license_ids:
                return True
            if any(license_id in file_assignment.license_ids for file_assignment in model_assignment.files.values()):
                return True
        return False

    def refresh_replica_export(self, export_payload: dict[str, Any]) -> None:
        """Replace cached auxiliary datasets from a validated PRIMARY export.

        Args:
            export_payload: JSON-compatible response from the PRIMARY licensing export endpoint.

        Raises:
            ValueError: If the export omits its definitions or auxiliary asset records.

        Side Effects:
            Atomically replaces the replica's local JSON cache. This does not make the
            store writable through its public CRUD methods.
        """
        raw_definitions = export_payload.get("licenses")
        raw_assets = export_payload.get("assets")
        raw_metadata = export_payload.get("metadata")
        if not isinstance(raw_definitions, list) or not isinstance(raw_assets, list):
            raise ValueError("Licensing export must contain license and asset arrays")
        metadata = LicensingDatasetMetadata.model_validate(raw_metadata or {})
        definitions: dict[str, LicenseDefinition] = {}
        for raw_definition in raw_definitions:
            definition = LicenseDefinition.model_validate(raw_definition)
            if definition.license_id in definitions:
                raise ValueError(f"Duplicate license definition in export: {definition.license_id}")
            definitions[definition.license_id] = definition
        assets: dict[str, LicensedAsset] = {}
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, dict) or raw_asset.get("asset_kind") == "model":
                continue
            licensing_payload = raw_asset.get("licensing")
            if isinstance(licensing_payload, dict):
                licensing_payload = {key: value for key, value in licensing_payload.items() if key != "files"}
            asset_payload = {
                key: raw_asset[key]
                for key in (
                    "asset_kind",
                    "asset_identifier",
                    "display_name",
                    "source_url",
                    "version",
                    "locations",
                    "related_assets",
                    "notes",
                    "metadata",
                )
                if key in raw_asset
            }
            asset_payload["licensing"] = licensing_payload
            asset = LicensedAsset.model_validate(asset_payload)
            if asset.asset_key in assets:
                raise ValueError(f"Duplicate licensed asset in export: {asset.asset_key}")
            for license_id in asset.licensing.license_ids:
                if license_id not in definitions:
                    raise ValueError(f"Unknown license definition: {license_id}")
            assets[asset.asset_key] = asset

        with self._lock:
            self._definitions = _LicenseDefinitionsEnvelope(
                schema_version=metadata.schema_version,
                revision=metadata.revision,
                updated_at=metadata.updated_at,
                records=definitions,
            )
            self._assets = _LicensedAssetsEnvelope(
                schema_version=metadata.schema_version,
                revision=metadata.revision,
                updated_at=metadata.updated_at,
                records=assets,
            )
            self._root_path.mkdir(parents=True, exist_ok=True)
            self._persist_definitions()
            self._persist_assets()

    def _load_definitions(self) -> _LicenseDefinitionsEnvelope:
        source_path = self._select_source(_LICENSE_DEFINITIONS_FILENAME)
        if source_path is None:
            return _LicenseDefinitionsEnvelope()
        return _LicenseDefinitionsEnvelope.model_validate_json(source_path.read_text(encoding="utf-8"))

    def _load_assets(self) -> _LicensedAssetsEnvelope:
        source_path = self._select_source(_LICENSED_ASSETS_FILENAME)
        if source_path is None:
            return _LicensedAssetsEnvelope()
        return _LicensedAssetsEnvelope.model_validate_json(source_path.read_text(encoding="utf-8"))

    def _select_source(self, filename: str) -> Path | None:
        mutable_path = self._root_path / filename
        if mutable_path.is_file():
            return mutable_path
        if self._bootstrap_path is None:
            return None
        bootstrap_file = self._bootstrap_path / filename
        return bootstrap_file if bootstrap_file.is_file() else None

    def _persist_missing_bootstrap_files(self) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)
        definitions_path = self._root_path / _LICENSE_DEFINITIONS_FILENAME
        assets_path = self._root_path / _LICENSED_ASSETS_FILENAME
        if not definitions_path.exists():
            self._persist_definitions()
        if not assets_path.exists():
            self._persist_assets()

    def _advance_definitions(self, timestamp: int) -> None:
        self._definitions = self._definitions.model_copy(
            update={"revision": self._definitions.revision + 1, "updated_at": timestamp},
        )
        self._persist_definitions()

    def _advance_assets(self, timestamp: int) -> None:
        self._assets = self._assets.model_copy(
            update={"revision": self._assets.revision + 1, "updated_at": timestamp},
        )
        self._persist_assets()

    def _persist_definitions(self) -> None:
        atomic_write_json(
            self._root_path / _LICENSE_DEFINITIONS_FILENAME,
            self._definitions.model_dump(mode="json"),
        )

    def _persist_assets(self) -> None:
        atomic_write_json(
            self._root_path / _LICENSED_ASSETS_FILENAME,
            self._assets.model_dump(mode="json"),
        )

    def _audit(
        self,
        *,
        editor_id: str,
        operation: str,
        record_kind: str,
        record_key: str,
        before: BaseModel | None = None,
        after: BaseModel | None = None,
    ) -> None:
        event = _LicensingAuditEvent(
            timestamp=_now_timestamp(),
            editor_id=editor_id,
            operation=operation,
            record_kind=record_kind,
            record_key=record_key,
            before=before.model_dump(mode="json") if before is not None else None,
            after=after.model_dump(mode="json") if after is not None else None,
        )
        self._root_path.mkdir(parents=True, exist_ok=True)
        with (self._root_path / _LICENSING_AUDIT_FILENAME).open("a", encoding="utf-8") as audit_file:
            audit_file.write(event.model_dump_json() + "\n")

    def _require_writable(self) -> None:
        if not self._writable:
            raise PermissionError("Licensing store is read-only")


def _now_timestamp() -> int:
    """Return the current UTC Unix timestamp."""
    return int(datetime.now(tz=UTC).timestamp())
