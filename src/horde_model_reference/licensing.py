"""Define validated licensing records shared by storage, library, and HTTP consumers.

The public models in this module deliberately separate a reusable license definition
from the reviewed conclusion for one model, file, or software asset.  A license
identifier therefore behaves like a foreign key while an assignment records the
scope-specific evidence and permission conclusion.
"""

from __future__ import annotations

from datetime import date

from license_expression import ExpressionError, Licensing
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from strenum import StrEnum

__all__ = [
    "LICENSE_DATA_SCHEMA_VERSION",
    "LicenseAssignment",
    "LicenseDefinition",
    "LicenseEvidence",
    "LicenseObligation",
    "LicensedAsset",
    "LicensedAssetKind",
    "LicensingRecordMetadata",
    "ModelLicensing",
    "PermissionStatus",
    "aggregate_permission_statuses",
    "unknown_model_licensing",
]

LICENSE_DATA_SCHEMA_VERSION = 1
"""Current schema version for the normalized licensing datasets."""

_NOASSERTION = "NOASSERTION"
_LICENSING = Licensing()


class PermissionStatus(StrEnum):
    """Represents the reviewed permission status for a particular activity."""

    ALLOWED = "allowed"
    ALLOWED_WITH_CONDITIONS = "allowed_with_conditions"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class LicenseObligation(StrEnum):
    """Represents a common obligation that consumers may need to surface."""

    ATTRIBUTION = "attribution"
    INCLUDE_LICENSE = "include_license"
    SHARE_ALIKE = "share_alike"
    DISCLOSE_SOURCE = "disclose_source"
    NETWORK_SOURCE_DISCLOSURE = "network_source_disclosure"
    RESPONSIBLE_USE_RESTRICTIONS = "responsible_use_restrictions"
    OTHER = "other"


class LicensedAssetKind(StrEnum):
    """Identifies the kind of non-model asset held in the auxiliary store."""

    SOFTWARE_COMPONENT = "software_component"
    CUSTOM_NODE = "custom_node"
    OTHER = "other"


class LicensingRecordMetadata(BaseModel):
    """Represents lifecycle metadata for an independently mutable licensing record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(default=1, ge=1)
    created_at: int | None = None
    created_by: str | None = None
    updated_at: int | None = None
    updated_by: str | None = None


class LicenseEvidence(BaseModel):
    """Represents one source supporting a scope-specific licensing conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    description: str | None = None
    checked_at: date | None = None


class LicenseDefinition(BaseModel):
    """Represents one normalized, reusable license definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    license_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    spdx_identifier: str | None = None
    canonical_url: str = Field(min_length=1)
    commercial_use: PermissionStatus
    redistribution: PermissionStatus
    obligations: tuple[LicenseObligation, ...] = ()
    restrictions: tuple[str, ...] = ()
    notes: str | None = None
    deprecated: bool = False
    metadata: LicensingRecordMetadata = Field(default_factory=LicensingRecordMetadata)

    @field_validator("license_id")
    @classmethod
    def validate_license_id(cls, license_id: str) -> str:
        """Validate that a definition uses an SPDX-compatible atomic identifier."""
        try:
            parsed_expression = _LICENSING.parse(license_id, validate=False)
        except ExpressionError as exception:
            raise ValueError(f"Invalid license identifier: {license_id}") from exception
        if parsed_expression is None:
            raise ValueError(f"Invalid license identifier: {license_id}")
        parsed_symbols = {str(symbol) for symbol in parsed_expression.symbols}
        if parsed_symbols != {license_id}:
            raise ValueError("license_id must be one atomic SPDX identifier or LicenseRef")
        if license_id == _NOASSERTION:
            raise ValueError("NOASSERTION is a conclusion, not a reusable license definition")
        return license_id


class LicenseAssignment(BaseModel):
    """Represents a reviewed license conclusion for one declared scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    license_expression: str = Field(min_length=1)
    license_ids: tuple[str, ...] = ()
    commercial_use: PermissionStatus
    redistribution: PermissionStatus
    obligations: tuple[LicenseObligation, ...] = ()
    attribution: str | None = None
    evidence: tuple[LicenseEvidence, ...] = ()
    reviewed_by: str | None = None
    reviewed_at: date | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_expression_and_unknown_state(self) -> LicenseAssignment:
        """Validate SPDX syntax and the invariants for an unknown conclusion."""
        try:
            parsed_expression = _LICENSING.parse(self.license_expression, validate=False)
        except ExpressionError as exception:
            raise ValueError(f"Invalid license expression: {self.license_expression}") from exception
        if parsed_expression is None:
            raise ValueError(f"Invalid license expression: {self.license_expression}")

        parsed_identifiers = {str(symbol) for symbol in parsed_expression.symbols}
        declared_identifiers = set(self.license_ids)
        if self.license_expression == _NOASSERTION:
            if declared_identifiers:
                raise ValueError("NOASSERTION must not reference license definitions")
            if self.commercial_use is not PermissionStatus.UNKNOWN:
                raise ValueError("NOASSERTION requires commercial_use='unknown'")
            if self.redistribution is not PermissionStatus.UNKNOWN:
                raise ValueError("NOASSERTION requires redistribution='unknown'")
            return self

        if parsed_identifiers != declared_identifiers:
            raise ValueError("license_ids must exactly match the identifiers in license_expression")
        if not self.license_ids:
            raise ValueError("A known license expression must reference at least one definition")
        return self


class ModelLicensing(LicenseAssignment):
    """Represents a model-level conclusion with optional per-download-file overrides."""

    files: dict[str, LicenseAssignment] = Field(default_factory=dict)

    def assignment_for_file(self, file_name: str) -> LicenseAssignment:
        """Return the effective assignment for a declared model file.

        Args:
            file_name: Exact ``DownloadRecord.file_name`` to resolve.

        Returns:
            The file override when present, otherwise the model-level assignment.
        """
        return self.files.get(file_name, self)


class LicensedAsset(BaseModel):
    """Represents a licensed non-model asset managed directly by PRIMARY editors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_kind: LicensedAssetKind
    asset_identifier: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    source_url: str | None = None
    version: str | None = None
    locations: tuple[str, ...] = ()
    related_assets: tuple[str, ...] = ()
    licensing: LicenseAssignment
    notes: str | None = None
    metadata: LicensingRecordMetadata = Field(default_factory=LicensingRecordMetadata)

    @property
    def asset_key(self) -> str:
        """Return the globally stable key used by the auxiliary asset store."""
        return f"{self.asset_kind.value}:{self.asset_identifier}"


def aggregate_permission_statuses(statuses: tuple[PermissionStatus, ...]) -> PermissionStatus:
    """Return the conservative aggregate of a set of scoped permission conclusions.

    Args:
        statuses: Permission conclusions to aggregate.

    Returns:
        The most restrictive or uncertain effective status.
    """
    if PermissionStatus.PROHIBITED in statuses:
        return PermissionStatus.PROHIBITED
    if PermissionStatus.UNKNOWN in statuses or not statuses:
        return PermissionStatus.UNKNOWN
    if PermissionStatus.ALLOWED_WITH_CONDITIONS in statuses:
        return PermissionStatus.ALLOWED_WITH_CONDITIONS
    return PermissionStatus.ALLOWED


def unknown_model_licensing(*, note: str | None = None) -> ModelLicensing:
    """Create an explicit fail-closed licensing conclusion for an unaudited model.

    Args:
        note: Optional explanation of why no assertion is available.

    Returns:
        A model licensing record using the SPDX ``NOASSERTION`` sentinel.
    """
    return ModelLicensing(
        license_expression=_NOASSERTION,
        commercial_use=PermissionStatus.UNKNOWN,
        redistribution=PermissionStatus.UNKNOWN,
        notes=note or "No reviewed licensing conclusion is currently available.",
    )
