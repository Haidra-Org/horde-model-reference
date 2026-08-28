"""Typed records for the served image-generation baseline catalog.

The catalog names each baseline family the reference publishes, the architecture and ecosystem
facts consumers use to decide what a family can do, and the AI-Horde scheduling and pricing policy
attached to it. It is a first-class resource rather than a model category: every enum-traversing
consumer treats a category as a set of model records.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IMAGE_BASELINE_SCHEMA_VERSION = 1
"""Current serialized schema version for the image baseline catalog."""


class BaselineCapabilities(BaseModel):
    """Architecture and ecosystem facts about what exists for one baseline family.

    Defaults are conservative: omitting a capability never claims weights or a mechanism that have
    not been published. Whether a worker's engine can run a published capability is a separate axis
    that lives with the worker, not here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    controlnet: bool = False
    """Whether any conditioned ControlNet weights exist for the family."""

    controlnet_types_unavailable: tuple[str, ...] = ()
    """ControlNet styles with no published weights for the family."""

    transparent: bool = False
    """Whether layer-diffusion weights exist for the family."""

    qr_code: bool = False
    """Whether QR-code ControlNet weights exist for the family."""

    remix: bool = False
    """Whether the family's architecture provides an image-remix mechanism."""

    flow_matching: bool = False
    """Whether a flow-matching shift parameter is meaningful for the family."""


class HordeBaselinePolicy(BaseModel):
    """AI-Horde scheduling and pricing policy for one baseline.

    Defaults are the par values an unremarkable baseline receives, so a policy needs to state only
    what departs from par.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kudos: float = 1
    """Base kudos multiplier applied to generations on this baseline."""

    kudos_qr_code: float | None = None
    """Kudos multiplier for QR-code generations, or ``None`` to use ``kudos``."""

    kudos_hires: float | None = None
    """Kudos multiplier for hires-fix generations, or ``None`` to use ``kudos``."""

    batching: int = Field(default=1, ge=1)
    """Multiplier on the batch-size cost calculation; larger values batch fewer images per job."""

    ttl: int = Field(default=1, ge=1)
    """Multiplier applied to a request's expiry allowance."""

    resolution_floor: int = Field(default=0, ge=0)
    """Single-side resolution every request may reach under queue pressure, or ``0`` for no floor."""


class BaselineRecordMetadata(BaseModel):
    """Lifecycle metadata for a published baseline record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(default=1, ge=1)
    created_at: int | None = None
    created_by: str | None = None
    updated_at: int | None = None
    updated_by: str | None = None


class ImageBaselineRecord(BaseModel):
    """Represents one published image-generation baseline family."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    display_name: str | None = None
    native_resolution: int | None = Field(default=None, ge=1)
    alternative_names: tuple[str, ...] = ()
    capabilities: BaselineCapabilities = BaselineCapabilities()
    horde_policy: HordeBaselinePolicy = HordeBaselinePolicy()
    metadata: BaselineRecordMetadata = Field(default_factory=BaselineRecordMetadata)


class ImageBaselineCatalogMetadata(BaseModel):
    """Revision metadata for the complete baseline catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = IMAGE_BASELINE_SCHEMA_VERSION
    revision: int = Field(default=1, ge=1)
    updated_at: int | None = None


class ImageBaselineCatalog(BaseModel):
    """Atomic machine-readable catalog of published image baselines."""

    model_config = ConfigDict(extra="forbid")

    metadata: ImageBaselineCatalogMetadata = Field(default_factory=ImageBaselineCatalogMetadata)
    baselines: dict[str, ImageBaselineRecord] = Field(default_factory=dict)


class ImageBaselineChange(BaseModel):
    """One baseline operation and the value against which it was reviewed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["upsert", "delete"]
    name: str = Field(min_length=1)
    record: ImageBaselineRecord | None = None
    expected_before: ImageBaselineRecord | None = None


class ImageBaselineChangeSet(BaseModel):
    """Coherent baseline edits submitted as one pending-queue resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=160)
    changes: list[ImageBaselineChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_changes(self) -> ImageBaselineChangeSet:
        """Require at least one baseline edit."""
        if not self.changes:
            raise ValueError("A baseline change set must contain at least one change.")
        return self
