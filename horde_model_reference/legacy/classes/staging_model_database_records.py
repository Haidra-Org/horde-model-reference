"""Helper classes to convert the legacy model reference to the new model reference format."""

# If you're here in search of an explanation, please know that this isn't really
# a set of classes exactly representative of the legacy model reference. It's
# more of a hybrid representation of the legacy model reference and the new model.

# These classes will persist until the legacy model reference is fully deprecated.

from __future__ import annotations

from collections.abc import Mapping

from loguru import logger
from pydantic import BaseModel, ConfigDict, model_validator

from horde_model_reference.model_reference_records import MODEL_PURPOSE
from horde_model_reference.path_consts import MODEL_REFERENCE_CATEGORY


class StagingLegacy_Config_FileRecord(BaseModel):
    """An entry in the `config` field of a `StagingLegacy_Generic_ModelRecord`."""

    # class Config:
    #     extra = "forbid"

    path: str
    # md5sum: str | None
    sha256sum: str | None = None

    @model_validator(mode="after")
    def validate_model_file_has_sha256sum(self):
        if ".yaml" in self.path or ".json" in self.path:
            return self

        if self.sha256sum is None:
            raise ValueError("A model file must have a sha256sum.")

        return self


class StagingLegacy_Config_DownloadRecord(BaseModel):
    """An entry in the `config` field of a `StagingLegacy_Generic_ModelRecord`."""

    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_path: str = ""
    file_type: str | None = None
    file_url: str
    sha256sum: str | None = None
    known_slow_download: bool | None = False


class StagingLegacy_Generic_ModelRecord(BaseModel):
    """This is a helper class, a hybrid representation of the legacy model reference and the new format."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str | None = None
    version: str | None = None
    style: str | None = None
    nsfw: bool | None = None
    download_all: bool | None = None
    config: dict[str, list[StagingLegacy_Config_FileRecord | StagingLegacy_Config_DownloadRecord]]
    available: bool | None = None

    purpose: MODEL_PURPOSE | str | None = None
    features_not_supported: list[str] | None = None

    @model_validator(mode="after")
    def validator_known_purpose(self) -> StagingLegacy_Generic_ModelRecord:
        """Check if the purpose is known."""
        if self.purpose is not None and str(self.purpose) not in MODEL_PURPOSE.__members__:
            logger.warning(f"Unknown purpose {self.purpose} for model {self.name}")

        return self


class Legacy_CLIP_ModelRecord(StagingLegacy_Generic_ModelRecord):
    """A model entry in the legacy model reference."""

    pretrained_name: str | None = None


class Legacy_StableDiffusion_ModelRecord(StagingLegacy_Generic_ModelRecord):
    """A model entry in the legacy model reference."""

    inpainting: bool
    baseline: str
    tags: list[str] | None = None
    showcases: list[str] | None = None
    min_bridge_version: int | None = None
    trigger: list[str] | None = None
    homepage: str | None = None
    size_on_disk_bytes: int | None = None


class Legacy_Generic_ModelReference(BaseModel):
    """A helper class to convert the legacy model reference to the new model reference format."""

    model_config = ConfigDict(extra="forbid")

    models: Mapping[str, StagingLegacy_Generic_ModelRecord]


class Staging_StableDiffusion_ModelReference(Legacy_Generic_ModelReference):
    """A helper class to convert the legacy model reference to the new model reference format."""

    baseline: dict[str, int]
    styles: dict[str, int]
    tags: dict[str, int]
    download_hosts: dict[str, int]
    models: Mapping[str, Legacy_StableDiffusion_ModelRecord]


MODEL_REFERENCE_LEGACY_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, type[StagingLegacy_Generic_ModelRecord]] = {
    MODEL_REFERENCE_CATEGORY.blip: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.clip: Legacy_CLIP_ModelRecord,
    MODEL_REFERENCE_CATEGORY.codeformer: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.controlnet: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.esrgan: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.gfpgan: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.safety_checker: StagingLegacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORY.stable_diffusion: Legacy_StableDiffusion_ModelRecord,
}
