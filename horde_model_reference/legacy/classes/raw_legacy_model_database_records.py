"""The classes which can represent a legacy model reference file."""

from __future__ import annotations

from collections.abc import Mapping
from enum import auto

from pydantic import BaseModel, ConfigDict, RootModel
from strenum import StrEnum


class RawLegacy_DownloadRecord(BaseModel):
    """An entry in the `config` field of a `RawLegacy_StableDiffusion_ModelRecord`."""

    file_name: str
    file_path: str
    file_url: str
    file_type: str | None = None


class RawLegacy_FileRecord(BaseModel):
    """An entry in the `config` field of a `RawLegacy_StableDiffusion_ModelRecord`."""

    path: str
    md5sum: str | None = None
    sha256sum: str | None = None
    file_type: str | None = None


class FEATURE_SUPPORTED(StrEnum):
    """A feature supported by a model."""

    hires_fix = auto()
    loras = auto()
    inpainting = auto()
    controlnet = auto()


class RawLegacy_StableDiffusion_ModelRecord(BaseModel):
    """A model entry in the legacy model reference."""

    # This is a better representation of the legacy model reference than the one in `staging_model_database_records.py`
    # which is a hybrid representation of the legacy model reference and the new model reference format.

    model_config = ConfigDict(extra="forbid")

    name: str
    baseline: str
    type: str
    inpainting: bool
    description: str | None = None
    tags: list[str] | None = None
    showcases: list[str] | None = None
    min_bridge_version: int | None = None
    version: str
    style: str | None = None
    trigger: list[str] | None = None
    homepage: str | None = None
    nsfw: bool
    download_all: bool
    config: Mapping[str, list[RawLegacy_FileRecord | RawLegacy_DownloadRecord]]
    available: bool | None = None
    features_not_supported: list[FEATURE_SUPPORTED] | None = None
    size_on_disk_bytes: int | None = None


class RawLegacy_StableDiffusion_ModelReference(RootModel[Mapping[str, RawLegacy_StableDiffusion_ModelRecord]]):
    pass
