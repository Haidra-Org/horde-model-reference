"""Helper classes to convert the legacy model reference to the new model reference format."""

# If you're here in search of an explanation, please know that this isn't really
# a set of classes exactly representative of the legacy model reference. It's
# more of a hybrid representation of the legacy model reference and the new model.

# These classes will only persist until the legacy model reference is fully deprecated.

from typing import Mapping

from pydantic import BaseModel

from horde_model_reference.model_reference_records import MODEL_PURPOSE
from horde_model_reference.path_consts import MODEL_REFERENCE_CATEGORIES


class Legacy_Config_FileRecord(BaseModel):
    # class Config:
    #     extra = "forbid"

    path: str
    # md5sum: str | None
    sha256sum: str | None


class Legacy_Config_DownloadRecord(BaseModel):
    class Config:
        extra = "forbid"

    file_name: str
    file_path: str = ""
    file_url: str
    sha256sum: str | None
    known_slow_download: bool | None


class Legacy_Generic_ModelRecord(BaseModel):
    """This is hybrid representation of the legacy model reference and the new model reference format."""

    class Config:
        extra = "forbid"

    name: str
    type: str  # noqa: A003
    description: str | None
    version: str | None
    style: str | None
    nsfw: bool | None
    download_all: bool | None
    config: dict[str, list[Legacy_Config_FileRecord | Legacy_Config_DownloadRecord]]
    available: bool

    model_purpose: MODEL_PURPOSE | None


class Legacy_CLIP_ModelRecord(Legacy_Generic_ModelRecord):
    """A model entry in the legacy model reference. Note that `.dict()` exports to the new model reference format."""

    pretrained_name: str | None


class Legacy_StableDiffusion_ModelRecord(Legacy_Generic_ModelRecord):
    """A model entry in the legacy model reference. Note that `.dict()` exports to the new model reference format."""

    baseline: str
    tags: list[str] | None
    showcases: list[str] | None
    min_bridge_version: int | None
    trigger: list[str] | None
    homepage: str | None

    def dict(  # noqa: A003
        self,
        *,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ):
        return super().dict(
            include=include,
            exclude={"available", "type", "download_all"},
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )


class Legacy_Generic_ModelReference(BaseModel):
    """A helper class to convert the legacy model reference to the new model reference format."""

    class Config:
        extra = "forbid"

    models: Mapping[str, Legacy_Generic_ModelRecord]


class Legacy_StableDiffusion_ModelReference(Legacy_Generic_ModelReference):
    """A helper class to convert the legacy model reference to the new model reference format."""

    baseline_categories: dict[str, int]
    styles: dict[str, int]
    tags: dict[str, int]
    model_hosts: dict[str, int]
    models: Mapping[str, Legacy_StableDiffusion_ModelRecord]


MODEL_REFERENCE_LEGACY_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, type[Legacy_Generic_ModelRecord]] = {
    MODEL_REFERENCE_CATEGORIES.BLIP: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CLIP: Legacy_CLIP_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CODEFORMER: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.ESRGAN: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.GFPGAN: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.SAFETY_CHECKER: Legacy_Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: Legacy_StableDiffusion_ModelRecord,
}
