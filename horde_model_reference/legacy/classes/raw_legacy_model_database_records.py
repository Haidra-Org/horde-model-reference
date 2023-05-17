from typing import Mapping

from pydantic import BaseModel


class RawLegacy_DownloadRecord(BaseModel):
    """An entry in the `config` field of a `RawLegacy_StableDiffusion_ModelRecord`."""

    file_name: str
    file_path: str
    file_url: str


class RawLegacy_FileRecord(BaseModel):
    """An entry in the `config` field of a `RawLegacy_StableDiffusion_ModelRecord`."""

    path: str
    md5sum: str | None = None
    sha256sum: str | None = None


class RawLegacy_StableDiffusion_ModelRecord(BaseModel):
    """A model entry in the legacy model reference. Note that `.dict()` exports to the new model reference format."""

    # This is a better representation of the legacy model reference than the one in `staging_model_database_records.py`
    # which is a hybrid representation of the legacy model reference and the new model reference format.

    class Config:
        extra = "forbid"

    name: str
    baseline: str
    type: str  # noqa: A003
    inpainting: bool | None
    description: str | None
    tags: list[str] | None
    showcases: list[str] | None
    min_bridge_version: int | None
    version: str
    style: str | None
    trigger: list[str] | None
    homepage: str | None
    nsfw: bool
    download_all: bool
    config: Mapping[str, list[RawLegacy_FileRecord | RawLegacy_DownloadRecord]]
    available: bool

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
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=True,
            exclude_defaults=exclude_defaults,
            exclude_none=True,
        )
