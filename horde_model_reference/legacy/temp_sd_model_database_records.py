from typing import Mapping

from pydantic import BaseModel


class Temp_DownloadRecord(BaseModel):
    """A record of a file to download for a model. Typically a ckpt file."""

    file_name: str
    file_path: str
    file_url: str


class Temp_FileRecord(BaseModel):
    path: str
    md5sum: str | None = None
    sha256sum: str | None = None


class Temp_StableDiffusion_ModelRecord(BaseModel):
    class Config:
        extra = "forbid"

    name: str
    baseline: str
    type: str  # noqa: A003
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
    config: Mapping[str, list[Temp_FileRecord | Temp_DownloadRecord]]
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
