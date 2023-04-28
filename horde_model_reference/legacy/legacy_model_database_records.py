from pydantic import BaseModel


class Legacy_Config_FileRecord(BaseModel):
    # class Config:
    # extra = "forbid"

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
    class Config:
        extra = "forbid"

    name: str
    type: str  # noqa: A003
    description: str
    version: str
    style: str
    nsfw: bool | None
    download_all: bool | None
    config: dict[str, list[Legacy_Config_FileRecord | Legacy_Config_DownloadRecord]]
    available: bool


class Legacy_StableDiffusion_ModelRecord(Legacy_Generic_ModelRecord):
    """A model entry in the legacy model reference. Note that `.dict()` exports to the new model reference format."""

    baseline: str
    tags: list[str] | None
    showcases: list[str] | None
    min_bridge_version: int | None
    trigger: list[str] | None
    homepage: str | None

    def dict(
        self,
        *,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False
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


class Legacy_StableDiffusion_ModelReference(BaseModel):
    """A helper class to convert the legacy model reference to the new model reference format."""

    class Config:
        extra = "forbid"

    baseline_types: dict[str, int]
    styles: dict[str, int]
    tags: dict[str, int]
    model_hosts: dict[str, int]
    models: dict[str, Legacy_StableDiffusion_ModelRecord]
