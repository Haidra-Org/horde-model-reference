from pydantic import BaseModel


class Legacy_ConfigFileRecord(BaseModel):
    # class Config:
    # extra = "forbid"

    path: str
    # md5sum: str | None
    sha256sum: str | None


class Legacy_DownloadRecord(BaseModel):
    class Config:
        extra = "forbid"

    file_name: str
    file_path: str = ""
    file_url: str
    sha256sum: str | None
    known_slow_download: bool | None


class Legacy_ModelDatabaseEntry(BaseModel):
    """A model entry in the legacy model reference. Note that `.dict()` exports to the new model reference format."""

    class Config:
        extra = "forbid"

    name: str
    baseline: str
    type: str  # noqa: A003
    description: str
    tags: list[str] | None
    showcases: list[str] | None
    min_bridge_version: int | None
    version: str
    style: str
    trigger: list[str] | None
    homepage: str | None
    nsfw: bool
    download_all: bool
    config: dict[str, list[Legacy_ConfigFileRecord | Legacy_DownloadRecord]]
    available: bool

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


class Legacy_StableDiffusionModelReference(BaseModel):
    """A helper class to convert the legacy model reference to the new model reference format."""

    class Config:
        extra = "forbid"

    baseline_types: list[str]
    styles: list[str]
    tags: list[str]
    model_hosts: list[str]
    models: dict[str, Legacy_ModelDatabaseEntry]
