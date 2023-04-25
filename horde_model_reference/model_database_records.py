from pydantic import BaseModel


class DownloadRecord(BaseModel):
    class Config:
        extra = "forbid"

    file_name: str
    file_url: str
    sha256sum: str | None
    known_slow_download: bool | None


class ModelDatabaseEntry(BaseModel):
    """A model entry in the model reference."""

    class Config:
        extra = "forbid"

    name: str
    baseline: str
    description: str
    tags: list[str] | None
    showcases: list[str] | None
    min_bridge_version: int | None
    version: str
    style: str
    trigger: list[str] | None
    homepage: str | None
    nsfw: bool
    config: dict[str, list[DownloadRecord]]


class StableDiffusionModelReference(BaseModel):
    """The combined metadata and model list."""

    class Config:
        extra = "forbid"

    baseline_types: list[str]
    styles: list[str]
    tags: list[str]
    model_hosts: list[str]
    models: dict[str, ModelDatabaseEntry]
