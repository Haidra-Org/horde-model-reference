from enum import Enum
from pydantic import BaseModel


class STABLEDIFFUSION_BASELINE(str, Enum):
    """An enum of all the stable diffusion baselines."""

    stable_diffusion_1 = "stable_diffusion_1"
    stable_diffusion_2_768 = "stable_diffusion_2_768"
    stable_diffusion_2_512 = "stable_diffusion_2_512"


class MODEL_STYLES(str, Enum):
    """An enum of all the model styles."""

    generalist = "generalist"
    anime = "anime"
    furry = "furry"
    artistic = "artistic"
    other = "other"
    realistic = "realistic"


class DownloadRecord(BaseModel):
    """A record of a file to download for a model. Typically a ckpt file."""

    class Config:
        extra = "forbid"

    file_name: str
    """The horde specific filename. This is not necessarily the same as the file's name on the model host."""
    file_url: str
    """The fully qualified URL to download the file from."""
    sha256sum: str
    """The sha256sum of the file."""
    known_slow_download: bool | None
    """Whether the download is known to be slow or not."""


class ModelDatabaseEntry(BaseModel):
    """A model entry in the model reference."""

    class Config:
        extra = "forbid"

    name: str
    """The name of the model."""
    baseline: STABLEDIFFUSION_BASELINE
    """The model on which this model is based."""
    description: str
    """A short description of the model."""
    tags: list[str] | None
    """Any tags associated with the model which may be useful for searching."""
    showcases: list[str] | None
    """Links to any showcases of the model which illustrate its style."""
    min_bridge_version: int | None
    """The minimum version of AI-Horde-Worker required to use this model."""
    version: str
    """The version of the model (not the version of SD it is based on, see `baseline` for that info)."""
    style: str
    """The style of the model."""
    trigger: list[str] | None
    """A list of trigger words or phrases which can be used to activate the model."""
    homepage: str | None
    """A link to the model's homepage."""
    nsfw: bool
    """Whether the model is NSFW or not."""
    config: dict[str, list[DownloadRecord]]
    """A dictionary of any configuration files and information on where to download the model file(s)."""


class StableDiffusionModelReference(BaseModel):
    """The combined metadata and model list."""

    class Config:
        extra = "forbid"

    baseline_types: dict[STABLEDIFFUSION_BASELINE, int]
    """A dictionary of all the baseline types and how many models use them."""
    styles: dict[MODEL_STYLES, int]
    """A dictionary of all the styles and how many models use them."""
    tags: dict[str, int]
    """A dictionary of all the tags and how many models use them."""
    model_hosts: dict[str, int]
    """A dictionary of all the model hosts and how many models use them."""
    models: dict[str, ModelDatabaseEntry]
    """A dictionary of all the models."""
