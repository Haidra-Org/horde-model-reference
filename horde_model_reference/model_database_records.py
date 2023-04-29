import enum
from enum import Enum
from typing import Mapping

from pydantic import BaseModel

from horde_model_reference.consts import MODEL_REFERENCE_CATEGORIES


class MODEL_PURPOSE(str, enum.Enum):
    image_generation = "image_generation"
    """The model is for image generation."""

    controlnet = "controlnet"
    """The model is a controlnet."""

    clip = "clip"
    """The model is a CLIP model."""

    blip = "blip"
    """The model is a BLIP model."""

    post_processor = "post_processor"
    """The model is a post processor of some variety."""


MODEL_PURPOSE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, MODEL_PURPOSE] = {
    MODEL_REFERENCE_CATEGORIES.CLIP: MODEL_PURPOSE.clip,
    MODEL_REFERENCE_CATEGORIES.BLIP: MODEL_PURPOSE.blip,
    MODEL_REFERENCE_CATEGORIES.CODEFORMER: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: MODEL_PURPOSE.controlnet,
    MODEL_REFERENCE_CATEGORIES.ESRGAN: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.GFPGAN: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.SAFETY_CHECKER: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: MODEL_PURPOSE.image_generation,
}


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


class DownloadRecord(BaseModel):  # TODO Rename? (record to subrecord?)
    """A record of a file to download for a model. Typically a ckpt file."""

    file_name: str
    """The horde specific filename. This is not necessarily the same as the file's name on the model host."""
    file_url: str
    """The fully qualified URL to download the file from."""
    sha256sum: str
    """The sha256sum of the file."""
    known_slow_download: bool | None
    """Whether the download is known to be slow or not."""


class Generic_ModelRecord(BaseModel):
    # TODO forbid extra?
    name: str
    """The name of the model."""
    description: str | None
    """A short description of the model."""
    version: str | None
    """The version of the  model (not the version of SD it is based on, see `baseline` for that info)."""
    style: str | None
    """The style of the model."""
    config: dict[str, list[DownloadRecord]]
    """A dictionary of any configuration files and information on where to download the model file(s)."""

    model_purpose: MODEL_PURPOSE
    """The purpose of the model."""


class StableDiffusion_ModelRecord(Generic_ModelRecord):
    """A model entry in the model reference."""

    class Config:
        extra = "forbid"

    baseline: STABLEDIFFUSION_BASELINE
    """The model on which this model is based."""
    tags: list[str] | None
    """Any tags associated with the model which may be useful for searching."""
    showcases: list[str] | None
    """Links to any showcases of the model which illustrate its style."""
    min_bridge_version: int | None
    """The minimum version of AI-Horde-Worker required to use this model."""
    trigger: list[str] | None
    """A list of trigger words or phrases which can be used to activate the model."""
    homepage: str | None
    """A link to the model's homepage."""
    nsfw: bool
    """Whether the model is NSFW or not."""


class CLIP_ModelRecord(Generic_ModelRecord):

    pretrained_name: str | None


class Generic_ModelReference(BaseModel):
    models: Mapping[str, Generic_ModelRecord]
    """A dictionary of all the models."""


class StableDiffusion_ModelReference(Generic_ModelReference):
    """The combined metadata and model list."""

    class Config:
        extra = "forbid"

    baseline_categories: dict[STABLEDIFFUSION_BASELINE, int]
    """A dictionary of all the baseline types and how many models use them."""
    styles: dict[MODEL_STYLES, int]
    """A dictionary of all the styles and how many models use them."""
    tags: dict[str, int]
    """A dictionary of all the tags and how many models use them."""
    model_hosts: dict[str, int]
    """A dictionary of all the model hosts and how many models use them."""
    models: Mapping[str, StableDiffusion_ModelRecord]
    """A dictionary of all the models."""


class CLIP_ModelReference(Generic_ModelReference):
    models: Mapping[str, CLIP_ModelRecord]


MODEL_REFERENCE_RECORD_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, type[Generic_ModelRecord]] = {
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: StableDiffusion_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CLIP: CLIP_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.BLIP: Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.ESRGAN: Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.GFPGAN: Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.SAFETY_CHECKER: Generic_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CODEFORMER: Generic_ModelRecord,
}
"""A lookup for the model record type based on the model category. See also `MODEL_REFERENCE_TYPE_LOOKUP`."""

MODEL_REFERENCE_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, type[Generic_ModelReference]] = {
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: StableDiffusion_ModelReference,
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: Generic_ModelReference,
    MODEL_REFERENCE_CATEGORIES.CLIP: CLIP_ModelReference,
    MODEL_REFERENCE_CATEGORIES.BLIP: Generic_ModelReference,
    MODEL_REFERENCE_CATEGORIES.ESRGAN: Generic_ModelReference,
    MODEL_REFERENCE_CATEGORIES.GFPGAN: Generic_ModelReference,
    MODEL_REFERENCE_CATEGORIES.SAFETY_CHECKER: Generic_ModelReference,
    MODEL_REFERENCE_CATEGORIES.CODEFORMER: Generic_ModelReference,
}
"""A lookup for the model reference type based on the model category. See also `MODEL_REFERENCE_RECORD_TYPE_LOOKUP`."""
