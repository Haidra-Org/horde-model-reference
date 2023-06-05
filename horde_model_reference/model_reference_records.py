"""The model database pydantic models and associate enums/lookups."""
from typing import Mapping

from pydantic import BaseModel

from horde_model_reference import (
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORIES,
    MODEL_STYLES,
    STABLE_DIFFUSION_BASELINE_CATEGORIES,
)


class DownloadRecord(BaseModel):  # TODO Rename? (record to subrecord?)
    """A record of a file to download for a model. Typically a ckpt file."""

    file_name: str
    """The horde specific filename. This is not necessarily the same as the file's name on the model host."""
    file_url: str
    """The fully qualified URL to download the file from."""
    sha256sum: str
    """The sha256sum of the file."""
    known_slow_download: bool | None = None
    """Whether the download is known to be slow or not."""


class Generic_ModelRecord(BaseModel):
    # TODO forbid extra?
    name: str
    """The name of the model."""
    description: str | None
    """A short description of the model."""
    version: str | None
    """The version of the  model (not the version of SD it is based on, see `baseline` for that info)."""
    style: MODEL_STYLES | str | None  # TODO remove str
    """The style of the model."""
    config: dict[str, list[DownloadRecord]]
    """A dictionary of any configuration files and information on where to download the model file(s)."""

    model_purpose: MODEL_PURPOSE
    """The purpose of the model."""


class StableDiffusion_ModelRecord(Generic_ModelRecord):
    """A model entry in the model reference."""

    class Config:
        extra = "forbid"

    inpainting: bool | None
    """If this is an inpainting model or not."""
    baseline: STABLE_DIFFUSION_BASELINE_CATEGORIES
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
    # TODO docstring


class Generic_ModelReference(BaseModel):
    models: Mapping[str, Generic_ModelRecord]
    """A dictionary of all the models."""


class StableDiffusion_ModelReference(Generic_ModelReference):
    """The combined metadata and model list."""

    class Config:
        extra = "forbid"

    baseline: dict[STABLE_DIFFUSION_BASELINE_CATEGORIES, int]
    """A dictionary of all the baseline types and how many models use them."""
    styles: dict[MODEL_STYLES, int]
    """A dictionary of all the styles and how many models use them."""
    tags: dict[str, int]
    """A dictionary of all the tags and how many models use them."""
    model_hosts: dict[str, int]
    """A dictionary of all the model hosts and how many models use them."""
    models: Mapping[str, StableDiffusion_ModelRecord]
    """A dictionary of all the models."""


def create_stablediffusion_modelreference(models: Mapping[str, StableDiffusion_ModelRecord]):
    """Create a StableDiffusion_ModelReference from a mapping of {str: StableDiffusion_ModelRecords}."""
    baseline_categories: dict[STABLE_DIFFUSION_BASELINE_CATEGORIES, int] = {}
    styles: dict[MODEL_STYLES, int] = {}
    tags: dict[str, int] = {}
    model_hosts: dict[str, int] = {}
    for model in models.values():
        baseline_categories[model.baseline] = baseline_categories.get(model.baseline, 0) + 1
        if model.style is not None and model.style in MODEL_STYLES:
            styles[MODEL_STYLES(model.style)] = styles.get(MODEL_STYLES(model.style), 0) + 1
        if model.tags is not None:
            for tag in model.tags:
                tags[tag] = tags.get(tag, 0) + 1
        for host in model.config:
            model_hosts[host] = model_hosts.get(host, 0) + 1
    return StableDiffusion_ModelReference(
        baseline=baseline_categories,
        styles=styles,
        tags=tags,
        model_hosts=model_hosts,
        models=models,
    )


class CLIP_ModelReference(Generic_ModelReference):
    models: Mapping[str, CLIP_ModelRecord]
    """A dictionary of all the models."""


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
