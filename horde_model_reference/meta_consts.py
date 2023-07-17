from enum import auto

from strenum import StrEnum


class MODEL_STYLES(StrEnum):
    """An enum of all the model styles."""

    generalist = auto()
    anime = auto()
    furry = auto()
    artistic = auto()
    other = auto()
    realistic = auto()


KNOWN_TAGS = [
    "anime",
    "manga",
    "cyberpunk",
    "tv show",
    "booru",
    "retro",
    "character",
    "hentai",
    "scenes",
    "low poly",
    "cg",
    "sketch",
    "high resolution",
    "landscapes",
    "comic",
    "cartoon",
    "painting",
    "game",
]


class MODEL_REFERENCE_CATEGORIES(StrEnum):
    """The categories of model reference entries."""

    blip = auto()
    clip = auto()
    codeformer = auto()
    controlnet = auto()
    esrgan = auto()
    gfpgan = auto()
    safety_checker = auto()
    stable_diffusion = auto()


class MODEL_PURPOSE(StrEnum):
    image_generation = auto()
    """The model is for image generation."""

    controlnet = auto()
    """The model is a controlnet."""

    clip = auto()
    """The model is a CLIP model."""

    blip = auto()
    """The model is a BLIP model."""

    post_processor = auto()
    """The model is a post processor (after image generation) of some variety."""


class STABLE_DIFFUSION_BASELINE_CATEGORIES(StrEnum):
    """An enum of all the stable diffusion baselines."""

    stable_diffusion_1 = auto()
    stable_diffusion_2_768 = auto()
    stable_diffusion_2_512 = auto()


MODEL_PURPOSE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, MODEL_PURPOSE] = {
    MODEL_REFERENCE_CATEGORIES.clip: MODEL_PURPOSE.clip,
    MODEL_REFERENCE_CATEGORIES.blip: MODEL_PURPOSE.blip,
    MODEL_REFERENCE_CATEGORIES.codeformer: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.controlnet: MODEL_PURPOSE.controlnet,
    MODEL_REFERENCE_CATEGORIES.esrgan: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.gfpgan: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.safety_checker: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORIES.stable_diffusion: MODEL_PURPOSE.image_generation,
}
