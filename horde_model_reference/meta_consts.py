from enum import Enum


class MODEL_STYLES(str, Enum):
    """An enum of all the model styles."""

    generalist = "generalist"
    anime = "anime"
    furry = "furry"
    artistic = "artistic"
    other = "other"
    realistic = "realistic"


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


class MODEL_REFERENCE_CATEGORIES(str, Enum):
    """The categories of model reference entries."""

    BLIP = "blip"
    CLIP = "clip"
    ESRGAN = "esrgan"
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"
    CONTROLNET = "controlnet"
    SAFETY_CHECKER = "safety_checker"
    STABLE_DIFFUSION = "stable_diffusion"


class MODEL_PURPOSE(str, Enum):
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


class STABLE_DIFFUSION_BASELINE_CATEGORIES(str, Enum):
    """An enum of all the stable diffusion baselines."""

    stable_diffusion_1 = "stable_diffusion_1"
    stable_diffusion_2_768 = "stable_diffusion_2_768"
    stable_diffusion_2_512 = "stable_diffusion_2_512"


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
