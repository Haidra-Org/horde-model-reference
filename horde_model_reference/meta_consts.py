from enum import auto

from strenum import StrEnum


class MODEL_STYLE(StrEnum):
    """An enum of all the model styles."""

    generalist = auto()
    anime = auto()
    furry = auto()
    artistic = auto()
    other = auto()
    realistic = auto()


class CONTROLNET_STYLE(StrEnum):
    control_seg = auto()
    control_scribble = auto()
    control_fakescribbles = auto()
    control_openpose = auto()
    control_normal = auto()
    control_mlsd = auto()
    control_hough = auto()
    control_hed = auto()
    control_canny = auto()
    control_depth = auto()


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


class MODEL_REFERENCE_CATEGORY(StrEnum):
    """The categories of model reference entries."""

    blip = auto()
    clip = auto()
    codeformer = auto()
    controlnet = auto()
    esrgan = auto()
    gfpgan = auto()
    safety_checker = auto()
    stable_diffusion = auto()
    miscellaneous = auto()


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

    miscellaneous = auto()


class STABLE_DIFFUSION_BASELINE_CATEGORY(StrEnum):
    """An enum of all the image generation baselines."""

    stable_diffusion_1 = auto()
    stable_diffusion_2_768 = auto()
    stable_diffusion_2_512 = auto()
    stable_diffusion_xl = auto()
    stable_cascade = auto()
    flux_1 = auto()  # TODO: Extract flux and create "IMAGE_GENERATION_BASELINE_CATEGORY" due to name inconsistency


MODEL_PURPOSE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, MODEL_PURPOSE] = {
    MODEL_REFERENCE_CATEGORY.clip: MODEL_PURPOSE.clip,
    MODEL_REFERENCE_CATEGORY.blip: MODEL_PURPOSE.blip,
    MODEL_REFERENCE_CATEGORY.codeformer: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORY.controlnet: MODEL_PURPOSE.controlnet,
    MODEL_REFERENCE_CATEGORY.esrgan: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORY.gfpgan: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORY.safety_checker: MODEL_PURPOSE.post_processor,
    MODEL_REFERENCE_CATEGORY.stable_diffusion: MODEL_PURPOSE.image_generation,
    MODEL_REFERENCE_CATEGORY.miscellaneous: MODEL_PURPOSE.miscellaneous,
}

STABLE_DIFFUSION_BASELINE_NATIVE_RESOLUTION_LOOKUP: dict[STABLE_DIFFUSION_BASELINE_CATEGORY, int] = {
    STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_1: 512,
    STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_768: 768,
    STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_2_512: 512,
    STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_xl: 1024,
    STABLE_DIFFUSION_BASELINE_CATEGORY.stable_cascade: 1024,
    STABLE_DIFFUSION_BASELINE_CATEGORY.flux_1: 1024,
}
"""The single-side preferred resolution for each known stable diffusion baseline."""


def get_baseline_native_resolution(baseline: STABLE_DIFFUSION_BASELINE_CATEGORY) -> int:
    """
    Get the native resolution of a stable diffusion baseline.

    Args:
        baseline: The stable diffusion baseline.

    Returns:
        The native resolution of the baseline.
    """
    return STABLE_DIFFUSION_BASELINE_NATIVE_RESOLUTION_LOOKUP[baseline]
