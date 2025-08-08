from __future__ import annotations

from enum import auto

from loguru import logger
from pydantic import BaseModel, model_validator
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
    control_qr = auto()
    control_qr_xl = auto()


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
    image_generation = "image_generation"
    stable_diffusion = "image_generation"  # alias for backward compatibility
    miscellaneous = auto()


class MODEL_DOMAIN(StrEnum):
    """The domain of a model, i.e., what it pertains to (image, text, video, etc.)."""

    image = auto()
    text = auto()
    video = auto()
    audio = auto()
    rendered_3d = auto()


class MODEL_PURPOSE(StrEnum):
    """The primary purpose of a model, for example, image generation or feature extraction."""

    generation = auto()
    """The model is the central part of a generative AI system."""

    post_processing = auto()
    """The model is used for post-processing user input or generation tasks."""

    auxiliary_or_patch = auto()
    """The model is an auxiliary or patch model, e.g. LoRA or ControlNet."""

    feature_extractor = auto()
    """The model is a feature extractor, e.g. CLIP or BLIP."""

    safety_checker = auto()
    """A special case of feature extraction."""

    miscellaneous = auto()
    """The model does not fit into any other category or is very specialized."""


class ModelClassification(BaseModel):
    """Contains specific information about how to categorize a model.

    This includes the model's `MODEL_DOMAIN` and `MODEL_PURPOSE`.
    """

    domain: MODEL_DOMAIN | str
    """The domain of the model, i.e., what it pertains to (image, text, video, etc.)"""

    purpose: MODEL_PURPOSE | str
    """The purpose of the model."""

    @model_validator(mode="after")
    def validator_known_purpose(self) -> ModelClassification:
        """Check if the purpose is known."""
        if str(self.purpose) not in MODEL_PURPOSE.__members__:
            logger.debug(f"Unknown purpose {self.purpose} for model classification {self}")
        if str(self.domain) not in MODEL_DOMAIN.__members__:
            logger.debug(f"Unknown domain {self.domain} for model classification {self}")

        return self


class KNOWN_IMAGE_GENERATION_BASELINE(StrEnum):
    """An enum of all the image generation baselines."""

    infer = auto()
    """The baseline is not known and should be inferred from the model name."""

    stable_diffusion_1 = auto()
    stable_diffusion_2_768 = auto()
    stable_diffusion_2_512 = auto()
    stable_diffusion_xl = auto()
    stable_cascade = auto()
    flux_1 = auto()  # TODO: Extract flux and create "IMAGE_GENERATION_BASELINE_CATEGORY" due to name inconsistency
    flux_schnell = auto()  # FIXME
    flux_dev = auto()  # FIXME


STABLE_DIFFUSION_BASELINE_CATEGORY = KNOWN_IMAGE_GENERATION_BASELINE
"""Deprecated: Use KNOWN_IMAGE_GENERATION_BASELINE instead."""

_alternative_sd1_baseline_names = [
    "stable diffusion 1",
    "stable diffusion 1.4",
    "stable diffusion 1.5",
    "SD1",
    "SD14",
    "SD1.4",
    "SD15",
    "SD1.5",
    "stable_diffusion",
    "stable_diffusion_1",
    "stable_diffusion_1.4",
    "stable_diffusion_1.5",
]

alternative_sdxl_baseline_names = [
    "stable diffusion xl",
    "SDXL",
    "stable_diffusion_xl",
]

_alternative_flux_schnell_baseline_names = [
    "flux_schnell",
    "flux schnell",
]

_alternative_flux_dev_baseline_names = [
    "flux_dev",
    "flux dev",
]

_alternative_stable_cascade_baseline_names = [
    "stable_cascade",
    "stable cascade",
]


def matching_baseline_exists(
    baseline: str,
    known_image_generation_baseline: KNOWN_IMAGE_GENERATION_BASELINE,
) -> bool:
    """Return True if a matching baseline exists.

    Args:
        baseline (str): The baseline name.
        known_image_generation_baseline (KNOWN_IMAGE_GENERATION_BASELINE): The known image generation baseline to
            check against.

    Returns:
        True if the baseline name is of the category, False otherwise.
    """
    if known_image_generation_baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1:
        return baseline in _alternative_sd1_baseline_names
    if known_image_generation_baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl:
        return baseline in alternative_sdxl_baseline_names
    if known_image_generation_baseline == KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell:
        return baseline in _alternative_flux_schnell_baseline_names
    if known_image_generation_baseline == KNOWN_IMAGE_GENERATION_BASELINE.flux_dev:
        return baseline in _alternative_flux_dev_baseline_names
    if known_image_generation_baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade:
        return baseline in _alternative_stable_cascade_baseline_names

    return baseline == known_image_generation_baseline.name


MODEL_CLASSIFICATION_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, ModelClassification] = {
    MODEL_REFERENCE_CATEGORY.clip: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
    ),
    MODEL_REFERENCE_CATEGORY.blip: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
    ),
    MODEL_REFERENCE_CATEGORY.codeformer: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
    ),
    MODEL_REFERENCE_CATEGORY.controlnet: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
    ),
    MODEL_REFERENCE_CATEGORY.esrgan: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
    ),
    MODEL_REFERENCE_CATEGORY.gfpgan: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
    ),
    MODEL_REFERENCE_CATEGORY.safety_checker: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
    ),
    MODEL_REFERENCE_CATEGORY.image_generation: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
    ),
    MODEL_REFERENCE_CATEGORY.miscellaneous: ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.miscellaneous,
    ),
}

IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP: dict[KNOWN_IMAGE_GENERATION_BASELINE, int] = {
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1: 512,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768: 768,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512: 512,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl: 1024,
    KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade: 1024,
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1: 1024,
}
"""The single-side preferred resolution for each known stable diffusion baseline."""


def get_baseline_native_resolution(baseline: KNOWN_IMAGE_GENERATION_BASELINE) -> int:
    """Get the native resolution of a stable diffusion baseline.

    Args:
        baseline: The stable diffusion baseline.

    Returns:
        The native resolution of the baseline.
    """
    return IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP[baseline]


def get_baselines_by_resolution(resolution: int) -> list[KNOWN_IMAGE_GENERATION_BASELINE]:
    """Get all baselines that have the given native resolution.

    Args:
        resolution: The native resolution to look for.

    Returns:
        A list of baselines that have the given native resolution.
    """
    return [
        baseline
        for baseline, native_resolution in IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.items()
        if native_resolution == resolution
    ]
