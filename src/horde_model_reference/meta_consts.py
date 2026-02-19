from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto
from typing import Literal

from loguru import logger
from pydantic import BaseModel, model_validator
from strenum import StrEnum
from horde_model_reference.registries import DescriptorRegistry, EnumRegistry


class MODEL_STYLE(StrEnum):
    """An enum of all the model styles."""

    generalist = auto()
    anime = auto()
    furry = auto()
    artistic = auto()
    other = auto()
    realistic = auto()


_MODEL_STYLE_REGISTRY = EnumRegistry(item.value for item in MODEL_STYLE)


def register_model_style(style: MODEL_STYLE | str) -> None:
    """Register a new model style."""

    _MODEL_STYLE_REGISTRY.register(style)


def is_known_model_style(style: MODEL_STYLE | str) -> bool:
    """Check if a model style is known."""

    return _MODEL_STYLE_REGISTRY.is_known(style)


class CONTROLNET_STYLE(StrEnum):
    """An enum of all the ControlNet 'styles' - the process that defines the model's behavior.

    Examples include canny, depth, and openpose.
    """

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


_CONTROLNET_STYLE_REGISTRY = EnumRegistry(item.value for item in CONTROLNET_STYLE)


def register_controlnet_style(style: CONTROLNET_STYLE | str) -> None:
    """Register a new ControlNet style."""

    _CONTROLNET_STYLE_REGISTRY.register(style)


def is_known_controlnet_style(style: CONTROLNET_STYLE | str) -> bool:
    """Check if a ControlNet style is known."""

    return _CONTROLNET_STYLE_REGISTRY.is_known(style)


_KNOWN_TAGS_INITIAL = (
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
)

_TAG_REGISTRY = EnumRegistry(_KNOWN_TAGS_INITIAL)
KNOWN_TAGS = _TAG_REGISTRY.mutable_values()


def get_known_tags() -> list[str]:
    """Return a snapshot of all known tags as a list."""
    return sorted(_TAG_REGISTRY.values())


def register_tag(tag: str | StrEnum) -> None:
    """Register a new known tag."""

    _TAG_REGISTRY.register(tag)


def is_known_tag(tag: str | StrEnum) -> bool:
    """Check if a tag is known."""

    return _TAG_REGISTRY.is_known(tag)


class MODEL_DOMAIN(StrEnum):
    """The domain of a model, i.e., what it pertains to (image, text, video, etc.)."""

    image = auto()
    text = auto()
    video = auto()
    audio = auto()
    rendered_3d = auto()


_MODEL_DOMAIN_REGISTRY = EnumRegistry(item.value for item in MODEL_DOMAIN)


def register_model_domain(domain: MODEL_DOMAIN | str) -> None:
    """Register a new model domain."""

    _MODEL_DOMAIN_REGISTRY.register(domain)


def is_known_model_domain(domain: MODEL_DOMAIN | str) -> bool:
    """Check if a model domain is known."""

    return _MODEL_DOMAIN_REGISTRY.is_known(domain)


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


_MODEL_PURPOSE_REGISTRY = EnumRegistry(item.value for item in MODEL_PURPOSE)


def register_model_purpose(purpose: MODEL_PURPOSE | str) -> None:
    """Register a new model purpose."""

    _MODEL_PURPOSE_REGISTRY.register(purpose)


def is_known_model_purpose(purpose: MODEL_PURPOSE | str) -> bool:
    """Check if a model purpose is known."""

    return _MODEL_PURPOSE_REGISTRY.is_known(purpose)


class ModelClassification(BaseModel):
    """Contains specific information about how to categorize a model.

    This includes the model's `MODEL_DOMAIN` and `MODEL_PURPOSE`.
    """

    domain: MODEL_DOMAIN
    """The domain of the model, i.e., what it pertains to (image, text, video, etc.)"""

    purpose: MODEL_PURPOSE
    """The purpose of the model."""

    @model_validator(mode="after")
    def validator_known_purpose(self) -> ModelClassification:
        """Check if the purpose is known."""
        if not is_known_model_purpose(str(self.purpose)):
            logger.debug(f"Unknown purpose {self.purpose} for model classification {self}")
        if not is_known_model_domain(str(self.domain)):
            logger.debug(f"Unknown domain {self.domain} for model classification {self}")

        return self


class MODEL_REFERENCE_CATEGORY(StrEnum):
    """The categories of model reference entries."""

    blip = auto()
    clip = auto()
    codeformer = auto()
    controlnet = auto()
    esrgan = auto()
    gfpgan = auto()
    safety_checker = auto()
    image_generation = auto()
    text_generation = auto()
    video_generation = auto()
    audio_generation = auto()
    miscellaneous = auto()
    lora = auto()
    ti = auto()


@dataclass(frozen=True)
class CategoryDescriptor:
    """Describes a model reference category's traits in a single place."""

    domain: MODEL_DOMAIN
    """The ``MODEL_DOMAIN`` this category belongs to, e.g. image, text, video, etc."""
    purpose: MODEL_PURPOSE
    """The ``MODEL_PURPOSE`` of models in this category, e.g. generation, feature extraction, etc."""
    github_source: Literal["image", "text"] | None = None
    """Whether a legacy-format JSON file exists for this category. (e.g., ``"image"`` or ``"text"``).
    ``None`` means the category has no legacy GitHub source.
    """
    has_legacy_format: bool = True
    """Whether a legacy-format JSON file exists for this category."""
    managed_elsewhere: bool = False
    """Whether this category is managed by an external system."""
    filename_override: str | None = None
    """Non-default v2 filename (default is ``{category}.json``)."""
    legacy_filename_override: str | None = None
    """Non-default legacy filename (default matches v2)."""


github_image_model_reference_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""This distinguishes the original github repo locations and has no other meaning."""

github_text_model_reference_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""This distinguishes the original github repo locations and has no other meaning."""

no_legacy_format_available_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""Categories for which no legacy-format JSON file exists.."""

categories_managed_elsewhere: list[MODEL_REFERENCE_CATEGORY | str] = []
"""Categories that are managed by an external system."""

MODEL_CLASSIFICATION_LOOKUP: dict[MODEL_REFERENCE_CATEGORY | str, ModelClassification] = {}


def _rebuild_category_derived_data(
    data: dict[MODEL_REFERENCE_CATEGORY | str, CategoryDescriptor],
) -> None:
    """Rebuild derived category data from the registry."""

    global github_image_model_reference_categories
    global github_text_model_reference_categories
    global no_legacy_format_available_categories
    global categories_managed_elsewhere

    github_image_model_reference_categories = [c for c, d in data.items() if d.github_source == "image"]
    github_text_model_reference_categories = [c for c, d in data.items() if d.github_source == "text"]
    no_legacy_format_available_categories = [c for c, d in data.items() if not d.has_legacy_format]
    categories_managed_elsewhere = [c for c, d in data.items() if d.managed_elsewhere]

    MODEL_CLASSIFICATION_LOOKUP.clear()
    MODEL_CLASSIFICATION_LOOKUP.update({
        c: ModelClassification(domain=d.domain, purpose=d.purpose) for c, d in data.items()
    })


_CATEGORY_REGISTRY = DescriptorRegistry[MODEL_REFERENCE_CATEGORY | str, CategoryDescriptor](
    _rebuild_category_derived_data
)


def register_category(name: MODEL_REFERENCE_CATEGORY | str, descriptor: CategoryDescriptor) -> None:
    """Register a new model reference category."""

    _CATEGORY_REGISTRY.register(name, descriptor)


def get_github_image_categories() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Return categories whose legacy JSON lives in the image GitHub repo."""
    return list(github_image_model_reference_categories)


def get_github_text_categories() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Return categories whose legacy JSON lives in the text GitHub repo."""
    return list(github_text_model_reference_categories)


def get_no_legacy_format_categories() -> list[MODEL_REFERENCE_CATEGORY | str]:
    """Return categories that have no legacy-format JSON file."""
    return list(no_legacy_format_available_categories)


def get_model_classification(
    category: MODEL_REFERENCE_CATEGORY | str,
) -> ModelClassification:
    """Return the ModelClassification for *category*, or raise KeyError."""
    return MODEL_CLASSIFICATION_LOOKUP[category]


register_category(
    MODEL_REFERENCE_CATEGORY.blip,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.clip,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.codeformer,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.controlnet,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.esrgan,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.gfpgan,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.safety_checker,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.safety_checker,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.image_generation,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.generation,
        github_source="image",
        filename_override="stable_diffusion.json",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.text_generation,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.text,
        purpose=MODEL_PURPOSE.generation,
        github_source="text",
        filename_override="text_generation.json",
        legacy_filename_override="models.csv",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.video_generation,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.video,
        purpose=MODEL_PURPOSE.generation,
        has_legacy_format=False,
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.audio_generation,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.audio,
        purpose=MODEL_PURPOSE.generation,
        has_legacy_format=False,
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.miscellaneous,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.miscellaneous,
        github_source="image",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.lora,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        has_legacy_format=False,
        managed_elsewhere=True,
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.ti,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        has_legacy_format=False,
        managed_elsewhere=True,
    ),
)

_CATEGORY_REGISTRY.finalize()


def get_category_descriptor(category: MODEL_REFERENCE_CATEGORY | str) -> CategoryDescriptor:
    """Return the ``CategoryDescriptor`` for *category*.

    Raises:
        KeyError: If the category is not registered.
    """
    return _CATEGORY_REGISTRY.get(category)


def get_all_registered_categories() -> dict[MODEL_REFERENCE_CATEGORY | str, CategoryDescriptor]:
    """Return a shallow copy of the category registry."""

    return _CATEGORY_REGISTRY.all()


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------


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
    qwen_image = auto()
    z_image_turbo = auto()


@dataclass(frozen=True)
class BaselineDescriptor:
    """Describes a known image-generation baseline in a single place.

    Attributes:
        native_resolution: Preferred single-side resolution, or ``None`` for baselines
            like ``infer`` that have no fixed resolution.
        alternative_names: Alternative human/API names that map to this baseline.
    """

    native_resolution: int | None
    alternative_names: tuple[str, ...] = field(default_factory=tuple)


IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP: dict[KNOWN_IMAGE_GENERATION_BASELINE | str, int] = {}
"""The single-side preferred resolution for each known stable diffusion baseline."""

_ALTERNATIVE_NAME_TO_BASELINE: dict[str, KNOWN_IMAGE_GENERATION_BASELINE | str] = {}


def _rebuild_baseline_derived_data(
    data: dict[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor],
) -> None:
    """Rebuild derived baseline lookups from the registry."""

    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.clear()
    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.update({
        b: d.native_resolution for b, d in data.items() if d.native_resolution is not None
    })

    _ALTERNATIVE_NAME_TO_BASELINE.clear()
    for bl, desc in data.items():
        for alt in desc.alternative_names:
            _ALTERNATIVE_NAME_TO_BASELINE[alt] = bl


_IMAGE_BASELINE_REGISTRY = DescriptorRegistry[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor](
    _rebuild_baseline_derived_data
)


def register_image_baseline(name: KNOWN_IMAGE_GENERATION_BASELINE | str, descriptor: BaselineDescriptor) -> None:
    """Register a new image-generation baseline."""

    _IMAGE_BASELINE_REGISTRY.register(name, descriptor)


register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.infer,
    BaselineDescriptor(native_resolution=None),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    BaselineDescriptor(
        native_resolution=512,
        alternative_names=(
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
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768,
    BaselineDescriptor(native_resolution=768),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512,
    BaselineDescriptor(native_resolution=512),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "stable diffusion xl",
            "SDXL",
            "stable_diffusion_xl",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "stable_cascade",
            "stable cascade",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
    BaselineDescriptor(native_resolution=1024),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "flux_schnell",
            "flux schnell",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "flux_dev",
            "flux dev",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=("qwen_image", "qwen image", "qwen-image", "qwen"),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=("z_image_turbo", "z image turbo", "zimage-turbo", "zimage"),
    ),
)

_IMAGE_BASELINE_REGISTRY.finalize()

alternative_sdxl_baseline_names: list[str] = list(
    _IMAGE_BASELINE_REGISTRY.get(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl).alternative_names,
)


def _matching_image_baseline_exists(
    baseline: str,
    known_image_generation_baseline: KNOWN_IMAGE_GENERATION_BASELINE | str,
) -> bool:
    """Return True if *baseline* is a recognized alternative name for *known_image_generation_baseline*.

    Args:
        baseline: The baseline name to look up.
        known_image_generation_baseline: The known image generation baseline to check against.

    Returns:
        True if the baseline name matches the given known baseline, False otherwise.
    """
    desc = _IMAGE_BASELINE_REGISTRY.get(known_image_generation_baseline)
    if desc is not None and desc.alternative_names:
        return baseline in desc.alternative_names
    return baseline == str(known_image_generation_baseline)


def is_known_image_baseline(baseline: str) -> bool:
    """Return True if *baseline* is a known baseline or alternative name.

    Args:
        baseline: The baseline name to check.

    Returns:
        True if the baseline is known, False otherwise.
    """
    return _IMAGE_BASELINE_REGISTRY.contains(baseline) or baseline in _ALTERNATIVE_NAME_TO_BASELINE


def get_baseline_descriptor(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> BaselineDescriptor:
    """Return the ``BaselineDescriptor`` for *baseline*.

    Args:
        baseline: The known image generation baseline (enum member or plain string).

    Raises:
        KeyError: If the baseline is not registered.
    """
    return _IMAGE_BASELINE_REGISTRY.get(baseline)


def get_all_registered_baselines() -> dict[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor]:
    """Return a shallow copy of the baseline registry.

    This includes both built-in ``KNOWN_IMAGE_GENERATION_BASELINE`` members and
    any externally registered baselines.
    """
    return _IMAGE_BASELINE_REGISTRY.all()


def get_baseline_native_resolution(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> int:
    """Get the native resolution of a stable diffusion baseline.

    Args:
        baseline: The stable diffusion baseline (enum member or plain string).

    Returns:
        The native resolution of the baseline.
    """
    return IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP[baseline]


def get_baselines_by_resolution(resolution: int) -> list[KNOWN_IMAGE_GENERATION_BASELINE | str]:
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


_unregistered_categories = {c for c in MODEL_REFERENCE_CATEGORY if not _CATEGORY_REGISTRY.contains(c)}
if _unregistered_categories:
    raise RuntimeError(
        f"MODEL_REFERENCE_CATEGORY members not registered in _CATEGORY_REGISTRY: {_unregistered_categories}"
    )

_unregistered_baselines = {b for b in KNOWN_IMAGE_GENERATION_BASELINE if not _IMAGE_BASELINE_REGISTRY.contains(b)}
if _unregistered_baselines:
    raise RuntimeError(
        f"KNOWN_IMAGE_GENERATION_BASELINE members not registered in _BASELINE_REGISTRY: {_unregistered_baselines}"
    )


class TEXT_BACKENDS(StrEnum):
    """An enum of all the text backends."""

    aphrodite = auto()
    koboldcpp = auto()


_TEXT_BACKEND_REGISTRY = EnumRegistry(item.value for item in TEXT_BACKENDS)
KNOWN_TEXT_BACKENDS = _TEXT_BACKEND_REGISTRY.mutable_values()


def register_text_backend(backend: str) -> None:
    """Register a new text backend.

    Args:
        backend: The text backend to register.
    """
    _TEXT_BACKEND_REGISTRY.register(backend)


def is_known_text_backend(backend: str) -> bool:
    """Check if a text backend is known.

    Args:
        backend: The text backend to check.

    Returns:
        True if the text backend is known, False otherwise.
    """
    return _TEXT_BACKEND_REGISTRY.is_known(backend)
