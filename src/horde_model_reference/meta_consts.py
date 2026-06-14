"""Domain enums, category descriptors, and runtime registries for model reference metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from typing import Literal

from loguru import logger
from pydantic import BaseModel, model_validator
from strenum import StrEnum

from horde_model_reference.model_consts.image import (
    CONTROLNET_STYLE,
    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP,
    KNOWN_IMAGE_GENERATION_BASELINE,
    BaselineDescriptor,
    get_all_registered_baselines,
    get_baseline_descriptor,
    get_baseline_native_resolution,
    get_baselines_by_resolution,
    is_known_controlnet_style,
    is_known_image_baseline,
    register_controlnet_style,
    register_image_baseline,
)
from horde_model_reference.model_consts.shared import (
    KNOWN_TAGS,
    MODEL_STYLE,
    get_known_tags,
    is_known_model_style,
    is_known_tag,
    register_model_style,
    register_tag,
)
from horde_model_reference.model_consts.text import (
    KNOWN_TEXT_BACKENDS,
    TEXT_BACKENDS,
    is_known_text_backend,
    register_text_backend,
)
from horde_model_reference.registries import DescriptorRegistry, EnumRegistry


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
    on_disk_folder_name: str | None = None
    """The folder under the model-weights root where this category's files live (e.g. ``"compvis"`` for
    image generation). ``None`` for categories with no on-disk weights folder in this ecosystem (e.g.
    text/video/audio generation)."""
    weights_marker: bool = False
    """Whether this category's folder is one of the markers used to locate the model-weights root.

    See :func:`horde_model_reference.on_disk_layout.resolve_weights_root`."""
    managed_download_elsewhere: bool = False
    """Whether downloading is handled outside the horde_model_reference engine.

    ``True`` for categories whose files are fetched by a specialised external mechanism (e.g. LoRA/TI via
    CivitAI in hordelib); the :mod:`horde_model_reference.download_engine` is not used for them."""


github_image_model_reference_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""This distinguishes the original github repo locations and has no other meaning."""

github_text_model_reference_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""This distinguishes the original github repo locations and has no other meaning."""

no_legacy_format_available_categories: list[MODEL_REFERENCE_CATEGORY | str] = []
"""Categories for which no legacy-format JSON file exists.."""

categories_managed_elsewhere: list[MODEL_REFERENCE_CATEGORY | str] = []
"""Categories that are managed by an external system."""

MODEL_CLASSIFICATION_LOOKUP: dict[MODEL_REFERENCE_CATEGORY | str, ModelClassification] = {}

WEIGHTS_MARKER_FOLDERS: tuple[str, ...] = ()
"""Folder names whose joint presence marks the model-weights root (derived from the category registry)."""


def _rebuild_category_derived_data(
    data: dict[MODEL_REFERENCE_CATEGORY | str, CategoryDescriptor],
) -> None:
    """Rebuild derived category data from the registry."""
    global github_image_model_reference_categories
    global github_text_model_reference_categories
    global no_legacy_format_available_categories
    global categories_managed_elsewhere
    global WEIGHTS_MARKER_FOLDERS

    github_image_model_reference_categories = [c for c, d in data.items() if d.github_source == "image"]
    github_text_model_reference_categories = [c for c, d in data.items() if d.github_source == "text"]
    no_legacy_format_available_categories = [c for c, d in data.items() if not d.has_legacy_format]
    categories_managed_elsewhere = [c for c, d in data.items() if d.managed_elsewhere]
    WEIGHTS_MARKER_FOLDERS = tuple(
        d.on_disk_folder_name for d in data.values() if d.weights_marker and d.on_disk_folder_name is not None
    )

    MODEL_CLASSIFICATION_LOOKUP.clear()
    MODEL_CLASSIFICATION_LOOKUP.update(
        {c: ModelClassification(domain=d.domain, purpose=d.purpose) for c, d in data.items()}
    )


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
        on_disk_folder_name="blip",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.clip,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.feature_extractor,
        github_source="image",
        on_disk_folder_name="clip",
        weights_marker=True,
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.codeformer,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
        on_disk_folder_name="codeformer",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.controlnet,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        github_source="image",
        on_disk_folder_name="controlnet",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.esrgan,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
        on_disk_folder_name="esrgan",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.gfpgan,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.post_processing,
        github_source="image",
        on_disk_folder_name="gfpgan",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.safety_checker,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.safety_checker,
        github_source="image",
        on_disk_folder_name="safety_checker",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.image_generation,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.generation,
        github_source="image",
        filename_override="stable_diffusion.json",
        on_disk_folder_name="compvis",
        weights_marker=True,
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
        on_disk_folder_name="miscellaneous",
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.lora,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        has_legacy_format=False,
        managed_elsewhere=True,
        on_disk_folder_name="lora",
        managed_download_elsewhere=True,
    ),
)
register_category(
    MODEL_REFERENCE_CATEGORY.ti,
    CategoryDescriptor(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.auxiliary_or_patch,
        has_legacy_format=False,
        managed_elsewhere=True,
        on_disk_folder_name="ti",
        managed_download_elsewhere=True,
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


def get_weights_marker_folders() -> tuple[str, ...]:
    """Return the folder names whose joint presence marks the model-weights root."""
    return WEIGHTS_MARKER_FOLDERS


_unregistered_categories = {c for c in MODEL_REFERENCE_CATEGORY if not _CATEGORY_REGISTRY.contains(c)}
if _unregistered_categories:
    raise RuntimeError(
        f"MODEL_REFERENCE_CATEGORY members not registered in _CATEGORY_REGISTRY: {_unregistered_categories}"
    )


__all__ = [
    "CONTROLNET_STYLE",
    "IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP",
    "KNOWN_IMAGE_GENERATION_BASELINE",
    "KNOWN_TAGS",
    "KNOWN_TEXT_BACKENDS",
    "MODEL_DOMAIN",
    "MODEL_PURPOSE",
    "MODEL_REFERENCE_CATEGORY",
    "MODEL_STYLE",
    "TEXT_BACKENDS",
    "BaselineDescriptor",
    "CategoryDescriptor",
    "ModelClassification",
    "get_all_registered_baselines",
    "get_all_registered_categories",
    "get_baseline_descriptor",
    "get_baseline_native_resolution",
    "get_baselines_by_resolution",
    "get_category_descriptor",
    "get_known_tags",
    "get_weights_marker_folders",
    "is_known_controlnet_style",
    "is_known_image_baseline",
    "is_known_model_domain",
    "is_known_model_purpose",
    "is_known_model_style",
    "is_known_tag",
    "is_known_text_backend",
    "register_category",
    "register_controlnet_style",
    "register_image_baseline",
    "register_model_domain",
    "register_model_purpose",
    "register_model_style",
    "register_tag",
    "register_text_backend",
]
