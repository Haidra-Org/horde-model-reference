"""The model database pydantic models and associate enums/lookups."""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Iterator
from typing import TypeVar

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    RootModel,
    model_validator,
)

from horde_model_reference import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    ModelClassification,
)
from horde_model_reference.meta_consts import CONTROLNET_STYLE, MODEL_PURPOSE


class DownloadRecord(BaseModel):  # TODO Rename? (record to subrecord?)
    """A record of a file to download for a model. Typically a ckpt file."""

    model_config = ConfigDict(extra="ignore")

    file_name: str
    """The horde specific filename. This is not necessarily the same as the file's name on the model host."""
    file_url: str
    """The fully qualified URL to download the file from."""
    sha256sum: str
    """The sha256sum of the file."""
    file_type: str | None = None
    known_slow_download: bool | None = None
    """Whether the download is known to be slow or not."""


class Generic_ModelRecord(BaseModel):
    """A generic model reference record."""

    # TODO forbid extra?
    name: str
    """The name of the model."""
    description: str | None = None
    """A short description of the model."""
    version: str | None = None
    """The version of the  model (not the version of SD it is based on, see `baseline` for that info)."""

    config: dict[str, list[DownloadRecord]]
    """A dictionary of any configuration files and information on where to download the model file(s)."""

    model_classification: ModelClassification
    """The classification of the model."""

    # model_reference_category: MODEL_REFERENCE_CATEGORY
    # """The reference category of the model, for example BLIP, CLIP, etc."""

    features_not_supported: list[str] | None = None


MODEL_RECORD_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, type[Generic_ModelRecord]] = {}


def register_record_type(
    category: MODEL_REFERENCE_CATEGORY,
) -> Callable[[type[Generic_ModelRecord]], type[Generic_ModelRecord]]:
    """Register a model record type with its category."""

    def decorator(cls: type[Generic_ModelRecord]) -> type[Generic_ModelRecord]:
        if category in MODEL_RECORD_TYPE_LOOKUP:
            logger.warning(
                f"Overriding existing record type for category {category}: "
                f"{MODEL_RECORD_TYPE_LOOKUP[category]} -> {cls}",
            )
        MODEL_RECORD_TYPE_LOOKUP[category] = cls
        return cls

    return decorator


@register_record_type(MODEL_REFERENCE_CATEGORY.image_generation)
class ImageGeneration_ModelRecord(Generic_ModelRecord):
    """A model entry in the model reference."""

    model_config = ConfigDict(extra="ignore")

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        ),
    )
    """The domain (e.g., image, text) and purpose (e.g., generation, classification) of the model."""

    inpainting: bool | None = False
    """If this is an inpainting model or not."""
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str
    """The model on which this model is based."""
    optimization: str | None = None
    """The optimization type of the model."""
    tags: list[str] | None = []
    """Any tags associated with the model which may be useful for searching."""
    showcases: list[str] | None = []
    """Links to any showcases of the model which illustrate its style."""
    min_bridge_version: int | None = None
    """The minimum version of AI-Horde-Worker required to use this model."""
    trigger: list[str] | None = []
    """A list of trigger words or phrases which can be used to activate the model."""
    homepage: str | None = None
    """A link to the model's homepage."""
    nsfw: bool
    """Whether the model is NSFW or not."""

    style: MODEL_STYLE | str | None = None
    """The style of the model."""

    requirements: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None

    size_on_disk_bytes: int | None = None

    @model_validator(mode="after")
    def validator_set_arrays_to_empty_if_none(self) -> ImageGeneration_ModelRecord:
        """Set any `None` values to empty lists."""
        if self.tags is None:
            self.tags = []
        if self.showcases is None:
            self.showcases = []
        if self.trigger is None:
            self.trigger = []
        return self

    @model_validator(mode="after")
    def validator_is_baseline_and_style_known(self) -> ImageGeneration_ModelRecord:
        """Check if the baseline is known."""
        if str(self.baseline) not in KNOWN_IMAGE_GENERATION_BASELINE.__members__:
            logger.debug(f"Unknown baseline {self.baseline} for model {self.name}")

        if self.style is not None and str(self.style) not in MODEL_STYLE.__members__:
            logger.debug(f"Unknown style {self.style} for model {self.name}")

        return self


@register_record_type(MODEL_REFERENCE_CATEGORY.clip)
class CLIP_ModelRecord(Generic_ModelRecord):
    """A CLIP model entry in the model reference."""

    pretrained_name: str | None = None
    """The huggingface-style pretrained model name."""


@register_record_type(MODEL_REFERENCE_CATEGORY.controlnet)
class ControlNet_ModelRecord(Generic_ModelRecord):
    """A ControlNet model entry in the model reference."""

    style: CONTROLNET_STYLE | str | None = None
    """The 'style' (purpose) of the controlnet. See `CONTROLNET_STYLE` for all possible values and more info."""

    @model_validator(mode="after")
    def validator_is_style_known(self) -> ControlNet_ModelRecord:
        """Check if the style is known."""
        if self.style is not None and str(self.style) not in CONTROLNET_STYLE.__members__:
            logger.debug(f"Unknown style {self.style} for model {self.name}")

        return self


MR = TypeVar("MR", bound=Generic_ModelRecord)


class Generic_ModelReference(RootModel[dict[str, MR]]):
    """A generic model reference."""

    root: dict[str, MR]
    """A dictionary of all the models."""

    def __iter__(self) -> Iterator[tuple[str, MR]]:  # type: ignore
        return iter(self.root.items())

    def __getitem__(self, item: str) -> MR:
        return self.root[item]


class ImageGeneration_ModelReference(Generic_ModelReference[ImageGeneration_ModelRecord]):
    """The combined metadata and model list."""

    _baseline: dict[KNOWN_IMAGE_GENERATION_BASELINE | str, int] = PrivateAttr(default_factory=dict)
    """A dictionary of all the baseline types and how many models use them."""
    _styles: dict[MODEL_STYLE | str, int] = PrivateAttr(default_factory=dict)
    """A dictionary of all the styles and how many models use them."""
    _tags: dict[str, int] = PrivateAttr(default_factory=dict)
    """A dictionary of all the tags and how many models use them."""
    _models_hosts: dict[str, int] = PrivateAttr(default_factory=dict)
    """A dictionary of all the model hosts and how many models use them."""
    root: dict[str, ImageGeneration_ModelRecord]
    """A dictionary of all the models."""

    _models_dict_hash: int | None = None

    def check_was_models_modified(self) -> bool:
        """Check if the models dictionary has been modified since the last time it was hashed.

        Note: this only checks if the keys differ, and does not check into the value objects.
        You should call `rebuild_metadata` if you want to be sure the metadata is up to date.
        """
        if not self._models_dict_hash or self._models_dict_hash != hash(frozenset(self.root.keys())):
            self._models_dict_hash = hash(frozenset(self.root.keys()))
            self.rebuild_metadata()
            return True

        return False

    def rebuild_metadata(self) -> None:
        """Rebuild the metadata dictionaries."""
        # Initialize empty dictionaries to store metadata
        self._baseline = {}  # Dictionary to count models by baseline
        self._styles = {}  # Dictionary to count models by style
        self._tags = {}  # Dictionary to count models by tag
        self._models_hosts = {}  # Dictionary to count models by host

        # Iterate over all models in the dictionary
        for model in self.root.values():
            # Count models by baseline
            self._baseline[model.baseline] = self._baseline.get(model.baseline, 0) + 1

            # Count models by style, if the style is valid
            if model.style is not None and model.style in MODEL_STYLE.__members__:
                self._styles[MODEL_STYLE(model.style)] = self._styles.get(MODEL_STYLE(model.style), 0) + 1

            # Count models by tag
            if model.tags is not None:
                for tag in model.tags:
                    self._tags[tag] = self._tags.get(tag, 0) + 1

            # Count models by host, based on the download URLs in the config
            for entries in model.config.items():
                entry_name, data = entries

                # Look for the "download" entry in the config
                if entry_name == "download":
                    for download_entry in data:
                        # Extract the host from the download URL
                        host = urllib.parse.urlparse(download_entry.file_url).netloc
                        # Count models by host
                        self._models_hosts[host] = self._models_hosts.get(host, 0) + 1

    @property
    def baseline(self) -> dict[KNOWN_IMAGE_GENERATION_BASELINE | str, int]:
        """Return a dictionary of all the baseline types and how many models use them."""
        self.check_was_models_modified()
        return self._baseline

    @property
    def styles(self) -> dict[MODEL_STYLE | str, int]:
        """Return a dictionary of all the styles and how many models use them."""
        self.check_was_models_modified()
        return self._styles

    @property
    def tags(self) -> dict[str, int]:
        """Return a dictionary of all the tags and how many models use them."""
        self.check_was_models_modified()
        return self._tags

    @property
    def models_hosts(self) -> dict[str, int]:
        """Return a dictionary of all the model hosts and how many models use them."""
        self.check_was_models_modified()
        return self._models_hosts

    @property
    def models_names(self) -> set[str]:
        """Return a list of all the model names."""
        return set(self.root.keys())

    def get_model_baseline(self, model_name: str) -> KNOWN_IMAGE_GENERATION_BASELINE | str | None:
        """Return the baseline for a given model name."""
        model: ImageGeneration_ModelRecord | None = self.root.get(model_name)

        return model.baseline if model else None

    def get_model_style(self, model_name: str) -> MODEL_STYLE | str | None:
        """Return the style for a given model name."""
        model: ImageGeneration_ModelRecord | None = self.root.get(model_name)

        return model.style if model else None

    def get_model_tags(self, model_name: str) -> list[str] | None:
        """Return the tags for a given model name."""
        model: ImageGeneration_ModelRecord | None = self.root.get(model_name)

        return model.tags if model else None


StableDiffusion_ModelReference = ImageGeneration_ModelReference


class CLIP_ModelReference(Generic_ModelReference[CLIP_ModelRecord]):
    """A CLIP model reference."""

    root: dict[str, CLIP_ModelRecord]
    """A dictionary of all the models."""


class ControlNet_ModelReference(Generic_ModelReference[ControlNet_ModelRecord]):
    """A ControlNet model reference."""

    root: dict[str, ControlNet_ModelRecord]
    """A dictionary of all the models."""


for category in MODEL_REFERENCE_CATEGORY:
    if category not in MODEL_RECORD_TYPE_LOOKUP:
        logger.trace(f"No record type registered for category {category}. Using Generic_ModelReference.")
        MODEL_RECORD_TYPE_LOOKUP[category] = Generic_ModelRecord

KNOWN_MODEL_REFERENCE_INSTANCES = (
    ImageGeneration_ModelReference
    | ControlNet_ModelReference
    | CLIP_ModelReference
    | Generic_ModelReference[Generic_ModelRecord]
)
"""A type alias for any known model reference instance. Note the absence of `type[]` here. \
    Use `KNOWN_MODEL_REFERENCE_TYPES` for the corresponding `type[]` objects."""

KNOWN_MODEL_REFERENCE_TYPES = (
    type[Generic_ModelReference[Generic_ModelRecord]]
    | type[ImageGeneration_ModelReference]
    | type[CLIP_ModelReference]
    | type[ControlNet_ModelReference]
)
"""A type alias for any known model reference (python) type. Note the use of `type[]` to indicate a `type` \
    object."""


MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_TYPES] = {
    MODEL_REFERENCE_CATEGORY.image_generation: ImageGeneration_ModelReference,
    MODEL_REFERENCE_CATEGORY.controlnet: ControlNet_ModelReference,
    MODEL_REFERENCE_CATEGORY.clip: CLIP_ModelReference,
    MODEL_REFERENCE_CATEGORY.blip: Generic_ModelReference[Generic_ModelRecord],
    MODEL_REFERENCE_CATEGORY.esrgan: Generic_ModelReference[Generic_ModelRecord],
    MODEL_REFERENCE_CATEGORY.gfpgan: Generic_ModelReference[Generic_ModelRecord],
    MODEL_REFERENCE_CATEGORY.safety_checker: Generic_ModelReference[Generic_ModelRecord],
    MODEL_REFERENCE_CATEGORY.codeformer: Generic_ModelReference[Generic_ModelRecord],
    MODEL_REFERENCE_CATEGORY.miscellaneous: Generic_ModelReference[Generic_ModelRecord],
}
"""A lookup for the model reference type based on the model category. See also `MODEL_RECORD_TYPE_LOOKUP`."""
