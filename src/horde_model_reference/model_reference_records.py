"""The model database pydantic models and associate enums/lookups."""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from horde_model_reference import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    SCHEMA_VERSION,
    ModelClassification,
    ai_horde_ci_settings,
)
from horde_model_reference.meta_consts import CONTROLNET_STYLE, MODEL_PURPOSE


def get_default_config() -> ConfigDict:
    """Get the default config for model records based on whether AI Horde is being tested or not."""
    if ai_horde_ci_settings.ai_horde_testing:
        return ConfigDict(extra="forbid", populate_by_name=True)
    return ConfigDict(extra="ignore", populate_by_name=True)


class DownloadRecord(BaseModel):  # TODO Rename? (record to subrecord?)
    """A record of a file to download for a model. Typically a ckpt file."""

    model_config = get_default_config()

    file_name: str
    """The horde specific filename. This is not necessarily the same as the file's name on the model host."""
    file_url: str
    """The fully qualified URL to download the file from."""
    sha256sum: str = "FIXME"
    """The sha256sum of the file."""
    file_purpose: str | None = None
    """"""
    known_slow_download: bool | None = None
    """Whether the download is known to be slow or not."""


class GenericModelRecordConfig(BaseModel):
    """Configuration for a generic model record."""

    model_config = get_default_config()

    download: list[DownloadRecord] = Field(
        default_factory=list,
    )
    """A list of files to download for the model."""


class GenericModelRecordMetadata(BaseModel):
    """Metadata for a generic model record."""

    model_config = get_default_config()

    schema_version: str = SCHEMA_VERSION
    """The version of the schema used to create this record."""

    created_at: int | None = None
    """The Unix time of when the record was created."""
    updated_at: int | None = None
    """The Unix time of when the record was last updated."""

    created_by: str | None = None
    """The name or identifier of the person or system which created the record."""
    updated_by: str | None = None
    """The name or identifier of the person or system which last updated the record."""


class FineTuneSeriesInfo(BaseModel):
    """Information about a fine-tuning series."""

    model_config = get_default_config()

    name: str
    """The name of the fine-tuning series."""
    version: str | None = None
    """The version of the fine-tuning series."""

    author: str | None = None
    """The author of the fine-tuning series."""

    description: str | None = None
    """A short description of the fine-tuning series."""
    homepage: str | None = None
    """A link to the homepage of the fine-tuning series."""


class GenericModelRecord(BaseModel):
    """A generic model reference record."""

    model_config = get_default_config()

    record_type: MODEL_REFERENCE_CATEGORY
    """Discriminator field for polymorphic deserialization. Identifies the specific record type."""

    name: str
    """The name of the model."""
    description: str | None = None
    """A short description of the model."""
    version: str | None = None
    """The version of the  model (not the version of SD it is based on, see `baseline` for that info)."""

    finetune_series: FineTuneSeriesInfo | None = None
    """Information about the fine-tuning of the model. For image, some examples are 'Pony', 'Illustrious", etc."""

    metadata: GenericModelRecordMetadata = Field(
        default_factory=GenericModelRecordMetadata,
    )
    """Metadata about the record itself, such as creation and update times."""

    config: GenericModelRecordConfig = Field(
        default_factory=GenericModelRecordConfig,
    )
    """A dictionary of any configuration files and information on where to download the model file(s)."""

    model_classification: ModelClassification
    """The classification of the model."""


MODEL_RECORD_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY, type[GenericModelRecord]] = {}


def register_record_type(
    category: MODEL_REFERENCE_CATEGORY,
) -> Callable[[type[GenericModelRecord]], type[GenericModelRecord]]:
    """Register a model record type with its category."""

    def decorator(cls: type[GenericModelRecord]) -> type[GenericModelRecord]:
        if category in MODEL_RECORD_TYPE_LOOKUP:
            logger.warning(
                f"Overriding existing record type for category {category}: "
                f"{MODEL_RECORD_TYPE_LOOKUP[category]} -> {cls}",
            )
        MODEL_RECORD_TYPE_LOOKUP[category] = cls
        return cls

    return decorator


@register_record_type(MODEL_REFERENCE_CATEGORY.image_generation)
class ImageGenerationModelRecord(GenericModelRecord):
    """A model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.image_generation
    """Discriminator field identifying this as an image generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        ),
    )
    """The domain (e.g., image, text) and purpose (e.g., generation, classification) of the model."""

    inpainting: bool | None = False
    """If this is an inpainting model or not."""
    baseline: KNOWN_IMAGE_GENERATION_BASELINE
    """The model on which this model is based."""
    optimization: str | None = None
    """The optimization type of the model."""
    tags: list[str] | None = None
    """Any tags associated with the model which may be useful for searching."""
    showcases: list[str] | None = None
    """Links to any showcases of the model which illustrate its style."""
    min_bridge_version: int | None = None
    """The minimum version of AI-Horde-Worker required to use this model."""
    trigger: list[str] | None = None
    """A list of trigger words or phrases which can be used to activate the model."""
    homepage: str | None = None
    """A link to the model's homepage."""
    nsfw: bool
    """Whether the model is NSFW or not."""

    style: MODEL_STYLE | None = None
    """The style of the model."""

    requirements: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None

    size_on_disk_bytes: int | None = None

    @model_validator(mode="after")
    def validator_set_arrays_to_empty_if_none(self) -> ImageGenerationModelRecord:
        """Set any `None` values to empty lists."""
        if self.tags is None:
            self.tags = []
        if self.showcases is None:
            self.showcases = []
        if self.trigger is None:
            self.trigger = []
        return self

    @model_validator(mode="after")
    def validator_is_baseline_and_style_known(self) -> ImageGenerationModelRecord:
        """Check if the baseline is known."""
        if str(self.baseline) not in KNOWN_IMAGE_GENERATION_BASELINE.__members__:
            logger.debug(f"Unknown baseline {self.baseline} for model {self.name}")

        if self.style is not None and str(self.style) not in MODEL_STYLE.__members__:
            logger.debug(f"Unknown style {self.style} for model {self.name}")

        return self


@register_record_type(MODEL_REFERENCE_CATEGORY.controlnet)
class ControlNetModelRecord(GenericModelRecord):
    """A ControlNet model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.controlnet
    """Discriminator field identifying this as a ControlNet model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.auxiliary_or_patch,
        )
    )

    controlnet_style: CONTROLNET_STYLE
    """The 'style' (purpose) of the controlnet. See `CONTROLNET_STYLE` for all possible values and more info."""

    @model_validator(mode="after")
    def validator_is_style_known(self) -> ControlNetModelRecord:
        """Check if the style is known."""
        if self.controlnet_style is not None and str(self.controlnet_style) not in CONTROLNET_STYLE.__members__:
            logger.debug(f"Unknown style {self.controlnet_style} for model {self.name}")

        return self


@register_record_type(MODEL_REFERENCE_CATEGORY.text_generation)
class TextGenerationModelRecord(GenericModelRecord):
    """A text generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.text_generation
    """Discriminator field identifying this as a text generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.text,
            purpose=MODEL_PURPOSE.generation,
        )
    )

    baseline: str | None = None
    parameters_count: int = Field(alias="parameters")
    nsfw: bool = False
    style: str | None = None
    display_name: str | None = None
    url: str | None = None
    tags: list[str] | None = None
    settings: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None


@register_record_type(MODEL_REFERENCE_CATEGORY.blip)
class BlipModelRecord(GenericModelRecord):
    """A BLIP model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.blip
    """Discriminator field identifying this as a BLIP model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.feature_extractor,
        )
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.clip)
class ClipModelRecord(GenericModelRecord):
    """A CLIP model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.clip
    """Discriminator field identifying this as a CLIP model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.feature_extractor,
        )
    )

    pretrained_name: str | None = None
    """The pretrained model name, if applicable."""


@register_record_type(MODEL_REFERENCE_CATEGORY.codeformer)
class CodeformerModelRecord(GenericModelRecord):
    """A Codeformer model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.codeformer
    """Discriminator field identifying this as a Codeformer model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.post_processing,
        )
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.esrgan)
class EsrganModelRecord(GenericModelRecord):
    """An ESRGAN model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.esrgan
    """Discriminator field identifying this as an ESRGAN model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.post_processing,
        )
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.gfpgan)
class GfpganModelRecord(GenericModelRecord):
    """A GFPGAN model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.gfpgan
    """Discriminator field identifying this as a GFPGAN model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.post_processing,
        )
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.safety_checker)
class SafetyCheckerModelRecord(GenericModelRecord):
    """A safety checker model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.safety_checker
    """Discriminator field identifying this as a safety checker model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.safety_checker,
        )
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.video_generation)
class VideoGenerationModelRecord(GenericModelRecord):
    """A video generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.video_generation
    """Discriminator field identifying this as a video generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.video,
            purpose=MODEL_PURPOSE.generation,
        )
    )

    baseline: str | None = None
    """The model on which this model is based."""
    nsfw: bool = False
    """Whether the model is NSFW or not."""
    tags: list[str] | None = None
    """Any tags associated with the model which may be useful for searching."""


@register_record_type(MODEL_REFERENCE_CATEGORY.audio_generation)
class AudioGenerationModelRecord(GenericModelRecord):
    """An audio generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.audio_generation
    """Discriminator field identifying this as an audio generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.audio,
            purpose=MODEL_PURPOSE.generation,
        )
    )

    baseline: str | None = None
    """The model on which this model is based."""
    nsfw: bool = False
    """Whether the model is NSFW or not."""
    tags: list[str] | None = None
    """Any tags associated with the model which may be useful for searching."""


@register_record_type(MODEL_REFERENCE_CATEGORY.miscellaneous)
class MiscellaneousModelRecord(GenericModelRecord):
    """A miscellaneous model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous
    """Discriminator field identifying this as a miscellaneous model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.miscellaneous,
        )
    )


for category in MODEL_REFERENCE_CATEGORY:
    if category not in MODEL_RECORD_TYPE_LOOKUP:
        logger.trace(f"No record type registered for category {category}. Using GenericModelRecord.")
        MODEL_RECORD_TYPE_LOOKUP[category] = GenericModelRecord
