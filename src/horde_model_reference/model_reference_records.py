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
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    SCHEMA_VERSION,
    ModelClassification,
    ai_horde_ci_settings,
)
from horde_model_reference.meta_consts import (
    CONTROLNET_STYLE,
    get_category_descriptor,
    is_known_controlnet_style,
    is_known_image_baseline,
    is_known_model_style,
)
from horde_model_reference.model_kind_validation import (
    FieldPolicy,
    KindPolicy,
    NormalizedModelStyle,
    NormalizedTag,
    category_key,
    kind_policy_registry,
)


def _classification_for(category: MODEL_REFERENCE_CATEGORY | str) -> ModelClassification:
    """Build the default ``ModelClassification`` for *category* from the registry."""
    desc = get_category_descriptor(category)
    return ModelClassification(domain=desc.domain, purpose=desc.purpose)


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

    record_type: str | MODEL_REFERENCE_CATEGORY
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

    @property
    def primary_download_url(self) -> str | None:
        """Return the URL of the first download entry, or None if there are no downloads."""
        if self.config and self.config.download:
            return self.config.download[0].file_url
        return None

    @property
    def all_download_urls(self) -> list[str]:
        """Return all download URLs for this model."""
        if self.config and self.config.download:
            return [d.file_url for d in self.config.download]
        return []

    @property
    def download_count(self) -> int:
        """Return the number of download entries for this model."""
        if self.config and self.config.download:
            return len(self.config.download)
        return 0


MODEL_RECORD_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORY | str, type[GenericModelRecord]] = {}


def register_record_type(
    category: MODEL_REFERENCE_CATEGORY | str,
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


_ERROR_POLICY = FieldPolicy(severity="error")
_WARNING_POLICY = FieldPolicy(severity="warning")


def _field_policy_for(
    category: MODEL_REFERENCE_CATEGORY | str,
    field_name: str,
    fallback: FieldPolicy,
) -> FieldPolicy:
    policy = kind_policy_registry.get(category_key(category))
    if policy is None:
        return fallback
    return policy.field_policies.get(field_name, fallback)


def _apply_policy(
    *,
    category: MODEL_REFERENCE_CATEGORY | str,
    field_name: str,
    value: str,
    fallback_policy: FieldPolicy,
    model_name: str,
) -> None:
    field_policy = _field_policy_for(category, field_name, fallback_policy)
    if field_policy.severity == "error":
        raise ValueError(f"Unknown {field_name}: {value}")

    logger.debug(f"Unknown {field_name} {value} for model {model_name}")


kind_policy_registry.register(
    category_key(MODEL_REFERENCE_CATEGORY.image_generation),
    KindPolicy(
        field_policies={
            "baseline": FieldPolicy(severity="error"),
            "style": FieldPolicy(severity="error"),
        },
    ),
)

kind_policy_registry.register(
    category_key(MODEL_REFERENCE_CATEGORY.controlnet),
    KindPolicy(
        field_policies={
            "controlnet_style": FieldPolicy(severity="warning"),
        },
    ),
)


@register_record_type(MODEL_REFERENCE_CATEGORY.image_generation)
class ImageGenerationModelRecord(GenericModelRecord):
    """A model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.image_generation
    """Discriminator field identifying this as an image generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.image_generation),
    )
    """The domain (e.g., image, text) and purpose (e.g., generation, classification) of the model."""

    inpainting: bool | None = False
    """If this is an inpainting model or not."""
    baseline: KNOWN_IMAGE_GENERATION_BASELINE | str
    """The model on which this model is based."""
    optimization: str | None = None
    """The optimization type of the model."""
    tags: list[NormalizedTag] | None = None
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

    style: NormalizedModelStyle | MODEL_STYLE | None = None
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
        if not is_known_image_baseline(str(self.baseline)):
            _apply_policy(
                category=self.record_type,
                field_name="baseline",
                value=str(self.baseline),
                fallback_policy=_ERROR_POLICY,
                model_name=self.name,
            )

        if self.style is not None and not is_known_model_style(str(self.style)):
            _apply_policy(
                category=self.record_type,
                field_name="style",
                value=str(self.style),
                fallback_policy=_ERROR_POLICY,
                model_name=self.name,
            )

        return self


@register_record_type(MODEL_REFERENCE_CATEGORY.controlnet)
class ControlNetModelRecord(GenericModelRecord):
    """A ControlNet model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.controlnet
    """Discriminator field identifying this as a ControlNet model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.controlnet),
    )

    controlnet_style: CONTROLNET_STYLE | str | None = None
    """The 'style' (purpose) of the controlnet. See `CONTROLNET_STYLE` for all possible values and more info."""

    @model_validator(mode="after")
    def validator_is_style_known(self) -> ControlNetModelRecord:
        """Check if the style is known."""
        if self.controlnet_style is not None and not is_known_controlnet_style(str(self.controlnet_style)):
            _apply_policy(
                category=self.record_type,
                field_name="controlnet_style",
                value=str(self.controlnet_style),
                fallback_policy=_WARNING_POLICY,
                model_name=self.name,
            )

        return self


@register_record_type(MODEL_REFERENCE_CATEGORY.text_generation)
class TextGenerationModelRecord(GenericModelRecord):
    """A text generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.text_generation
    """Discriminator field identifying this as a text generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.text_generation),
    )

    baseline: str | None = None
    parameters_count: int = Field(alias="parameters")
    nsfw: bool = False
    style: str | None = None
    display_name: str | None = None
    url: str | None = None
    tags: list[NormalizedTag] | None = None
    instruct_format: str | None = None
    """The instruction template format used by this model (e.g., ChatML, Mistral, Alpaca)."""
    settings: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None
    text_model_group: str | None = None
    """The base model group name for grouping model variants together."""
    name_schema_exception: str | None = None
    """If set, this model does not follow the group's naming schema. The value is the reason."""


class TextModelGroupNameSchema(BaseModel):
    """Persisted naming convention for a text model group.

    When saved, overrides the inferred schema from ``infer_name_format()``.
    """

    model_config = ConfigDict(extra="forbid")

    separator: str = "-"
    part_order: list[str] = Field(default_factory=lambda: ["base", "size", "variant", "version", "quant"])
    author_included: bool = True
    common_author: str | None = None
    template: str | None = None
    extra_parts: list[str] = Field(default_factory=list)
    """Free-form labels for name segments outside the five primary categories (e.g., ``["date"]``)."""


@register_record_type(MODEL_REFERENCE_CATEGORY.blip)
class BlipModelRecord(GenericModelRecord):
    """A BLIP model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.blip
    """Discriminator field identifying this as a BLIP model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.blip),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.clip)
class ClipModelRecord(GenericModelRecord):
    """A CLIP model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.clip
    """Discriminator field identifying this as a CLIP model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.clip),
    )

    pretrained_name: str | None = None
    """The pretrained model name, if applicable."""


@register_record_type(MODEL_REFERENCE_CATEGORY.codeformer)
class CodeformerModelRecord(GenericModelRecord):
    """A Codeformer model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.codeformer
    """Discriminator field identifying this as a Codeformer model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.codeformer),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.esrgan)
class EsrganModelRecord(GenericModelRecord):
    """An ESRGAN model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.esrgan
    """Discriminator field identifying this as an ESRGAN model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.esrgan),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.gfpgan)
class GfpganModelRecord(GenericModelRecord):
    """A GFPGAN model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.gfpgan
    """Discriminator field identifying this as a GFPGAN model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.gfpgan),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.safety_checker)
class SafetyCheckerModelRecord(GenericModelRecord):
    """A safety checker model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.safety_checker
    """Discriminator field identifying this as a safety checker model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.safety_checker),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.video_generation)
class VideoGenerationModelRecord(GenericModelRecord):
    """A video generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.video_generation
    """Discriminator field identifying this as a video generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.video_generation),
    )

    baseline: str | None = None
    """The model on which this model is based."""
    nsfw: bool = False
    """Whether the model is NSFW or not."""
    tags: list[NormalizedTag] | None = None
    """Any tags associated with the model which may be useful for searching."""


@register_record_type(MODEL_REFERENCE_CATEGORY.audio_generation)
class AudioGenerationModelRecord(GenericModelRecord):
    """An audio generation model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.audio_generation
    """Discriminator field identifying this as an audio generation model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.audio_generation),
    )

    baseline: str | None = None
    """The model on which this model is based."""
    nsfw: bool = False
    """Whether the model is NSFW or not."""
    tags: list[NormalizedTag] | None = None
    """Any tags associated with the model which may be useful for searching."""


@register_record_type(MODEL_REFERENCE_CATEGORY.miscellaneous)
class MiscellaneousModelRecord(GenericModelRecord):
    """A miscellaneous model entry in the model reference."""

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.miscellaneous
    """Discriminator field identifying this as a miscellaneous model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.miscellaneous),
    )


@register_record_type(MODEL_REFERENCE_CATEGORY.lora)
class LoraModelRecord(GenericModelRecord):
    """A LoRA model entry in the model reference.

    LoRA records are ``managed_elsewhere`` in canonical data: the horde itself does
    not host them, so they are typically supplied by third-party providers (see
    :mod:`horde_model_reference.providers`). This built-in type is provided as a
    convenient default; providers may subclass and register their own type instead.
    """

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.lora
    """Discriminator field identifying this as a LoRA model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.lora),
    )

    baseline: str | None = None
    """The baseline (e.g., ``stable_diffusion_xl``) this LoRA is intended for."""
    nsfw: bool = False
    """Whether the LoRA is NSFW or not."""
    tags: list[NormalizedTag] | None = None
    """Any tags associated with the LoRA which may be useful for searching."""
    trigger: list[str] | None = None
    """A list of trigger words or phrases which activate the LoRA."""
    homepage: str | None = None
    """A link to the LoRA's homepage."""
    showcases: list[str] | None = None
    """Links to any showcases illustrating the LoRA's effect."""


@register_record_type(MODEL_REFERENCE_CATEGORY.ti)
class TextualInversionModelRecord(GenericModelRecord):
    """A textual inversion (embedding) model entry in the model reference.

    Like :class:`LoraModelRecord`, textual inversions are ``managed_elsewhere`` and
    are normally supplied by third-party providers.
    """

    record_type: MODEL_REFERENCE_CATEGORY | str = MODEL_REFERENCE_CATEGORY.ti
    """Discriminator field identifying this as a textual inversion model record."""

    model_classification: ModelClassification = Field(
        default_factory=lambda: _classification_for(MODEL_REFERENCE_CATEGORY.ti),
    )

    baseline: str | None = None
    """The baseline (e.g., ``stable_diffusion_1``) this embedding is intended for."""
    nsfw: bool = False
    """Whether the embedding is NSFW or not."""
    tags: list[NormalizedTag] | None = None
    """Any tags associated with the embedding which may be useful for searching."""
    trigger: list[str] | None = None
    """A list of trigger words or phrases which activate the embedding."""
    homepage: str | None = None
    """A link to the embedding's homepage."""
    showcases: list[str] | None = None
    """Links to any showcases illustrating the embedding's effect."""


for category in MODEL_REFERENCE_CATEGORY:
    if category not in MODEL_RECORD_TYPE_LOOKUP:
        logger.trace(f"No record type registered for category {category}. Using GenericModelRecord.")
        MODEL_RECORD_TYPE_LOOKUP[category] = GenericModelRecord


def get_record_type_for_category(category: MODEL_REFERENCE_CATEGORY | str) -> type[GenericModelRecord]:
    """Return the registered record type for *category*, falling back to ``GenericModelRecord``.

    Args:
        category: The model reference category (enum member or plain string).

    Returns:
        The record type class registered for the category, or ``GenericModelRecord``
        if no specific type has been registered.

    """
    return MODEL_RECORD_TYPE_LOOKUP.get(category, GenericModelRecord)


__all__ = [
    "MODEL_RECORD_TYPE_LOOKUP",
    "AudioGenerationModelRecord",
    "BlipModelRecord",
    "ClipModelRecord",
    "CodeformerModelRecord",
    "ControlNetModelRecord",
    "DownloadRecord",
    "EsrganModelRecord",
    "FineTuneSeriesInfo",
    "GenericModelRecord",
    "GenericModelRecordConfig",
    "GenericModelRecordMetadata",
    "GfpganModelRecord",
    "ImageGenerationModelRecord",
    "LoraModelRecord",
    "MiscellaneousModelRecord",
    "SafetyCheckerModelRecord",
    "TextGenerationModelRecord",
    "TextualInversionModelRecord",
    "VideoGenerationModelRecord",
    "get_record_type_for_category",
    "register_record_type",
]
