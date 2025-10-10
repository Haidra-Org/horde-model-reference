"""Test that the refactored converters use the new Pydantic models from legacy_models.py."""

from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyStableDiffusionConverter,
    LegacyTextGenerationConverter,
)
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyClipRecord,
    LegacyGenericRecord,
    LegacyStableDiffusionRecord,
    LegacyTextGenerationRecord,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


def test_stable_diffusion_converter_uses_legacy_models() -> None:
    """Test that SD converter uses the new LegacyStableDiffusionRecord."""
    sd_converter = LegacyStableDiffusionConverter()
    assert sd_converter.model_reference_type == LegacyStableDiffusionRecord

    converted = sd_converter.convert_to_new_format()
    assert len(converted) > 0


def test_clip_converter_uses_legacy_models() -> None:
    """Test that CLIP converter uses the new LegacyClipRecord."""
    clip_converter = LegacyClipConverter()
    assert clip_converter.model_reference_type == LegacyClipRecord

    converted = clip_converter.convert_to_new_format()
    assert len(converted) > 0


def test_text_generation_converter_uses_legacy_models() -> None:
    """Test that text generation converter uses the new LegacyTextGenerationRecord."""
    text_gen_converter = LegacyTextGenerationConverter()
    assert text_gen_converter.model_reference_type == LegacyTextGenerationRecord

    converted = text_gen_converter.convert_to_new_format()
    assert len(converted) > 0


def test_base_converter_uses_legacy_generic_model() -> None:
    """Test that base converter uses LegacyGenericRecord."""
    base_converter = BaseLegacyConverter(
        model_reference_category=MODEL_REFERENCE_CATEGORY.miscellaneous,
    )
    assert base_converter.model_reference_type == LegacyGenericRecord


def test_converters_track_validation_issues() -> None:
    """Test that converters collect validation issues from Pydantic models."""
    sd_converter = LegacyStableDiffusionConverter(debug_mode=True)
    sd_converter.convert_to_new_format()

    assert isinstance(sd_converter.all_validation_errors_log, dict), "Validation errors log should be a dictionary"


def test_converted_records_have_correct_type() -> None:
    """Test that converted records are of the correct new format type."""
    sd_converter = LegacyStableDiffusionConverter()
    sd_converter.convert_to_new_format()

    assert len(sd_converter._all_legacy_records) > 0
    assert len(sd_converter._all_converted_records) > 0

    for _, legacy_record in sd_converter._all_legacy_records.items():
        assert isinstance(legacy_record, LegacyStableDiffusionRecord)

    from horde_model_reference.model_reference_records import ImageGenerationModelRecord

    for _, record in sd_converter._all_converted_records.items():
        assert isinstance(record, ImageGenerationModelRecord)
