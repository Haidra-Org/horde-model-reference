import json
from typing import Any

from horde_model_reference import horde_model_reference_paths
from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyControlnetConverter,
    LegacyStableDiffusionConverter,
    LegacyTextGenerationConverter,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


def test_convert_legacy_stable_diffusion_database() -> None:
    """Test converting the legacy stable diffusion database to the new format."""
    sd_converter = LegacyStableDiffusionConverter()
    assert sd_converter.convert_to_new_format()


def test_convert_legacy_clip_database() -> None:
    """Test converting the legacy clip database to the new format."""
    clip_converter = LegacyClipConverter()
    converted_clip = clip_converter.convert_to_new_format()
    assert converted_clip


def test_convert_legacy_controlnet_database() -> None:
    """Test converting the legacy controlnet database to the new format."""
    controlnet_converter = LegacyControlnetConverter()
    assert controlnet_converter.convert_to_new_format()


def test_convert_legacy_text_generation_database() -> None:
    """Test converting the legacy text generation database to the new format."""
    text_gen_converter = LegacyTextGenerationConverter()
    assert text_gen_converter.convert_to_new_format()


def test_all_base_legacy_converters() -> None:
    """Test converting all legacy databases using the base converter."""
    for reference_category in MODEL_REFERENCE_CATEGORY:
        if reference_category in [
            MODEL_REFERENCE_CATEGORY.image_generation,
            MODEL_REFERENCE_CATEGORY.clip,
            MODEL_REFERENCE_CATEGORY.controlnet,
            MODEL_REFERENCE_CATEGORY.text_generation,
        ]:
            continue
        base_converter = BaseLegacyConverter(
            model_reference_category=reference_category,
        )
        assert base_converter.convert_to_new_format()


def test_validate_converted_stable_diffusion_database() -> None:
    """Test validating the converted stable diffusion database."""
    stable_diffusion_model_database_path = horde_model_reference_paths.get_model_reference_file_path(
        MODEL_REFERENCE_CATEGORY.image_generation,
    )
    sd_model_database_file_contents: str = ""
    with open(stable_diffusion_model_database_path) as f:
        sd_model_database_file_contents = f.read()

    sd_model_database_file_json: dict[str, Any] = json.loads(sd_model_database_file_contents)
    model_reference: dict[str, ImageGenerationModelRecord] = {
        k: ImageGenerationModelRecord.model_validate(v) for k, v in sd_model_database_file_json.items()
    }

    baseline_set = set()
    styles_set = set()
    tags_set = set()
    model_hosts_set = set()

    for model_key, model_info in model_reference.items():
        assert isinstance(model_info, ImageGenerationModelRecord)
        baseline_set.add(model_info.baseline)
        styles_set.add(model_info.style)
        if model_info.tags is not None:
            for tag in model_info.tags:
                tags_set.add(tag)

        for download_record in model_info.config.download:
            model_hosts_set.add(download_record.file_url)

        assert model_key == model_info.name
        assert model_info.description is not None
        assert model_info.description != ""
        assert model_info.version != ""

    assert len(baseline_set) >= 3
    assert len(styles_set) >= 6
    assert len(tags_set) >= 11
    assert len(model_hosts_set) >= 1
