import json
from pathlib import Path
from typing import Any

from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyControlnetConverter,
    LegacyStableDiffusionConverter,
    LegacyTextGenerationConverter,
)
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_legacy_text_generation_database
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


def test_convert_legacy_stable_diffusion_database(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test converting the legacy stable diffusion database to the new format."""
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
        target_file_folder=primary_base,
    )
    converted = sd_converter.convert_to_new_format()
    assert len(converted) > 0


def test_convert_legacy_clip_database(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test converting the legacy clip database to the new format."""
    clip_converter = LegacyClipConverter(
        legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
        target_file_folder=primary_base,
    )
    converted_clip = clip_converter.convert_to_new_format()
    assert len(converted_clip) > 0


def test_convert_legacy_controlnet_database(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test converting the legacy controlnet database to the new format."""
    controlnet_converter = LegacyControlnetConverter(
        legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
        target_file_folder=primary_base,
    )
    converted = controlnet_converter.convert_to_new_format()
    assert len(converted) > 0


def test_convert_legacy_text_generation_database(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test converting the legacy text generation database to the new format."""
    text_gen_converter = LegacyTextGenerationConverter(
        legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
        target_file_folder=primary_base,
    )
    converted = text_gen_converter.convert_to_new_format()
    assert len(converted) > 0


def test_convert_helper_accepts_legacy_directory(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Ensure helper functions tolerate paths pointing at the legacy/ folder itself."""

    result = convert_legacy_text_generation_database(
        legacy_path=populated_legacy_path,
        target_path=primary_base,
    )

    assert result is True
    output = primary_base / "text_generation.json"
    assert output.exists()


def test_all_base_legacy_converters(
    primary_base: Path,
    legacy_path: Path,
    minimal_legacy_generic_data: dict[str, Any],
) -> None:
    """Test converting all legacy databases using the base converter."""
    for reference_category in MODEL_REFERENCE_CATEGORY:
        if reference_category in [
            MODEL_REFERENCE_CATEGORY.image_generation,
            MODEL_REFERENCE_CATEGORY.clip,
            MODEL_REFERENCE_CATEGORY.controlnet,
            MODEL_REFERENCE_CATEGORY.text_generation,
        ]:
            continue

        # Create a minimal JSON file for this category
        category_file = legacy_path / f"{reference_category.value}.json"
        category_file.write_text(json.dumps(minimal_legacy_generic_data, indent=2))

        base_converter = BaseLegacyConverter(
            model_reference_category=reference_category,
            legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
            target_file_folder=primary_base,
        )
        converted = base_converter.convert_to_new_format()
        assert len(converted) > 0


def test_validate_converted_stable_diffusion_database(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test validating the converted stable diffusion database."""
    # First convert the database
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=primary_base,  # Pass base path, converter adds /legacy/
        target_file_folder=primary_base,
    )
    sd_converter.convert_to_new_format()

    stable_diffusion_model_database_path = primary_base / "stable_diffusion.json"
    assert stable_diffusion_model_database_path.exists(), "Converted database should exist"

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
