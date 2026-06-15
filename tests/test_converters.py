"""Test that the refactored converters use the new Pydantic models from legacy_models.py."""

from pathlib import Path

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


def test_stable_diffusion_converter_uses_legacy_models(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test that SD converter uses the new LegacyStableDiffusionRecord."""
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=populated_legacy_path,
        target_file_folder=primary_base,
    )
    assert sd_converter.model_reference_type == LegacyStableDiffusionRecord

    converted = sd_converter.convert_to_new_format()
    assert len(converted) > 0


def test_clip_converter_uses_legacy_models(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test that CLIP converter uses the new LegacyClipRecord."""
    clip_converter = LegacyClipConverter(
        legacy_folder_path=populated_legacy_path,
        target_file_folder=primary_base,
    )
    assert clip_converter.model_reference_type == LegacyClipRecord

    converted = clip_converter.convert_to_new_format()
    assert len(converted) > 0


def test_text_generation_converter_uses_legacy_models(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test that text generation converter uses the new LegacyTextGenerationRecord."""
    text_gen_converter = LegacyTextGenerationConverter(
        legacy_folder_path=populated_legacy_path,
        target_file_folder=primary_base,
    )
    assert text_gen_converter.model_reference_type == LegacyTextGenerationRecord

    converted = text_gen_converter.convert_to_new_format()
    assert len(converted) > 0


def test_base_converter_uses_legacy_generic_model(
    primary_base: Path,
    legacy_path: Path,
) -> None:
    """Test that base converter uses LegacyGenericRecord."""
    base_converter = BaseLegacyConverter(
        model_reference_category=MODEL_REFERENCE_CATEGORY.miscellaneous,
        legacy_folder_path=legacy_path,
        target_file_folder=primary_base,
    )
    assert base_converter.model_reference_type == LegacyGenericRecord


def test_converters_track_validation_issues(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test that converters collect validation issues from Pydantic models."""
    sd_converter = LegacyStableDiffusionConverter(
        debug_mode=True,
        legacy_folder_path=populated_legacy_path,
        target_file_folder=primary_base,
    )
    sd_converter.convert_to_new_format()

    assert isinstance(sd_converter.all_validation_errors_log, dict), "Validation errors log should be a dictionary"


def test_converted_records_have_correct_type(
    primary_base: Path,
    populated_legacy_path: Path,
) -> None:
    """Test that converted records are of the correct new format type."""
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=populated_legacy_path,
        target_file_folder=primary_base,
    )
    sd_converter.convert_to_new_format()

    assert len(sd_converter._all_legacy_records) > 0
    assert len(sd_converter._all_converted_records) > 0

    for _, legacy_record in sd_converter._all_legacy_records.items():
        assert isinstance(legacy_record, LegacyStableDiffusionRecord)

    from horde_model_reference.model_reference_records import ImageGenerationModelRecord

    for _, record in sd_converter._all_converted_records.items():
        assert isinstance(record, ImageGenerationModelRecord)


def test_legacy_sd_record_serialization_drops_none_config_keys() -> None:
    """A round-tripped SD record must match the hand-authored legacy JSON byte-for-byte.

    The legacy GitHub format omits absent optional config-file fields (``md5sum``,
    ``file_type``). Pydantic's default nested serialization emits them as ``null``, which
    pollutes the diff produced by the GitHub sync for newly added/modified models. The
    custom serializers on the config models must drop ``None`` so API-written records stay
    byte-compatible with the upstream legacy files.
    """
    entry = {
        "name": "PilotModel XL",
        "baseline": "stable_diffusion_xl",
        "type": "ckpt",
        "inpainting": False,
        "description": "Pilot model for sync fidelity.",
        "version": "1.0",
        "style": "realistic",
        "homepage": "https://example.com/pilot",
        "nsfw": True,
        "download_all": False,
        "config": {
            "files": [{"path": "pilot_v1.safetensors", "sha256sum": "a" * 64}],
            "download": [
                {
                    "file_name": "pilot_v1.safetensors",
                    "file_path": "",
                    "file_url": "https://example.com/pilot_v1.safetensors",
                }
            ],
        },
        "features_not_supported": ["hires_fix"],
        "size_on_disk_bytes": 123,
    }

    dumped = LegacyStableDiffusionRecord.model_validate(entry).model_dump(mode="json")

    assert dumped == entry, "round-trip must reproduce the hand-authored legacy entry exactly"
    file_entry = dumped["config"]["files"][0]
    assert "md5sum" not in file_entry
    assert "file_type" not in file_entry
    assert set(file_entry) == {"path", "sha256sum"}


def test_legacy_sd_record_serialization_keeps_present_config_keys() -> None:
    """Dropping ``None`` must not remove config-file fields that are actually populated."""
    entry = {
        "name": "PilotModel",
        "baseline": "stable diffusion 1",
        "type": "ckpt",
        "inpainting": False,
        "description": "Pilot ckpt with both checksums.",
        "version": "1.0",
        "style": "generalist",
        "nsfw": False,
        "download_all": True,
        "config": {
            "files": [{"path": "pilot.ckpt", "md5sum": "b" * 32, "sha256sum": "c" * 64}],
            "download": [
                {
                    "file_name": "pilot.ckpt",
                    "file_path": "",
                    "file_url": "https://example.com/pilot.ckpt",
                }
            ],
        },
    }

    dumped = LegacyStableDiffusionRecord.model_validate(entry).model_dump(mode="json")

    file_entry = dumped["config"]["files"][0]
    assert file_entry["md5sum"] == "b" * 32
    assert file_entry["sha256sum"] == "c" * 64
    assert "file_type" not in file_entry
