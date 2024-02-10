import json
import urllib.parse
from pathlib import Path

import pytest

import horde_model_reference.path_consts as path_consts
from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyStableDiffusionConverter,
)
from horde_model_reference.legacy.classes.staging_model_database_records import (
    MODEL_REFERENCE_LEGACY_TYPE_LOOKUP,
    StagingLegacy_Generic_ModelRecord,
)
from horde_model_reference.model_reference_records import StableDiffusion_ModelRecord, StableDiffusion_ModelReference


def test_convert_legacy_stable_diffusion_database(base_path_for_tests: Path, legacy_folder_for_tests: Path):
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=legacy_folder_for_tests,
        target_file_folder=base_path_for_tests,
    )
    assert sd_converter.normalize_and_convert()


def test_convert_legacy_clip_database(base_path_for_tests: Path, legacy_folder_for_tests: Path):
    clip_converter = LegacyClipConverter(
        legacy_folder_path=legacy_folder_for_tests,
        target_file_folder=base_path_for_tests,
    )
    assert clip_converter.normalize_and_convert()


def test_all_base_legacy_converters(base_path_for_tests: Path, legacy_folder_for_tests: Path):
    generic_references = {
        k: v for k, v in MODEL_REFERENCE_LEGACY_TYPE_LOOKUP.items() if v is StagingLegacy_Generic_ModelRecord
    }
    for reference_category in generic_references:
        base_converter = BaseLegacyConverter(
            legacy_folder_path=legacy_folder_for_tests,
            target_file_folder=base_path_for_tests,
            model_reference_category=reference_category,
        )
        assert base_converter.normalize_and_convert()


def test_validate_converted_stable_diffusion_database(base_path_for_tests) -> None:
    stable_diffusion_model_database_path = path_consts.get_model_reference_file_path(
        path_consts.MODEL_REFERENCE_CATEGORY.stable_diffusion,
        base_path=base_path_for_tests,
    )
    sd_model_database_file_contents: str = ""
    with open(stable_diffusion_model_database_path) as f:
        sd_model_database_file_contents = f.read()

    sd_model_database_file_json = json.loads(sd_model_database_file_contents)
    model_reference = StableDiffusion_ModelReference(
        root={k: StableDiffusion_ModelRecord.model_validate(v) for k, v in sd_model_database_file_json.items()},
    )

    assert len(model_reference.baseline) >= 3
    for baseline_type in model_reference.baseline:
        assert baseline_type != ""

    assert len(model_reference.styles) >= 6
    for style in model_reference.styles:
        assert style != ""

    assert len(model_reference.models_hosts) >= 1
    for model_host in model_reference.models_hosts:
        assert model_host != ""

    assert model_reference.root is not None
    assert len(model_reference.root) >= 100

    assert model_reference.root["stable_diffusion"] is not None
    assert model_reference.root["stable_diffusion"].name == "stable_diffusion"
    assert model_reference.root["stable_diffusion"].showcases is not None
    # assert len(model_reference.models["stable_diffusion"].showcases) >= 3
    # XXX This is commented out because the dynamic showcase generation should not currently be part of the CI

    for model_key, model_info in model_reference.root.items():
        assert model_info.name == model_key
        assert model_info.baseline in model_reference.baseline
        assert model_info.style in model_reference.styles

        if model_info.homepage is not None:
            assert model_info.homepage != ""
            parsedHomepage = urllib.parse.urlparse(model_info.homepage)
            assert parsedHomepage.scheme == "https" or parsedHomepage.scheme == "http"

        assert model_info.description is not None
        assert model_info.description != ""
        assert model_info.version != ""

        if model_info.tags is not None:
            for tag in model_info.tags:
                assert tag in model_reference.tags

        for config_key, config_section in model_info.config.items():
            assert config_key != "files"

            if config_key == "download":
                for download_record in config_section:
                    assert download_record.file_name is not None
                    assert download_record.file_url is not None
            else:
                pytest.fail(f"Unknown config section: {config_key}")

        if model_info.trigger is not None:
            for trigger_record in model_info.trigger:
                assert trigger_record != ""
