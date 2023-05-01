import urllib.parse
from pathlib import Path

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
from horde_model_reference.model_reference_records import (
    StableDiffusion_ModelReference,
)

TARGET_DIRECTORY_FOR_TESTDATA = Path(__file__).parent.joinpath("test_data_results")
"""The path to the converted stable diffusion model reference."""


def test_convert_legacy_stablediffusion_database():
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=path_consts.LEGACY_REFERENCE_FOLDER,
        target_file_folder=TARGET_DIRECTORY_FOR_TESTDATA,
    )
    assert sd_converter.normalize_and_convert()


def test_convert_legacy_clip_database():
    clip_converter = LegacyClipConverter(
        legacy_folder_path=path_consts.LEGACY_REFERENCE_FOLDER,
        target_file_folder=TARGET_DIRECTORY_FOR_TESTDATA,
    )
    assert clip_converter.normalize_and_convert()


def test_all_base_legacy_converters():
    generic_references = {
        k: v for k, v in MODEL_REFERENCE_LEGACY_TYPE_LOOKUP.items() if v is StagingLegacy_Generic_ModelRecord
    }
    for reference_category in generic_references:
        base_converter = BaseLegacyConverter(
            legacy_folder_path=path_consts.LEGACY_REFERENCE_FOLDER,
            target_file_folder=TARGET_DIRECTORY_FOR_TESTDATA,
            model_reference_category=reference_category,
        )
        assert base_converter.normalize_and_convert()


def test_validate_converted_stablediffusion_database():

    stablediffusion_model_database_path = path_consts.get_model_reference_filename(
        path_consts.MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION,
        base_path=TARGET_DIRECTORY_FOR_TESTDATA,
    )

    model_reference = StableDiffusion_ModelReference.parse_file(stablediffusion_model_database_path)

    assert len(model_reference.baseline) >= 3
    for baseline_type in model_reference.baseline:
        assert baseline_type != ""

    assert len(model_reference.styles) >= 6
    for style in model_reference.styles:
        assert style != ""

    assert len(model_reference.model_hosts) >= 1
    for model_host in model_reference.model_hosts:
        assert model_host != ""

    assert model_reference.models is not None
    assert len(model_reference.models) >= 100

    assert model_reference.models["stable_diffusion"] is not None
    assert model_reference.models["stable_diffusion"].name == "stable_diffusion"
    assert model_reference.models["stable_diffusion"].showcases is not None
    assert len(model_reference.models["stable_diffusion"].showcases) >= 3

    for model_key, model_info in model_reference.models.items():

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
                assert False

        if model_info.trigger is not None:
            for trigger_record in model_info.trigger:
                assert trigger_record != ""
