from horde_model_reference.model_database_records import (
    StableDiffusion_ModelReference as New_StableDiffusionModelReference,
)

from horde_model_reference.legacy.convert_legacy import LegacyStableDiffusionConverter
from pathlib import Path
import urllib.parse

import horde_model_reference.consts as consts

TARGET_DIRECTORY_FOR_TESTDATA = Path(__file__).parent.joinpath("test_data_results")
"""The path to the converted stable diffusion model reference."""


def test_convert_legacy_stablediffusion_database():
    converter = LegacyStableDiffusionConverter(
        legacy_folder_path=consts.LEGACY_REFERENCE_FOLDER,
        target_file_folder=TARGET_DIRECTORY_FOR_TESTDATA,
    )
    assert converter.normalize_and_convert()


def test_validate_converted_stablediffusion_database():

    stablediffusion_model_database_path = consts.get_model_reference_filename(
        consts.MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION,
        basePath=TARGET_DIRECTORY_FOR_TESTDATA,
    )

    model_reference = New_StableDiffusionModelReference.parse_file(stablediffusion_model_database_path)

    assert len(model_reference.baseline_categories) >= 3
    for baseline_type in model_reference.baseline_categories:
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
        assert model_info.baseline in model_reference.baseline_categories
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
