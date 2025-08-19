import os
from pathlib import Path

from horde_model_reference.legacy import LegacyReferenceDownloadManager
from horde_model_reference.legacy.validate_sd import validate_legacy_stable_diffusion_db
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import get_model_reference_file_path


def test_download_all_model_references(legacy_reference_download_manager: LegacyReferenceDownloadManager) -> None:
    download_models = legacy_reference_download_manager.download_all_legacy_model_references(overwrite_existing=True)
    assert len(download_models) == len(MODEL_REFERENCE_CATEGORY.__members__)


def test_validate_stable_diffusion_model_reference(legacy_folder_for_tests: Path) -> None:
    if os.environ.get("HORDELIB_CI_ONGOING"):
        assert validate_legacy_stable_diffusion_db(
            sd_db=get_model_reference_file_path(
                MODEL_REFERENCE_CATEGORY.image_generation,
                base_path=legacy_folder_for_tests,
            ),
            fail_on_extra=True,
        )
    else:
        assert validate_legacy_stable_diffusion_db(
            sd_db=get_model_reference_file_path(
                MODEL_REFERENCE_CATEGORY.image_generation,
                base_path=legacy_folder_for_tests,
            ),
            fail_on_extra=False,
        )
