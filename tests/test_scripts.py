from pathlib import Path

from horde_model_reference.legacy.download_live_legacy_dbs import LegacyReferenceDownloadManager
from horde_model_reference.legacy.validate_sd import validate_legacy_stable_diffusion_db
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import get_model_reference_file_path


def test_download_all_model_references(base_path_for_tests: Path):
    reference_download_manager = LegacyReferenceDownloadManager(base_path=base_path_for_tests)
    download_models = reference_download_manager.download_all_legacy_model_references(overwrite_existing=True)
    assert len(download_models) == 8


def test_validate_stable_diffusion_model_reference(legacy_folder_for_tests: Path):
    assert validate_legacy_stable_diffusion_db(
        sd_db=get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.stable_diffusion,
            base_path=legacy_folder_for_tests,
        ),
    )
