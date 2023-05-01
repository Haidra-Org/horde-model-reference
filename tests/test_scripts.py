from horde_model_reference.legacy.download_live_legacy_dbs import download_all_models
from horde_model_reference.legacy.validate_sd import validate_legacy_stable_diffusion_db
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES
from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER, get_model_reference_filename


def test_download_all_models():
    download_models = download_all_models(override_existing=True)
    assert len(download_models) == 8


def test_validate_stable_diffusion_model_reference():
    assert validate_legacy_stable_diffusion_db(
        sd_db=get_model_reference_filename(
            MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION,
            base_path=LEGACY_REFERENCE_FOLDER,
        ),
    )