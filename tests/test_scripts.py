import os

from horde_model_reference import ModelReferenceManager, horde_model_reference_paths
from horde_model_reference.legacy.validate_sd import validate_legacy_stable_diffusion_db
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


def test_download_all_model_references(model_reference_manager: ModelReferenceManager) -> None:
    """Test downloading all model references script."""
    download_models = model_reference_manager.get_all_model_references(overwrite_existing=True)
    assert len(download_models) == len(MODEL_REFERENCE_CATEGORY.__members__)


def test_validate_stable_diffusion_model_reference() -> None:
    """Test validating the validate stable diffusion model reference."""
    if os.environ.get("HORDELIB_CI_ONGOING"):
        assert validate_legacy_stable_diffusion_db(
            sd_db=horde_model_reference_paths.get_legacy_model_reference_file_path(
                MODEL_REFERENCE_CATEGORY.image_generation,
            ),
            fail_on_extra=True,
        )
    else:
        assert validate_legacy_stable_diffusion_db(
            sd_db=horde_model_reference_paths.get_legacy_model_reference_file_path(
                MODEL_REFERENCE_CATEGORY.image_generation,
            ),
            fail_on_extra=False,
        )
