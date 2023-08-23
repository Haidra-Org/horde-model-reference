from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager


def test_manager_init():
    ModelReferenceManager()


def test_manager_legacy():
    model_reference_manager = ModelReferenceManager()

    legacy_reference_locations = model_reference_manager.get_all_legacy_model_reference_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.stable_diffusion in legacy_reference_locations


def test_manager_new_format():
    model_reference_manager = ModelReferenceManager()

    all_model_references = model_reference_manager.get_all_model_references()

    assert len(all_model_references) > 0

    for model_reference_category in MODEL_REFERENCE_CATEGORY:
        assert model_reference_category in all_model_references
