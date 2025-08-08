from horde_model_reference.meta_consts import MODEL_CLASSIFICATION_LOOKUP, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager


def test_manager_legacy(model_reference_manager: ModelReferenceManager) -> None:
    """Basic test of the legacy model reference manager."""
    legacy_reference_locations = (
        model_reference_manager.legacy_reference_download_manager.get_all_legacy_model_references()
    )

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations


def test_manager_new_format(model_reference_manager: ModelReferenceManager) -> None:
    """Test the new format model reference manager."""
    all_model_references = model_reference_manager.get_all_model_references()

    assert len(all_model_references) > 0

    for model_reference_category in MODEL_REFERENCE_CATEGORY:
        assert model_reference_category in all_model_references
        model_reference_instance = all_model_references[model_reference_category]
        assert model_reference_instance is not None
        for _, model_entry in model_reference_instance:
            assert model_entry.model_classification == MODEL_CLASSIFICATION_LOOKUP[model_reference_category]
