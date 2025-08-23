from pytest import LogCaptureFixture

from horde_model_reference.meta_consts import MODEL_CLASSIFICATION_LOOKUP, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager


def test_manager_legacy(model_reference_manager: ModelReferenceManager, caplog: LogCaptureFixture) -> None:
    """Basic test of the legacy model reference manager."""
    legacy_reference_locations = (
        model_reference_manager.legacy_reference_download_manager.get_all_legacy_model_references_paths()
    )

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations

    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    legacy_references = model_reference_manager.legacy_reference_download_manager.get_all_legacy_model_references(
        redownload_all=True,
    )
    assert "cached" not in caplog.records[-1].message
    assert len(legacy_references) > 0
    assert all(ref_cat in legacy_references for ref_cat in MODEL_REFERENCE_CATEGORY)

    assert model_reference_manager.legacy_reference_download_manager._references_cache.currsize == 1
    assert len(model_reference_manager.legacy_reference_download_manager._references_paths_cache) == len(
        MODEL_REFERENCE_CATEGORY,
    )

    legacy_references = model_reference_manager.legacy_reference_download_manager.get_all_legacy_model_references()
    assert "cached" in caplog.records[-1].message
    assert len(legacy_references) > 0
    assert all(ref_cat in legacy_references for ref_cat in MODEL_REFERENCE_CATEGORY)


def test_manager_new_format(model_reference_manager: ModelReferenceManager, caplog: LogCaptureFixture) -> None:
    """Test the new format model reference manager."""

    def assert_all_model_references_exist(
        model_reference_manager: ModelReferenceManager,
        override_existing: bool,
    ) -> None:
        """Assert that all model references exist."""
        all_model_references = model_reference_manager.get_all_model_references(override_existing=override_existing)
        for model_reference_category in MODEL_REFERENCE_CATEGORY:
            assert (
                model_reference_category in all_model_references
            ), f"Model reference category {model_reference_category} is missing"
            model_reference_instance = all_model_references[model_reference_category]
            assert (
                model_reference_instance is not None
            ), f"Model reference instance for {model_reference_category} is None"
            for _, model_entry in model_reference_instance:
                assert (
                    model_entry.model_classification == MODEL_CLASSIFICATION_LOOKUP[model_reference_category]
                ), f"Model entry for {model_reference_category} is not classified correctly"

    assert_all_model_references_exist(model_reference_manager, override_existing=True)

    assert_all_model_references_exist(model_reference_manager, override_existing=False)
