import aiohttp
import pytest
from loguru import logger
from pydantic import ValidationError
from pytest import LogCaptureFixture

from horde_model_reference.meta_consts import (
    MODEL_CLASSIFICATION_LOOKUP,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.model_reference_records import GenericModelRecord


def test_manager_legacy(model_reference_manager: ModelReferenceManager, caplog: LogCaptureFixture) -> None:
    """Basic test of the legacy model reference manager."""
    model_reference_manager._instance = None  # Reset singleton for testing
    ModelReferenceManager()

    # Get legacy file paths from backend
    legacy_reference_locations = model_reference_manager.backend.get_all_category_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations

    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    # Force fetch all categories
    model_reference_manager.backend.fetch_all_categories(force_refresh=True)
    assert "Building" in caplog.records[-1].message

    # Verify backend has cached data
    legacy_references = {}
    for cat in MODEL_REFERENCE_CATEGORY:
        data = model_reference_manager.backend.fetch_category(cat)
        if data is not None:
            legacy_references[cat] = data

    assert len(legacy_references) > 0
    assert all(ref_cat in legacy_references for ref_cat in MODEL_REFERENCE_CATEGORY)

    # Access internal backend cache to verify caching
    from horde_model_reference.backends import LegacyGitHubBackend

    assert isinstance(model_reference_manager.backend, LegacyGitHubBackend)
    assert model_reference_manager.backend._converted_cache is not None
    assert len(model_reference_manager.backend._references_paths_cache) == len(MODEL_REFERENCE_CATEGORY)

    # Test cached access - backend should return cached data
    legacy_references_cached = {}
    for cat in MODEL_REFERENCE_CATEGORY:
        data = model_reference_manager.backend.fetch_category(cat)
        if data is not None:
            legacy_references_cached[cat] = data

    # Verify the cache is being used (should have "cached" or already have the data)
    assert len(legacy_references_cached) > 0
    assert all(ref_cat in legacy_references_cached for ref_cat in MODEL_REFERENCE_CATEGORY)
    # Verify cache is populated
    assert model_reference_manager.backend._converted_cache is not None


@pytest.mark.asyncio
async def test_manager_legacy_async(
    model_reference_manager: ModelReferenceManager,
    caplog: LogCaptureFixture,
) -> None:
    """Basic test of the legacy model reference manager async methods."""
    # Get legacy file paths from backend
    legacy_reference_locations = model_reference_manager.backend.get_all_category_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations

    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    client_session = aiohttp.ClientSession()

    # Fetch all categories asynchronously
    await model_reference_manager.backend.fetch_all_categories_async(
        aiohttp_client_session=client_session,
        force_refresh=True,
    )

    await client_session.close()

    # Verify backend has cached data
    legacy_references = {}
    for cat in MODEL_REFERENCE_CATEGORY:
        data = model_reference_manager.backend.fetch_category(cat)
        if data is not None:
            legacy_references[cat] = data

    assert len(legacy_references) > 0
    assert all(ref_cat in legacy_references for ref_cat in MODEL_REFERENCE_CATEGORY)

    # Access internal backend cache to verify caching
    from horde_model_reference.backends import LegacyGitHubBackend

    assert isinstance(model_reference_manager.backend, LegacyGitHubBackend)
    assert model_reference_manager.backend._converted_cache is not None
    assert len(model_reference_manager.backend._references_paths_cache) == len(MODEL_REFERENCE_CATEGORY)

    # Test cached access
    legacy_references_cached = {}
    for cat in MODEL_REFERENCE_CATEGORY:
        data = model_reference_manager.backend.fetch_category(cat)
        if data is not None:
            legacy_references_cached[cat] = data

    assert len(legacy_references_cached) > 0
    assert all(ref_cat in legacy_references_cached for ref_cat in MODEL_REFERENCE_CATEGORY)


def test_manager_new_format(model_reference_manager: ModelReferenceManager, caplog: LogCaptureFixture) -> None:
    """Test the new format model reference manager."""

    def assert_all_model_references_exist(
        model_reference_manager: ModelReferenceManager,
        override_existing: bool,
    ) -> None:
        """Assert that all model references exist."""
        all_model_references = model_reference_manager.get_all_model_references(override_existing=override_existing)
        for model_reference_category in MODEL_REFERENCE_CATEGORY:
            if model_reference_category == MODEL_REFERENCE_CATEGORY.text_generation:
                logger.warning("Skipping text generation model references, they are not yet implemented.")
                continue
            assert (
                model_reference_category in all_model_references
            ), f"Model reference category {model_reference_category} is missing"

            model_reference_instance = all_model_references[model_reference_category]

            assert (
                model_reference_instance is not None
            ), f"Model reference instance for {model_reference_category} is None"

            assert (
                model_reference_category in MODEL_CLASSIFICATION_LOOKUP
            ), f"Model reference category {model_reference_category} is not in the classification lookup"

            for _, model_entry in model_reference_instance.items():
                assert (
                    model_entry.model_classification == MODEL_CLASSIFICATION_LOOKUP[model_reference_category]
                ), f"Model entry for {model_reference_category} is not classified correctly"

    assert_all_model_references_exist(model_reference_manager, override_existing=True)

    assert_all_model_references_exist(model_reference_manager, override_existing=False)


def test_singleton_behavior() -> None:
    """Test that ModelReferenceManager is a singleton."""
    mgr1 = ModelReferenceManager()
    mgr2 = ModelReferenceManager()
    assert mgr1 is mgr2, "ModelReferenceManager should be a singleton"


def test_invalidate_cache(model_reference_manager: ModelReferenceManager) -> None:
    """Test that the cache invalidation works."""
    # Fill the cache
    model_reference_manager.get_all_model_references(override_existing=False)
    assert model_reference_manager._cached_file_json  # Should not be empty
    model_reference_manager._invalidate_cache()
    assert model_reference_manager._cached_file_json == {}


def test_file_json_dict_to_model_reference_valid(model_reference_manager: ModelReferenceManager) -> None:
    """Test valid conversion from a dict (parsed JSON) to model reference."""
    # Use a valid minimal dict for a known category
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    model_classification = ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.miscellaneous,
    )
    file_json_dict = {
        "test_model": {
            "name": "test_model",
            "model_classification": model_classification,
        }
    }
    result = model_reference_manager._file_json_dict_to_model_reference(category, file_json_dict)
    assert isinstance(result, dict)
    assert "test_model" in result
    assert isinstance(result["test_model"], GenericModelRecord)


def test_file_json_dict_to_model_reference_invalid(model_reference_manager: ModelReferenceManager) -> None:
    """Test invalid conversion from a dict (parsed JSON) to model reference for both safe and non-safe modes."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    file_json_dict = {
        "invalid_model": {
            "description": "An invalid model without a name or classification",
        }
    }
    with pytest.raises(ValidationError):
        model_reference_manager._file_json_dict_to_model_reference(category, file_json_dict, safe_mode=True)

    non_safe_mode_result = model_reference_manager._file_json_dict_to_model_reference(
        category, file_json_dict, safe_mode=False
    )

    assert non_safe_mode_result is None, "Expected None result for invalid input in non-safe mode"


def test_model_reference_to_json_dict(model_reference_manager: ModelReferenceManager) -> None:
    """Test conversion from model reference to dict (for JSON serialization)."""
    category = MODEL_REFERENCE_CATEGORY.miscellaneous
    model_classification = ModelClassification(
        domain=MODEL_DOMAIN.image,
        purpose=MODEL_PURPOSE.miscellaneous,
    )
    file_json_dict = {
        "test_model": {
            "name": "test_model",
            "model_classification": model_classification,
        }
    }
    model_ref = model_reference_manager._file_json_dict_to_model_reference(category, file_json_dict, safe_mode=True)
    assert model_ref is not None, "Model reference conversion failed in safe mode, this should be impossible here."
    json_dict = model_reference_manager.model_reference_to_json_dict(model_ref)
    assert isinstance(json_dict, dict)
    assert "test_model" in json_dict
    assert json_dict["test_model"]["name"] == "test_model"


def test_model_reference_to_json_dict_none(model_reference_manager: ModelReferenceManager) -> None:
    """Test conversion from None model reference to dict returns None."""
    with pytest.raises(ValueError):
        model_reference_manager.model_reference_to_json_dict_safe(None)  # type: ignore

    with pytest.raises(ValueError):
        model_reference_manager.model_reference_to_json_dict(None, safe_mode=True)  # type: ignore

    with pytest.raises(ValueError):
        model_reference_manager.model_reference_to_json_dict(None, safe_mode=False)  # type: ignore
