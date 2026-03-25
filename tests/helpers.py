from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord


def verify_model_references_structure(
    all_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]],
) -> None:
    """Verify the structure of model references dict.

    Args:
        all_references: Dictionary of model references to validate.

    Raises:
        AssertionError: If validation fails.
    """
    # Should have all categories
    assert len(all_references) == len(MODEL_REFERENCE_CATEGORY)

    for category in MODEL_REFERENCE_CATEGORY:
        assert category in all_references, f"Category {category} is missing"

        references = all_references[category]
        if references is None:
            # None is acceptable for categories without data
            continue

        # Verify structure
        assert isinstance(references, dict), f"References for {category} should be a dict"

        # Verify each model record
        for model_name, model_record in references.items():
            assert isinstance(
                model_record, GenericModelRecord
            ), f"Model record for {model_name} should be GenericModelRecord or subclass"
            assert model_record.name == model_name, f"Model name mismatch for {model_name}"
            assert model_record.model_classification is not None, f"Model {model_name} missing classification"


# All model categories for parameterized tests
ALL_MODEL_CATEGORIES = [
    MODEL_REFERENCE_CATEGORY.blip,
    MODEL_REFERENCE_CATEGORY.clip,
    MODEL_REFERENCE_CATEGORY.codeformer,
    MODEL_REFERENCE_CATEGORY.controlnet,
    MODEL_REFERENCE_CATEGORY.esrgan,
    MODEL_REFERENCE_CATEGORY.gfpgan,
    MODEL_REFERENCE_CATEGORY.safety_checker,
    MODEL_REFERENCE_CATEGORY.image_generation,
    MODEL_REFERENCE_CATEGORY.text_generation,
    MODEL_REFERENCE_CATEGORY.video_generation,
    MODEL_REFERENCE_CATEGORY.audio_generation,
    MODEL_REFERENCE_CATEGORY.miscellaneous,
]
