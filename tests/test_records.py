import sys

from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    KNOWN_MODEL_REFERENCE_INSTANCES,
    KNOWN_MODEL_REFERENCE_TYPES,
    MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP,
    DownloadRecord,
    ImageGeneration_ModelRecord,
)


def test_ensure_KNOWN_MODEL_consistency() -> None:
    """Tests the consistency of the KNOWN_MODEL_REFERENCE_INSTANCES and KNOWN_MODEL_REFERENCE_TYPES.

    `KNOWN_MODEL_REFERENCE_INSTANCES` should contain all of the (non-`type[]`) types, suitable for type hinting values
    which are *instances* of all of the model reference types.

    `KNOWN_MODEL_REFERENCE_TYPES` should contain all of the `type[]` types, suitable for type hinting values which are
    *types* of all of the model reference instances.
    """
    assert KNOWN_MODEL_REFERENCE_INSTANCES
    assert KNOWN_MODEL_REFERENCE_TYPES
    found_reference_types = []

    for model_reference_instance in KNOWN_MODEL_REFERENCE_INSTANCES.__args__:
        assert type[model_reference_instance] in KNOWN_MODEL_REFERENCE_TYPES.__args__
        found_reference_types.append(type[model_reference_instance])

    for model_reference_type in KNOWN_MODEL_REFERENCE_TYPES.__args__:
        assert model_reference_type in found_reference_types

    assert set(KNOWN_MODEL_REFERENCE_INSTANCES.__args__) == set(
        MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP.values(),
    ), "Found reference types do not match unique category lookup types"


def test_image_generation_model_record() -> None:
    """Tests the ImageGeneration_ModelRecord class."""
    ImageGeneration_ModelRecord(
        name="test_name",
        description="test_description",
        version="test_version",
        style="test_style",
        model_reference_category=MODEL_REFERENCE_CATEGORY.image_generation,
        model_classification=ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        ),
        purpose="test_purpose",
        inpainting=False,
        baseline="test_baseline",
        tags=["test_tag"],
        nsfw=False,
        config={
            "test_config": [
                DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
            ],
        },
    )
