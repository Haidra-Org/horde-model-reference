from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    ImageGenerationModelRecord,
)


def test_image_generation_model_record() -> None:
    """Tests the ImageGeneration_ModelRecord class."""
    ImageGenerationModelRecord(
        name="test_name",
        description="test_description",
        version="test_version",
        style="test_style",
        model_classification=ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        ),
        inpainting=False,
        baseline="test_baseline",
        tags=["test_tag"],
        nsfw=False,
        config={
            "download": [
                DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
            ],
        },
    )
