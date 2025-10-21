import pytest
from pydantic import ValidationError

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_STYLE,
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
        style=MODEL_STYLE.realistic,
        model_classification=ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        ),
        inpainting=False,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        tags=["test_tag"],
        nsfw=False,
        config={
            "download": [
                DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
            ],
        },
    )


def test_image_generation_model_record_unknown_baseline() -> None:
    """Tests the ImageGeneration_ModelRecord class with an unknown baseline."""
    with pytest.raises(ValidationError, match="baseline\n"):
        ImageGenerationModelRecord(
            name="test_name",
            description="test_description",
            version="test_version",
            style=MODEL_STYLE.realistic,
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
            inpainting=False,
            baseline="unknown_baseline",  # type: ignore
            tags=["test_tag"],
            nsfw=False,
            config={
                "download": [
                    DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
                ],
            },
        )


def test_image_generation_model_record_unknown_style() -> None:
    """Tests the ImageGeneration_ModelRecord class with an unknown style."""
    with pytest.raises(ValidationError, match="style\n"):
        ImageGenerationModelRecord(
            name="test_name",
            description="test_description",
            version="test_version",
            style="unknown_style",  # type: ignore
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
            inpainting=False,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            tags=["test_tag"],
            nsfw=False,
            config={
                "download": [
                    DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
                ],
            },
        )
