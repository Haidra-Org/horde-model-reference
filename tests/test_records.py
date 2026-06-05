import pytest
from pydantic import ValidationError

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
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
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
            ],
        ),
    )


def test_image_generation_model_record_unknown_baseline() -> None:
    """Tests the ImageGeneration_ModelRecord class with an unknown baseline."""
    with pytest.raises(ValidationError, match="Unknown baseline:"):
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
            baseline="unknown_baseline",
            tags=["test_tag"],
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
                ],
            ),
        )


def test_image_generation_model_record_unknown_style() -> None:
    """Tests the ImageGeneration_ModelRecord class with an unknown style."""
    with pytest.raises(ValidationError, match="Unknown style:"):
        ImageGenerationModelRecord(
            name="test_name",
            description="test_description",
            version="test_version",
            style="unknown_style",
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.generation,
            ),
            inpainting=False,
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            tags=["test_tag"],
            nsfw=False,
            config=GenericModelRecordConfig(
                download=[
                    DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
                ],
            ),
        )


def _make_generic_record(downloads: list[DownloadRecord] | None = None) -> GenericModelRecord:
    config = GenericModelRecordConfig(download=downloads or [])
    return GenericModelRecord(
        name="prop_test",
        record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.miscellaneous),
        config=config,
    )


def test_primary_download_url_with_downloads() -> None:
    """Verify primary_download_url returns the first download's URL."""
    record = _make_generic_record(
        [
            DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa"),
            DownloadRecord(file_name="b.bin", file_url="https://example.com/b.bin", sha256sum="bbb"),
        ]
    )
    assert record.primary_download_url == "https://example.com/a.bin"


def test_primary_download_url_empty() -> None:
    """Verify primary_download_url returns None when no downloads exist."""
    record = _make_generic_record()
    assert record.primary_download_url is None


def test_all_download_urls() -> None:
    """Verify all_download_urls returns every download URL in order."""
    record = _make_generic_record(
        [
            DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa"),
            DownloadRecord(file_name="b.bin", file_url="https://example.com/b.bin", sha256sum="bbb"),
        ]
    )
    assert record.all_download_urls == ["https://example.com/a.bin", "https://example.com/b.bin"]


def test_all_download_urls_empty() -> None:
    """Verify all_download_urls returns an empty list when no downloads exist."""
    record = _make_generic_record()
    assert record.all_download_urls == []


def test_download_count() -> None:
    """Verify download_count reflects the number of download entries."""
    record = _make_generic_record(
        [
            DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa"),
            DownloadRecord(file_name="b.bin", file_url="https://example.com/b.bin", sha256sum="bbb"),
        ]
    )
    assert record.download_count == 2


def test_download_count_zero() -> None:
    """Verify download_count is zero when no downloads exist."""
    record = _make_generic_record()
    assert record.download_count == 0


def test_model_record_union_covers_all_registered_types() -> None:
    """Verify ModelRecordUnionType includes every type registered in MODEL_RECORD_TYPE_LOOKUP.

    TS-4: If a new record class is registered via @register_record_type but not added
    to the union, type narrowing breaks silently. This test catches that drift.
    """
    from types import UnionType

    from horde_model_reference.service.v2.models import ModelRecordUnionType

    # Extract the set of types from the union
    assert isinstance(ModelRecordUnionType, UnionType), (
        f"ModelRecordUnionType should be a Union, got {type(ModelRecordUnionType)}"
    )

    assert hasattr(ModelRecordUnionType, "__args__"), "ModelRecordUnionType should have __args__ attribute"
    union_members = set(ModelRecordUnionType.__args__)

    # Every distinct record type in the lookup must appear in the union
    registered_types = set(MODEL_RECORD_TYPE_LOOKUP.values())
    missing = registered_types - union_members
    assert not missing, (
        f"Record types registered via @register_record_type but missing from ModelRecordUnionType: "
        f"{', '.join(cls.__name__ for cls in missing)}"
    )
