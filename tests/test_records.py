import pytest
from pydantic import ValidationError

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    KNOWN_IMAGE_SCHEDULER,
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


def test_category_returns_enum_for_known_record_type() -> None:
    """Verify category resolves a record's record_type to the matching enum member."""
    record = _make_generic_record()
    assert record.category is MODEL_REFERENCE_CATEGORY.miscellaneous


def test_category_returns_none_for_unknown_record_type() -> None:
    """Verify category yields None (rather than raising) for an unrecognised record_type string."""
    record = GenericModelRecord(
        name="junk",
        record_type="not_a_real_category",
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.miscellaneous),
    )
    assert record.category is None


def test_size_on_disk_bytes_available_on_base_record() -> None:
    """Verify size_on_disk_bytes is a base-record field (lifted from ImageGenerationModelRecord)."""
    record = GenericModelRecord(
        name="sized",
        record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.miscellaneous),
        size_on_disk_bytes=1234,
    )
    assert record.size_on_disk_bytes == 1234


def test_declared_total_size_prefers_summed_per_file_sizes() -> None:
    """Verify declared_total_size_bytes sums per-file sizes when every download declares one."""
    record = _make_generic_record(
        [
            DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa", size_bytes=10),
            DownloadRecord(file_name="b.bin", file_url="https://example.com/b.bin", sha256sum="bbb", size_bytes=20),
        ]
    )
    record.size_on_disk_bytes = 999
    assert record.declared_total_size_bytes == 30


def test_declared_total_size_falls_back_to_aggregate_when_a_file_size_is_missing() -> None:
    """Verify declared_total_size_bytes falls back to the aggregate when any per-file size is absent."""
    record = _make_generic_record(
        [
            DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa", size_bytes=10),
            DownloadRecord(file_name="b.bin", file_url="https://example.com/b.bin", sha256sum="bbb"),
        ]
    )
    record.size_on_disk_bytes = 999
    assert record.declared_total_size_bytes == 999


def test_declared_total_size_is_none_when_undeclared() -> None:
    """Verify declared_total_size_bytes is None when neither per-file nor aggregate sizes exist."""
    record = _make_generic_record()
    assert record.declared_total_size_bytes is None


def test_component_hash_fields_round_trip() -> None:
    """Verify content_hash and embedded_component_hashes round-trip and validate under extra='forbid'.

    Tests run with AI_HORDE_TESTING=True, so get_default_config() yields extra='forbid'; constructing and
    re-validating with the new component-hash fields confirms they are declared on the models rather than
    silently accepted (or rejected) as extras.
    """
    record = _make_generic_record(
        [
            DownloadRecord(
                file_name="ae.safetensors",
                file_url="https://example.com/ae.safetensors",
                sha256sum="aaa",
                content_hash="vae-region-hash",
                file_purpose="vae",
            ),
        ]
    )
    record.config.embedded_component_hashes = {"vae": "embedded-vae-hash", "text_encoders": "embedded-te-hash"}

    reloaded = GenericModelRecord.model_validate(record.model_dump())
    assert reloaded.config.download[0].content_hash == "vae-region-hash"
    assert reloaded.config.embedded_component_hashes == {
        "vae": "embedded-vae-hash",
        "text_encoders": "embedded-te-hash",
    }


def test_component_hash_fields_default_none() -> None:
    """Verify the new component-hash fields default to None so existing records validate unchanged."""
    download = DownloadRecord(file_name="a.bin", file_url="https://example.com/a.bin", sha256sum="aaa")
    assert download.content_hash is None
    assert GenericModelRecordConfig(download=[download]).embedded_component_hashes is None


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


def _record_with_requirements(
    requirements: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None,
) -> ImageGenerationModelRecord:
    """Build a minimal image generation record carrying the given requirements."""
    return ImageGenerationModelRecord(
        name="test_name",
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        requirements=requirements,
    )


def test_scheduler_requirement_accepts_the_whole_vocabulary() -> None:
    """Every known schedule can be required, since validation now checks against the real resolved one."""
    every_scheduler = [str(scheduler) for scheduler in KNOWN_IMAGE_SCHEDULER]
    record = _record_with_requirements({"schedulers": every_scheduler})

    assert record.requirements is not None
    assert record.requirements["schedulers"] == every_scheduler


def test_scheduler_requirement_left_alone_when_already_normalized() -> None:
    """A record written in the schedule vocabulary passes through untouched."""
    record = _record_with_requirements({"schedulers": ["karras", "gits"], "clip_skip": 2})

    assert record.requirements == {"schedulers": ["karras", "gits"], "clip_skip": 2}


def test_legacy_karras_true_requirement_normalizes_to_the_karras_schedule() -> None:
    """A record from before schedules could be named keeps its meaning rather than being rejected."""
    record = _record_with_requirements({"karras": True, "clip_skip": 2})

    assert record.requirements == {"schedulers": ["karras"], "clip_skip": 2}


def test_legacy_karras_false_requirement_normalizes_to_the_normal_schedule() -> None:
    """The flag being off selected a schedule; it did not decline to select one."""
    record = _record_with_requirements({"karras": False})

    assert record.requirements == {"schedulers": ["normal"]}


def test_explicit_schedulers_win_over_the_legacy_flag() -> None:
    """Naming a schedule is the more specific statement, so it is not overwritten by the flag."""
    record = _record_with_requirements({"karras": True, "schedulers": ["simple"]})

    assert record.requirements == {"schedulers": ["simple"]}


def test_a_non_boolean_legacy_flag_is_dropped_rather_than_guessed() -> None:
    """The legacy key only ever carried a boolean; anything else states no schedule to preserve."""
    record = _record_with_requirements({"karras": "yes"})

    assert record.requirements == {}


def test_records_without_requirements_are_unaffected() -> None:
    """Most records carry no requirements at all, and must not gain any."""
    assert _record_with_requirements(None).requirements is None


def test_requirements_without_schedules_are_unaffected() -> None:
    """Requirements that name no schedule are passed through untouched."""
    record = _record_with_requirements({"clip_skip": 2, "min_steps": 3})

    assert record.requirements == {"clip_skip": 2, "min_steps": 3}


def test_an_unknown_scheduler_requirement_does_not_reject_the_record() -> None:
    """A mistyped schedule in published data must not make the whole reference unparseable.

    Every worker and client parses this file. The unknown value is kept, where it simply matches no
    schedule a request can resolve to, rather than taking the reference down.
    """
    record = _record_with_requirements({"schedulers": ["not_a_real_schedule"]})

    assert record.requirements == {"schedulers": ["not_a_real_schedule"]}


def test_a_malformed_schedulers_value_is_left_alone() -> None:
    """The requirement is a list; a scalar is not something to iterate over character by character."""
    record = _record_with_requirements({"schedulers": "karras"})

    assert record.requirements == {"schedulers": "karras"}


def test_normalization_does_not_mutate_the_caller_dict() -> None:
    """Records are built from parsed JSON that a caller may still be holding."""
    original = {"karras": True}
    _record_with_requirements(original)

    assert original == {"karras": True}
