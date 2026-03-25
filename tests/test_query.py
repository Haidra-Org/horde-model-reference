"""Tests for the ModelQuery fluent query builder."""

from __future__ import annotations

import pytest

from horde_model_reference.meta_consts import (
    CONTROLNET_STYLE,
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    TEXT_BACKENDS,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    ControlNetModelRecord,
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)
from horde_model_reference.query import (
    ImageGenerationQuery,
    TextModelQuery,
    build_cross_category_query,
    build_image_query,
    build_query,
    build_text_query,
)
from horde_model_reference.query_fields import (
    ImageFields,
    TextFields,
    false,
    true,
)


def _img_cls() -> ModelClassification:
    """Return an image generation classification."""
    return ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation)


def _text_cls() -> ModelClassification:
    """Return a text generation classification."""
    return ModelClassification(domain=MODEL_DOMAIN.text, purpose=MODEL_PURPOSE.generation)


def _cnet_cls() -> ModelClassification:
    """Return a controlnet classification."""
    return ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.auxiliary_or_patch)


def _make_image_model(
    name: str,
    baseline: str = KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    nsfw: bool = False,
    tags: list[str] | None = None,
    style: str | None = None,
    inpainting: bool = False,
    size_on_disk_bytes: int | None = None,
) -> ImageGenerationModelRecord:
    """Create a test image generation model record."""
    return ImageGenerationModelRecord(
        name=name,
        baseline=baseline,
        nsfw=nsfw,
        tags=tags or [],
        style=style,
        inpainting=inpainting,
        size_on_disk_bytes=size_on_disk_bytes,
        model_classification=_img_cls(),
    )


def _make_text_model(
    name: str,
    parameters: int = 7_000_000_000,
    nsfw: bool = False,
    tags: list[str] | None = None,
) -> TextGenerationModelRecord:
    """Create a test text generation model record."""
    return TextGenerationModelRecord(
        name=name,
        parameters=parameters,
        nsfw=nsfw,
        tags=tags or [],
        model_classification=_text_cls(),
    )


def _make_controlnet_model(
    name: str,
    controlnet_style: str = CONTROLNET_STYLE.control_canny,
) -> ControlNetModelRecord:
    """Create a test controlnet model record."""
    return ControlNetModelRecord(
        name=name,
        controlnet_style=controlnet_style,
        model_classification=_cnet_cls(),
    )


@pytest.fixture()
def image_models() -> dict[str, ImageGenerationModelRecord]:
    """Return a set of test image generation models."""
    models = [
        _make_image_model("ModelA", nsfw=False, tags=["realistic", "generalist"], size_on_disk_bytes=4_000_000_000),
        _make_image_model("ModelB", nsfw=True, tags=["anime", "character"], size_on_disk_bytes=6_500_000_000),
        _make_image_model(
            "ModelC",
            baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
            nsfw=False,
            tags=["realistic"],
            size_on_disk_bytes=2_000_000_000,
        ),
        _make_image_model(
            "ModelD",
            nsfw=False,
            tags=["anime", "generalist"],
            inpainting=True,
            size_on_disk_bytes=6_500_000_000,
        ),
        _make_image_model("ModelE", nsfw=False, tags=[], size_on_disk_bytes=None),
    ]
    return {m.name: m for m in models}


@pytest.fixture()
def text_models() -> dict[str, TextGenerationModelRecord]:
    """Return a set of test text generation models."""
    models = [
        _make_text_model("SmallModel", parameters=3_000_000_000, tags=["instruct"]),
        _make_text_model("MediumModel", parameters=7_000_000_000, tags=["chat", "instruct"]),
        _make_text_model("LargeModel", parameters=13_000_000_000, tags=["chat"], nsfw=True),
        _make_text_model("HugeModel", parameters=70_000_000_000, tags=["instruct", "chat"]),
    ]
    return {m.name: m for m in models}


class TestWhereEquality:
    """Tests for equality-based .where() filters."""

    def test_simple_equality(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test filtering by a simple boolean field."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(nsfw=False).to_list()
        assert all(not m.nsfw for m in results)
        assert len(results) == 4

    def test_equality_with_enum(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test filtering by an enum field value."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelC"

    def test_multiple_where_calls_chain(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that chaining .where() calls combines predicates with AND."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(nsfw=False).where(inpainting=True).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelD"

    def test_multiple_kwargs_in_single_where(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test passing multiple kwargs in a single .where() call."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(nsfw=False, inpainting=True).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelD"


class TestWhereComparison:
    """Tests for comparison operator suffixes in .where()."""

    def test_lt(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test __lt operator."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(parameters_count__lt=10_000_000_000).to_list()
        names = {m.name for m in results}
        assert names == {"SmallModel", "MediumModel"}

    def test_gte(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test __gte operator."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(parameters_count__gte=13_000_000_000).to_list()
        names = {m.name for m in results}
        assert names == {"LargeModel", "HugeModel"}

    def test_lte(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test __lte operator."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(parameters_count__lte=7_000_000_000).to_list()
        assert len(results) == 2

    def test_gt(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test __gt operator."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(parameters_count__gt=13_000_000_000).to_list()
        assert len(results) == 1
        assert results[0].name == "HugeModel"

    def test_ne(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test __ne operator."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(nsfw__ne=True).to_list()
        assert len(results) == 3

    def test_in_explicit(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test explicit __in operator."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(name__in=["ModelA", "ModelC"]).to_list()
        assert len(results) == 2

    def test_in_implicit_from_list_value(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that passing a list value auto-upgrades to __in."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(name=["ModelA", "ModelC"]).to_list()
        assert len(results) == 2

    def test_in_implicit_from_set_value(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that passing a non-string iterable (set) auto-upgrades to __in."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(name={"ModelA", "ModelC"}).to_list()
        assert len(results) == 2

    def test_in_with_non_iterable_value_raises(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Validate __in rejects non-iterable values with a clear error."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="requires a non-string iterable"):
            q.where(name__in=123).to_list()

    def test_none_value_skipped_in_comparison(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that records with None values are excluded from comparison filters."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(size_on_disk_bytes__lt=5_000_000_000).to_list()
        names = {m.name for m in results}
        assert "ModelE" not in names
        assert "ModelA" in names
        assert "ModelC" in names

    def test_range_filter(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test combining __gte and __lte for range filtering."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.where(parameters_count__gte=7_000_000_000).where(parameters_count__lte=13_000_000_000).to_list()
        names = {m.name for m in results}
        assert names == {"MediumModel", "LargeModel"}


class TestTagFilters:
    """Tests for tag-based filtering."""

    def test_tags_any(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test tags_any with a single tag."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.tags_any(["anime"]).to_list()
        names = {m.name for m in results}
        assert names == {"ModelB", "ModelD"}

    def test_tags_any_multiple(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test tags_any with multiple tags (OR semantics)."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.tags_any(["realistic", "anime"]).to_list()
        names = {m.name for m in results}
        assert names == {"ModelA", "ModelB", "ModelC", "ModelD"}

    def test_tags_all(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test tags_all with multiple tags (AND semantics)."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.tags_all(["chat", "instruct"]).to_list()
        names = {m.name for m in results}
        assert names == {"MediumModel", "HugeModel"}

    def test_tags_none(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test tags_none excludes records with any matching tag."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.tags_none(["anime"]).to_list()
        names = {m.name for m in results}
        assert "ModelB" not in names
        assert "ModelD" not in names

    def test_tags_any_no_match(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test tags_any returns empty when no tags match."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.tags_any(["nonexistent_tag"]).to_list()
        assert len(results) == 0

    def test_tags_any_empty_tags_record(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that records with empty tags are not matched by tags_any."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.tags_any(["realistic"]).to_list()
        assert "ModelE" not in {m.name for m in results}


class TestFilter:
    """Tests for arbitrary lambda predicate filtering."""

    def test_lambda_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test filtering with a lambda predicate."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.filter(lambda m: m.name.startswith("Model") and m.name.endswith("A")).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelA"


class TestOrdering:
    """Tests for order_by."""

    def test_order_by_ascending(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test ascending sort order."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.order_by("parameters_count").to_list()
        params = [m.parameters_count for m in results]
        assert params == sorted(params)

    def test_order_by_descending(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test descending sort order."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.order_by("parameters_count", descending=True).to_list()
        params = [m.parameters_count for m in results]
        assert params == sorted(params, reverse=True)

    def test_order_by_with_none_values(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that None values sort last."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.order_by("size_on_disk_bytes").to_list()
        sizes = [m.size_on_disk_bytes for m in results]
        non_none = [s for s in sizes if s is not None]
        assert non_none == sorted(non_none)
        assert sizes[-1] is None

    def test_order_by_heterogeneous_values_raises(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Sorting on non-comparable values should raise a clear ValueError."""

        class DummyRecord(GenericModelRecord):
            sortable: object

        records = {
            "one": DummyRecord(
                record_type=MODEL_REFERENCE_CATEGORY.image_generation,
                name="one",
                model_classification=_img_cls(),
                sortable={"a": 1},
            ),
            "two": DummyRecord(
                record_type=MODEL_REFERENCE_CATEGORY.image_generation,
                name="two",
                model_classification=_img_cls(),
                sortable=[1, 2, 3],
            ),
        }
        q = build_query(records, DummyRecord)
        with pytest.raises(ValueError, match="not mutually comparable"):
            q.order_by("sortable").to_list()


class TestPagination:
    """Tests for limit and offset."""

    def test_limit(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that limit restricts result count."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.limit(2).to_list()
        assert len(results) == 2

    def test_offset(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that offset skips initial records."""
        q = build_query(text_models, TextGenerationModelRecord)
        all_results = q.order_by("parameters_count").to_list()
        offset_results = q.order_by("parameters_count").offset(2).to_list()
        assert offset_results == all_results[2:]

    def test_limit_and_offset(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test combined limit and offset."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.order_by("parameters_count").offset(1).limit(2).to_list()
        assert len(results) == 2

    def test_limit_exceeds_total(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that limit larger than total returns all results."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = q.limit(100).to_list()
        assert len(results) == 4


class TestTerminals:
    """Tests for terminal operations (first, count, distinct, group_by)."""

    def test_first_returns_record(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test first() returns a matching record."""
        q = build_query(image_models, ImageGenerationModelRecord)
        result = q.where(name="ModelA").first()
        assert result is not None
        assert result.name == "ModelA"

    def test_first_returns_none(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test first() returns None when no match."""
        q = build_query(image_models, ImageGenerationModelRecord)
        result = q.where(name="Nonexistent").first()
        assert result is None

    def test_count(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test count() returns correct number of matches."""
        q = build_query(image_models, ImageGenerationModelRecord)
        assert q.where(nsfw=False).count() == 4

    def test_distinct(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test distinct() returns unique field values."""
        q = build_query(image_models, ImageGenerationModelRecord)
        baselines = q.distinct("baseline")
        assert set(baselines) == {
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        }

    def test_group_by(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test group_by() groups records by field value."""
        q = build_query(image_models, ImageGenerationModelRecord)
        groups = q.group_by("nsfw")
        assert False in groups
        assert True in groups
        assert len(groups[False]) == 4
        assert len(groups[True]) == 1


class TestWhereClassification:
    """Tests for classification-based filtering."""

    def test_filter_by_domain(self) -> None:
        """Test filtering by MODEL_DOMAIN."""
        img_model = _make_image_model("ImgModel")
        text_model = _make_text_model("TextModel")
        cnet_model = _make_controlnet_model("CnetModel")

        all_records: dict[str, GenericModelRecord] = {
            img_model.name: img_model,
            text_model.name: text_model,
            cnet_model.name: cnet_model,
        }
        q = build_query(all_records, GenericModelRecord)
        results = q.where_classification(domain=MODEL_DOMAIN.text).to_list()
        assert len(results) == 1
        assert results[0].name == "TextModel"

    def test_filter_by_purpose(self) -> None:
        """Test filtering by MODEL_PURPOSE."""
        img_model = _make_image_model("ImgModel")
        cnet_model = _make_controlnet_model("CnetModel")

        all_records: dict[str, GenericModelRecord] = {
            img_model.name: img_model,
            cnet_model.name: cnet_model,
        }
        q = build_query(all_records, GenericModelRecord)
        results = q.where_classification(purpose=MODEL_PURPOSE.auxiliary_or_patch).to_list()
        assert len(results) == 1
        assert results[0].name == "CnetModel"

    def test_filter_by_domain_and_purpose(self) -> None:
        """Test filtering by both domain and purpose."""
        img_model = _make_image_model("ImgModel")
        cnet_model = _make_controlnet_model("CnetModel")

        all_records: dict[str, GenericModelRecord] = {
            img_model.name: img_model,
            cnet_model.name: cnet_model,
        }
        q = build_query(all_records, GenericModelRecord)
        results = q.where_classification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation).to_list()
        assert len(results) == 1
        assert results[0].name == "ImgModel"


class TestNestedFields:
    """Tests for nested field access via __ separator."""

    def test_nested_where_on_metadata(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test filtering on a nested metadata field."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = q.where(metadata__schema_version="2.0.0").to_list()
        assert len(results) == len(image_models)

    def test_nested_missing_segment_raises(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Missing nested segment should raise to surface typos early."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match=r"missing attribute|missing key|missing segment"):
            q.where(metadata__nonexistent_field="value").to_list()


class TestImmutability:
    """Tests that query chaining does not mutate previous instances."""

    def test_chaining_does_not_mutate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that each .where() returns a new independent query."""
        q1 = build_query(image_models, ImageGenerationModelRecord)
        q2 = q1.where(nsfw=False)
        q3 = q2.where(inpainting=True)

        assert q1.count() == 5
        assert q2.count() == 4
        assert q3.count() == 1


class TestValidation:
    """Tests for field and operator validation."""

    def test_invalid_field_raises(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that a nonexistent field raises ValueError."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.where(nonexistent_field="value")

    def test_invalid_nested_field_raises(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that a nonexistent top-level field in a nested path raises ValueError."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.where(bogus_field__subfield="value")

    def test_invalid_field_in_order_by(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that order_by with a nonexistent field raises ValueError."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.order_by("nonexistent_field")

    def test_invalid_field_in_distinct(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that distinct with a nonexistent field raises ValueError."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.distinct("nonexistent_field")

    def test_invalid_field_in_group_by(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that group_by with a nonexistent field raises ValueError."""
        q = build_query(image_models, ImageGenerationModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.group_by("nonexistent_field")

    def test_group_by_unhashable_value_raises(self) -> None:
        """Grouping on an unhashable value should surface a clear ValueError."""

        class DummyRecord(GenericModelRecord):
            unhashable: object

        records = {
            "one": DummyRecord(
                record_type=MODEL_REFERENCE_CATEGORY.image_generation,
                name="one",
                model_classification=_img_cls(),
                unhashable=[{"a": 1}],
            )
        }

        q = build_query(records, DummyRecord)
        with pytest.raises(ValueError, match="unhashable value"):
            q.group_by("unhashable")

    def test_invalid_field_in_tags_any(self) -> None:
        """Test that tags_any on a model without tags field raises ValueError."""
        model = _make_controlnet_model("CnetModel")
        q = build_query({"CnetModel": model}, ControlNetModelRecord)
        with pytest.raises(ValueError, match="does not exist"):
            q.tags_any(["tag"])


class TestCrossCategory:
    """Tests for cross-category query building."""

    def test_build_cross_category_query(self) -> None:
        """Test that build_cross_category_query includes all records."""
        img = _make_image_model("ImgModel")
        txt = _make_text_model("TextModel")

        refs: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {
            MODEL_REFERENCE_CATEGORY.image_generation: {img.name: img},
            MODEL_REFERENCE_CATEGORY.text_generation: {txt.name: txt},
        }
        q = build_cross_category_query(refs)
        assert q.count() == 2

    def test_cross_category_with_classification_filter(self) -> None:
        """Test cross-category query with classification-based filtering."""
        img = _make_image_model("ImgModel")
        txt = _make_text_model("TextModel")

        refs: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {
            MODEL_REFERENCE_CATEGORY.image_generation: {img.name: img},
            MODEL_REFERENCE_CATEGORY.text_generation: {txt.name: txt},
        }
        q = build_cross_category_query(refs)
        results = q.where_classification(domain=MODEL_DOMAIN.image).to_list()
        assert len(results) == 1
        assert results[0].name == "ImgModel"


class TestComplexQueries:
    """Tests that replicate aspirational user story patterns."""

    def test_story1_image_worker(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Story 1: Find SFW SDXL models under 5GB, ordered by size."""
        q = build_query(image_models, ImageGenerationModelRecord)
        results = (
            q.where(baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
            .where(nsfw=False)
            .where(size_on_disk_bytes__lt=5_000_000_000)
            .order_by("size_on_disk_bytes")
            .to_list()
        )
        assert len(results) == 1
        assert results[0].name == "ModelA"

    def test_story2_text_range(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Story 2: Find 7B-13B SFW instruct models."""
        q = build_query(text_models, TextGenerationModelRecord)
        results = (
            q.where(parameters_count__gte=7_000_000_000)
            .where(parameters_count__lte=13_000_000_000)
            .where(nsfw=False)
            .tags_any(["instruct"])
            .to_list()
        )
        assert len(results) == 1
        assert results[0].name == "MediumModel"

    def test_story3_cross_category_by_classification(self) -> None:
        """Story 3: Find all post-processing image models."""
        from horde_model_reference.model_reference_records import EsrganModelRecord, GfpganModelRecord

        esrgan = EsrganModelRecord(
            name="4x_Upscaler",
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.post_processing,
            ),
        )
        gfpgan = GfpganModelRecord(
            name="GFPGAN",
            model_classification=ModelClassification(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.post_processing,
            ),
        )
        img = _make_image_model("Generator")

        refs: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]] = {
            MODEL_REFERENCE_CATEGORY.esrgan: {esrgan.name: esrgan},
            MODEL_REFERENCE_CATEGORY.gfpgan: {gfpgan.name: gfpgan},
            MODEL_REFERENCE_CATEGORY.image_generation: {img.name: img},
        }
        results = (
            build_cross_category_query(refs)
            .where_classification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.post_processing)
            .to_list()
        )
        names = {m.name for m in results}
        assert names == {"4x_Upscaler", "GFPGAN"}

    def test_story4_worker_prefers_small_models(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """User story: A resource-constrained worker wants smallest non-NSFW models first."""
        # Rationale: demonstrates chaining filter + order_by with None-aware size handling.
        results = (
            build_query(image_models, ImageGenerationModelRecord)
            .where(nsfw=False)
            .order_by("size_on_disk_bytes")
            .limit(3)
            .to_list()
        )

        # User expectation: smallest available models that are SFW, ignoring None sizes at the end.
        names = [m.name for m in results]
        assert names[0] == "ModelC"  # 2 GB
        assert names[1] == "ModelA"  # 4 GB
        assert "ModelE" not in names  # size None sorts last and falls outside limit

    def test_story5_ops_audit_unique_baselines(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """User story: Ops needs the set of baselines in use to plan migrations."""
        # Rationale: exercises distinct on enum field and ensures deterministic set of values.
        baselines = build_query(image_models, ImageGenerationModelRecord).where(nsfw=False).distinct("baseline")

        assert set(baselines) == {
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
            KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        }

    def test_story6_api_client_contains_guard(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """User story: Client filters by tag membership without crashing on non-iterable fields."""
        # Rationale: demonstrates __contains guard returning False instead of TypeError on scalars.
        # Here we (ab)use size_on_disk_bytes__contains to ensure safety when field is numeric.
        results = (
            build_query(image_models, ImageGenerationModelRecord)
            .where(size_on_disk_bytes__contains=1)  # numeric field, should simply return no matches
            .to_list()
        )
        assert results == []


@pytest.fixture()
def text_models_with_backends() -> dict[str, TextGenerationModelRecord]:
    """Return text models including backend-prefixed variants."""
    models = [
        _make_text_model("sophosympatheia/Llama-3-8B-Instruct", parameters=8_000_000_000, tags=["instruct"]),
        _make_text_model("sophosympatheia/Llama-3-8B-Instruct-Q4_K_M", parameters=8_000_000_000, tags=["instruct"]),
        _make_text_model("sophosympatheia/Mistral-7B-v0.1", parameters=7_000_000_000),
        _make_text_model("koboldcpp/sophosympatheia/Llama-3-8B-Instruct", parameters=8_000_000_000, tags=["instruct"]),
        _make_text_model("aphrodite/sophosympatheia/Llama-3-8B-Instruct", parameters=8_000_000_000, tags=["instruct"]),
        _make_text_model("koboldcpp/Mistral-7B-v0.1", parameters=7_000_000_000),
    ]
    return {m.name: m for m in models}


class TestTextModelQuery:
    """Tests for the TextModelQuery subclass."""

    def test_build_text_query_returns_text_model_query(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """Test that build_text_query returns a TextModelQuery instance."""
        q = build_text_query(text_models_with_backends)
        assert isinstance(q, TextModelQuery)

    def test_clone_preserves_type_after_where(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """Test that _clone after .where() preserves TextModelQuery type."""
        q = build_text_query(text_models_with_backends)
        q2 = q.where(nsfw=False)
        assert isinstance(q2, TextModelQuery)

    def test_for_backend_koboldcpp(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test for_backend filters to koboldcpp-prefixed models only."""
        q = build_text_query(text_models_with_backends)
        results = q.for_backend(TEXT_BACKENDS.koboldcpp).to_list()
        assert len(results) == 2
        assert all(r.name.startswith("koboldcpp/") for r in results)

    def test_for_backend_aphrodite(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test for_backend filters to aphrodite-prefixed models only."""
        q = build_text_query(text_models_with_backends)
        results = q.for_backend(TEXT_BACKENDS.aphrodite).to_list()
        assert len(results) == 1
        assert results[0].name.startswith("aphrodite/")

    def test_for_backend_returns_text_model_query(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """Test for_backend returns TextModelQuery for continued chaining."""
        q = build_text_query(text_models_with_backends)
        assert isinstance(q.for_backend(TEXT_BACKENDS.koboldcpp), TextModelQuery)

    def test_exclude_backend_variations(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test exclude_backend_variations removes all prefixed entries."""
        q = build_text_query(text_models_with_backends)
        results = q.exclude_backend_variations().to_list()
        assert len(results) == 3
        for r in results:
            assert not r.name.startswith("koboldcpp/")
            assert not r.name.startswith("aphrodite/")

    def test_exclude_backend_variations_returns_text_model_query(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """Test exclude_backend_variations returns TextModelQuery."""
        q = build_text_query(text_models_with_backends)
        assert isinstance(q.exclude_backend_variations(), TextModelQuery)

    def test_only_quantized(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test only_quantized keeps only quantized variants."""
        q = build_text_query(text_models_with_backends)
        results = q.only_quantized().to_list()
        assert len(results) == 1
        assert "Q4_K_M" in results[0].name

    def test_exclude_quantized(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test exclude_quantized removes quantized variants."""
        q = build_text_query(text_models_with_backends)
        results = q.exclude_quantized().to_list()
        assert len(results) == 5
        assert all("Q4_K_M" not in r.name for r in results)

    def test_group_by_base_model(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test group_by_base_model groups Llama variants together and separates Mistral."""
        q = build_text_query(text_models_with_backends)
        groups = q.group_by_base_model()
        assert len(groups) >= 2
        llama_groups = {k: v for k, v in groups.items() if "Llama" in k or "llama" in k.lower()}
        mistral_groups = {k: v for k, v in groups.items() if "Mistral" in k or "mistral" in k.lower()}
        assert len(llama_groups) >= 1
        assert len(mistral_groups) >= 1
        total_llama = sum(len(v) for v in llama_groups.values())
        assert total_llama == 4

    def test_group_by_base_model_empty(self) -> None:
        """Test group_by_base_model on empty input returns empty dict."""
        q = build_text_query({})
        groups = q.group_by_base_model()
        assert groups == {}

    def test_exclude_backend_then_group(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Canonical-only records grouped by base model."""
        q = build_text_query(text_models_with_backends)
        groups = q.exclude_backend_variations().group_by_base_model()
        total_records = sum(len(v) for v in groups.values())
        assert total_records == 3

    def test_immutability(self, text_models_with_backends: dict[str, TextGenerationModelRecord]) -> None:
        """Test that chaining does not mutate original query."""
        q1 = build_text_query(text_models_with_backends)
        q2 = q1.for_backend(TEXT_BACKENDS.koboldcpp)
        q3 = q1.exclude_backend_variations()
        assert q1.count() == 6
        assert q2.count() == 2
        assert q3.count() == 3

    def test_story_grouped_without_backend_dupes(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """User story: get a grouped list without backend duplicates."""
        groups = build_text_query(text_models_with_backends).exclude_backend_variations().group_by_base_model()
        for records in groups.values():
            for r in records:
                assert not r.name.startswith("koboldcpp/")
                assert not r.name.startswith("aphrodite/")

    def test_story_quant_options_for_base(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """User story: find quantized options available for a given model."""
        quant_results = (
            build_text_query(text_models_with_backends).exclude_backend_variations().only_quantized().to_list()
        )
        assert len(quant_results) == 1
        assert "Q4_K_M" in quant_results[0].name

    def test_chain_for_backend_then_only_quantized(
        self, text_models_with_backends: dict[str, TextGenerationModelRecord]
    ) -> None:
        """Test chaining for_backend with only_quantized."""
        results = (
            build_text_query(text_models_with_backends).for_backend(TEXT_BACKENDS.koboldcpp).only_quantized().to_list()
        )
        assert len(results) == 0


class TestImageGenerationQuery:
    """Tests for the ImageGenerationQuery subclass."""

    def test_build_image_query_returns_image_generation_query(
        self, image_models: dict[str, ImageGenerationModelRecord]
    ) -> None:
        """Test that build_image_query returns an ImageGenerationQuery instance."""
        q = build_image_query(image_models)
        assert isinstance(q, ImageGenerationQuery)

    def test_clone_preserves_type_after_where(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that _clone after .where() preserves ImageGenerationQuery type."""
        q = build_image_query(image_models)
        q2 = q.where(nsfw=False)
        assert isinstance(q2, ImageGenerationQuery)

    def test_for_baseline(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test for_baseline filters to matching baseline only."""
        q = build_image_query(image_models)
        results = q.for_baseline(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelC"

    def test_for_baseline_returns_image_generation_query(
        self, image_models: dict[str, ImageGenerationModelRecord]
    ) -> None:
        """Test for_baseline returns ImageGenerationQuery for continued chaining."""
        q = build_image_query(image_models)
        assert isinstance(q.for_baseline(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl), ImageGenerationQuery)

    def test_only_nsfw(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test only_nsfw keeps only NSFW models."""
        q = build_image_query(image_models)
        results = q.only_nsfw().to_list()
        assert len(results) == 1
        assert results[0].name == "ModelB"

    def test_exclude_nsfw(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test exclude_nsfw removes NSFW models."""
        q = build_image_query(image_models)
        results = q.exclude_nsfw().to_list()
        assert len(results) == 4
        assert all(not m.nsfw for m in results)

    def test_only_inpainting(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test only_inpainting keeps only inpainting models."""
        q = build_image_query(image_models)
        results = q.only_inpainting().to_list()
        assert len(results) == 1
        assert results[0].name == "ModelD"

    def test_exclude_inpainting(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test exclude_inpainting removes inpainting models."""
        q = build_image_query(image_models)
        results = q.exclude_inpainting().to_list()
        assert len(results) == 4
        assert all(not m.inpainting for m in results)

    def test_immutability(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that chaining does not mutate original query."""
        q1 = build_image_query(image_models)
        q2 = q1.only_nsfw()
        q3 = q1.exclude_nsfw()
        assert q1.count() == 5
        assert q2.count() == 1
        assert q3.count() == 4

    def test_chained_filters(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test chaining baseline + NSFW + inpainting filters."""
        results = (
            build_image_query(image_models)
            .for_baseline(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)
            .exclude_nsfw()
            .exclude_inpainting()
            .order_by("size_on_disk_bytes")
            .to_list()
        )
        assert len(results) == 2
        assert results[0].name == "ModelA"


class TestSelfReturnType:
    """Tests that fluent methods preserve concrete subclass types via Self."""

    def test_text_query_where_preserves_type(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that where() on TextModelQuery returns TextModelQuery."""
        q = build_text_query(text_models)
        q2 = q.where(nsfw=False)
        assert isinstance(q2, TextModelQuery)

    def test_text_query_order_by_preserves_type(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that order_by() on TextModelQuery returns TextModelQuery."""
        q = build_text_query(text_models)
        q2 = q.order_by("parameters_count")
        assert isinstance(q2, TextModelQuery)

    def test_text_query_limit_preserves_type(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that limit() on TextModelQuery returns TextModelQuery."""
        q = build_text_query(text_models)
        q2 = q.limit(2)
        assert isinstance(q2, TextModelQuery)

    def test_text_query_filter_preserves_type(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that filter() on TextModelQuery returns TextModelQuery."""
        q = build_text_query(text_models)
        q2 = q.filter(lambda r: r.nsfw is False)
        assert isinstance(q2, TextModelQuery)

    def test_text_query_tags_any_preserves_type(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test that tags_any() on TextModelQuery returns TextModelQuery."""
        q = build_text_query(text_models)
        q2 = q.tags_any(["instruct"])
        assert isinstance(q2, TextModelQuery)

    def test_image_query_where_preserves_type(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that where() on ImageGenerationQuery returns ImageGenerationQuery."""
        q = build_image_query(image_models)
        q2 = q.where(nsfw=False)
        assert isinstance(q2, ImageGenerationQuery)

    def test_image_query_order_by_preserves_type(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that order_by() on ImageGenerationQuery returns ImageGenerationQuery."""
        q = build_image_query(image_models)
        q2 = q.order_by("size_on_disk_bytes")
        assert isinstance(q2, ImageGenerationQuery)

    def test_image_query_tags_any_preserves_type(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that tags_any() on ImageGenerationQuery returns ImageGenerationQuery."""
        q = build_image_query(image_models)
        q2 = q.tags_any(["realistic"])
        assert isinstance(q2, ImageGenerationQuery)


class TestFieldRefPredicates:
    """Tests for FieldRef comparison operators producing Predicate objects."""

    def test_eq_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test FieldRef == value produces a working Predicate."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.nsfw == false()).to_list()
        assert len(results) == 4
        assert all(not m.nsfw for m in results)

    def test_ne_predicate(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef != value."""
        q = build_text_query(text_models)
        results = q.where(TextFields.nsfw != true()).to_list()
        assert len(results) == 3

    def test_lt_predicate(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef < value."""
        q = build_text_query(text_models)
        results = q.where(TextFields.parameters_count < 10_000_000_000).to_list()
        names = {m.name for m in results}
        assert names == {"SmallModel", "MediumModel"}

    def test_gt_predicate(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef > value."""
        q = build_text_query(text_models)
        results = q.where(TextFields.parameters_count > 13_000_000_000).to_list()
        assert len(results) == 1
        assert results[0].name == "HugeModel"

    def test_le_predicate(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef <= value."""
        q = build_text_query(text_models)
        results = q.where(TextFields.parameters_count <= 7_000_000_000).to_list()
        assert len(results) == 2

    def test_ge_predicate(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef >= value."""
        q = build_text_query(text_models)
        results = q.where(TextFields.parameters_count >= 13_000_000_000).to_list()
        assert len(results) == 2

    def test_is_in_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test FieldRef.is_in()."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.name.is_in(["ModelA", "ModelC"])).to_list()
        assert len(results) == 2

    def test_contains_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test FieldRef.contains() on a list field."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.tags.contains("anime")).to_list()
        names = {m.name for m in results}
        assert names == {"ModelB", "ModelD"}

    def test_is_none_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test FieldRef.is_none()."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.size_on_disk_bytes.is_none()).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelE"

    def test_is_not_none_predicate(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test FieldRef.is_not_none()."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.size_on_disk_bytes.is_not_none()).to_list()
        assert len(results) == 4

    def test_none_skipped_in_lt(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test that None values are excluded from < comparisons."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.size_on_disk_bytes < 5_000_000_000).to_list()
        assert "ModelE" not in {m.name for m in results}


class TestPredicateComposition:
    """Tests for combining Predicate objects with &, |, ~."""

    def test_and_composition(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test (pred1 & pred2) filters by both conditions."""
        q = build_image_query(image_models)
        pred = (ImageFields.nsfw == false()) & (ImageFields.inpainting == true())
        results = q.where(pred).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelD"

    def test_or_composition(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test (pred1 | pred2) filters by either condition."""
        q = build_image_query(image_models)
        pred = (ImageFields.nsfw == true()) | (ImageFields.inpainting == true())
        results = q.where(pred).to_list()
        names = {m.name for m in results}
        assert names == {"ModelB", "ModelD"}

    def test_invert_composition(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test ~pred inverts the condition."""
        q = build_image_query(image_models)
        pred = ~(ImageFields.nsfw == true())
        results = q.where(pred).to_list()
        assert len(results) == 4

    def test_mixed_predicate_and_kwargs(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test mixing Predicate positional args with keyword args in where()."""
        q = build_image_query(image_models)
        results = q.where(ImageFields.nsfw == false(), inpainting=True).to_list()
        assert len(results) == 1
        assert results[0].name == "ModelD"


class TestOrderSpec:
    """Tests for OrderSpec with order_by()."""

    def test_asc_order_spec(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef.asc() with order_by()."""
        q = build_text_query(text_models)
        results = q.order_by(TextFields.parameters_count.asc()).to_list()
        params = [m.parameters_count for m in results]
        assert params == sorted(params)

    def test_desc_order_spec(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Test FieldRef.desc() with order_by()."""
        q = build_text_query(text_models)
        results = q.order_by(TextFields.parameters_count.desc()).to_list()
        params = [m.parameters_count for m in results]
        assert params == sorted(params, reverse=True)

    def test_order_spec_on_image_query(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Test OrderSpec on ImageGenerationQuery."""
        q = build_image_query(image_models)
        results = (
            q.where(ImageFields.size_on_disk_bytes.is_not_none())
            .order_by(ImageFields.size_on_disk_bytes.asc())
            .to_list()
        )
        sizes = [m.size_on_disk_bytes for m in results]
        assert sizes == sorted(s for s in sizes if s is not None)


class TestFieldDSLComplexQueries:
    """Integration tests combining field DSL with the full query API."""

    def test_story_sfw_xl_under_5gb(self, image_models: dict[str, ImageGenerationModelRecord]) -> None:
        """Story: Find SFW SDXL models under 5GB, ordered by size (field DSL version)."""
        results = (
            build_image_query(image_models)
            .where(
                ImageFields.baseline == KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
                ImageFields.nsfw == false(),
                ImageFields.size_on_disk_bytes < 5_000_000_000,
            )
            .order_by(ImageFields.size_on_disk_bytes.asc())
            .to_list()
        )
        assert len(results) == 1
        assert results[0].name == "ModelA"

    def test_story_text_range_filter(self, text_models: dict[str, TextGenerationModelRecord]) -> None:
        """Story: Find 7B-13B SFW instruct models (field DSL version)."""
        results = (
            build_text_query(text_models)
            .where(
                TextFields.parameters_count >= 7_000_000_000,
                TextFields.parameters_count <= 13_000_000_000,
                TextFields.nsfw == false(),
            )
            .tags_any(["instruct"])
            .to_list()
        )
        assert len(results) == 1
        assert results[0].name == "MediumModel"
