"""Tests for text_model_write_processor.py and text_backend_names.py validation."""

# ruff: noqa: D102

from __future__ import annotations

import pytest

from horde_model_reference.sync.text_generation_serializer import LegacyRecordDict
from horde_model_reference.text_backend_names import (
    get_model_name_variants,
    has_legacy_text_backend_prefix,
    strip_backend_prefix,
    validate_not_backend_prefixed,
)
from horde_model_reference.text_model_write_processor import TextModelWriteProcessor


class TestValidateNotBackendPrefixed:
    """Tests for the backend prefix rejection on write paths."""

    def test_canonical_name_passes(self) -> None:
        validate_not_backend_prefixed("ReadyArt/Broken-Tutu-24B")

    def test_simple_name_passes(self) -> None:
        validate_not_backend_prefixed("my-model-7B")

    def test_aphrodite_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="aphrodite"):
            validate_not_backend_prefixed("aphrodite/ReadyArt/Broken-Tutu-24B")

    def test_koboldcpp_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="koboldcpp"):
            validate_not_backend_prefixed("koboldcpp/Broken-Tutu-24B")


class TestGetModelNameVariants:
    """Tests for get_model_name_variants — the single source of truth."""

    def test_canonical_is_first(self) -> None:
        variants = get_model_name_variants("Author/Model")
        assert variants[0] == "Author/Model"

    def test_author_model_produces_three_variants(self) -> None:
        variants = get_model_name_variants("ReadyArt/Broken-Tutu-24B")
        assert len(variants) == 3
        assert variants == [
            "ReadyArt/Broken-Tutu-24B",
            "aphrodite/ReadyArt/Broken-Tutu-24B",
            "koboldcpp/Broken-Tutu-24B",
        ]

    def test_simple_name_produces_three_variants(self) -> None:
        variants = get_model_name_variants("SimpleModel")
        assert len(variants) == 3
        assert variants == [
            "SimpleModel",
            "aphrodite/SimpleModel",
            "koboldcpp/SimpleModel",
        ]

    def test_no_duplicates_in_output(self) -> None:
        for name in ["Author/Model", "Model", "A/B"]:
            variants = get_model_name_variants(name)
            assert len(variants) == len(set(variants)), f"Duplicates in variants for {name!r}"


class TestHasLegacyTextBackendPrefix:
    """Tests for has_legacy_text_backend_prefix."""

    def test_aphrodite_detected(self) -> None:
        assert has_legacy_text_backend_prefix("aphrodite/Model") is True

    def test_koboldcpp_detected(self) -> None:
        assert has_legacy_text_backend_prefix("koboldcpp/Model") is True

    def test_canonical_not_detected(self) -> None:
        assert has_legacy_text_backend_prefix("Author/Model") is False

    def test_simple_name_not_detected(self) -> None:
        assert has_legacy_text_backend_prefix("my-model") is False


class TestStripBackendPrefix:
    """Tests for strip_backend_prefix."""

    def test_strips_aphrodite(self) -> None:
        assert strip_backend_prefix("aphrodite/Author/Model") == "Author/Model"

    def test_strips_koboldcpp(self) -> None:
        assert strip_backend_prefix("koboldcpp/Model") == "Model"

    def test_noop_for_canonical(self) -> None:
        assert strip_backend_prefix("Author/Model") == "Author/Model"

    def test_noop_for_simple(self) -> None:
        assert strip_backend_prefix("my-model") == "my-model"


class TestWriteProcessorValidateAndTransform:
    """Tests for TextModelWriteProcessor.validate_and_transform."""

    processor: TextModelWriteProcessor

    def setup_method(self) -> None:
        self.processor = TextModelWriteProcessor()

    def _base_record(self, **overrides: object) -> LegacyRecordDict:
        record: LegacyRecordDict = {"parameters": 7_000_000_000}
        record.update(overrides)
        return record

    def test_basic_transform(self) -> None:
        result = self.processor.validate_and_transform("Author/Model-7B", self._base_record())
        assert result["name"] == "Author/Model-7B"
        assert result["model_name"] == "Model-7B"
        assert result["parameters"] == 7_000_000_000

    def test_display_name_auto_generated(self) -> None:
        result = self.processor.validate_and_transform("Author/llama-2-7b-chat", self._base_record())
        assert result["display_name"] == "llama 2 7b chat"

    def test_display_name_preserved_when_provided(self) -> None:
        result = self.processor.validate_and_transform(
            "Author/Model",
            self._base_record(display_name="Custom Display Name"),
        )
        assert result["display_name"] == "Custom Display Name"

    def test_tags_include_size_bucket(self) -> None:
        result = self.processor.validate_and_transform("Author/Model", self._base_record())
        tags = result["tags"]
        assert isinstance(tags, list)
        assert "7B" in tags

    def test_tags_include_style(self) -> None:
        result = self.processor.validate_and_transform(
            "Author/Model",
            self._base_record(style="chat"),
        )
        tags = result["tags"]
        assert isinstance(tags, list)
        assert "chat" in tags

    def test_existing_tags_preserved(self) -> None:
        result = self.processor.validate_and_transform(
            "Author/Model",
            self._base_record(tags=["custom-tag"]),
        )
        tags = result["tags"]
        assert isinstance(tags, list)
        assert "custom-tag" in tags

    def test_backend_prefixed_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend prefix"):
            self.processor.validate_and_transform("aphrodite/Author/Model", self._base_record())

    def test_koboldcpp_prefixed_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend prefix"):
            self.processor.validate_and_transform("koboldcpp/Model", self._base_record())

    def test_url_shaped_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="URL"):
            self.processor.validate_and_transform("https://example.com/model", self._base_record())

    def test_defaults_applied(self) -> None:
        result = self.processor.validate_and_transform("Author/Model", self._base_record())
        # defaults.json provides baseline fields — verify at least one is present
        assert len(result) > 4  # name, model_name, parameters, tags + defaults

    def test_defaults_skipped_when_disabled(self) -> None:
        result = self.processor.validate_and_transform(
            "Author/Model",
            self._base_record(),
            apply_defaults=False,
        )
        # Without defaults, only the fields we set + auto-generated ones should be present
        assert "name" in result
        assert "parameters" in result


class TestWriteProcessorNormalizeParameters:
    """Tests for parameter normalization edge cases."""

    def setup_method(self) -> None:
        self.processor = TextModelWriteProcessor()

    def test_int_passthrough(self) -> None:
        assert self.processor.normalize_parameters("test", 7_000_000_000) == 7_000_000_000

    def test_float_truncated(self) -> None:
        assert self.processor.normalize_parameters("test", 7.5e9) == 7_500_000_000

    def test_string_numeric(self) -> None:
        assert self.processor.normalize_parameters("test", "7000000000") == 7_000_000_000

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="required"):
            self.processor.normalize_parameters("test", None)

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.processor.normalize_parameters("test", 0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            self.processor.normalize_parameters("test", -1)

    def test_bool_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            self.processor.normalize_parameters("test", True)

    def test_non_numeric_string_raises(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            self.processor.normalize_parameters("test", "not-a-number")


class TestWriteProcessorNormalizeSettings:
    """Tests for settings validation."""

    def setup_method(self) -> None:
        self.processor = TextModelWriteProcessor()

    def test_none_returns_none(self) -> None:
        assert self.processor.normalize_settings("test", None) is None

    def test_empty_string_returns_none(self) -> None:
        assert self.processor.normalize_settings("test", "") is None

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid keys"):
            self.processor.normalize_settings("test", {"totally_bogus_key_xyz": 42})

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="dictionary"):
            self.processor.normalize_settings("test", 42)

    def test_json_string_parsed(self) -> None:
        valid_keys = list(self.processor.generation_params.keys())
        if valid_keys:
            import json

            settings_json = json.dumps({valid_keys[0]: 1})
            result = self.processor.normalize_settings("test", settings_json)
            assert result is not None
            assert valid_keys[0] in result


class TestWriteProcessorGenerateTags:
    """Tests for tag generation."""

    def setup_method(self) -> None:
        self.processor = TextModelWriteProcessor()

    def test_size_tag_generated(self) -> None:
        tags = self.processor.generate_tags(
            parameters=7_000_000_000,
            existing_tags=None,
            style_for_tag=None,
        )
        assert "7B" in tags

    def test_small_model_size_tag(self) -> None:
        tags = self.processor.generate_tags(
            parameters=560_000_000,
            existing_tags=None,
            style_for_tag=None,
        )
        assert "1B" in tags

    def test_style_added_as_tag(self) -> None:
        tags = self.processor.generate_tags(
            parameters=7_000_000_000,
            existing_tags=None,
            style_for_tag="chat",
        )
        assert "chat" in tags

    def test_existing_tags_preserved(self) -> None:
        tags = self.processor.generate_tags(
            parameters=7_000_000_000,
            existing_tags=["custom"],
            style_for_tag=None,
        )
        assert "custom" in tags

    def test_comma_separated_string_tags(self) -> None:
        tags = self.processor.generate_tags(
            parameters=7_000_000_000,
            existing_tags="a,b,c",
            style_for_tag=None,
        )
        assert "a" in tags
        assert "b" in tags
        assert "c" in tags

    def test_tags_sorted_and_unique(self) -> None:
        tags = self.processor.generate_tags(
            parameters=7_000_000_000,
            existing_tags=["z", "a", "a"],
            style_for_tag=None,
        )
        assert tags == sorted(set(tags))


class TestExtractModelName:
    """Tests for TextModelWriteProcessor.extract_model_name."""

    def test_with_slash(self) -> None:
        assert TextModelWriteProcessor.extract_model_name("Author/Model") == "Model"

    def test_without_slash(self) -> None:
        assert TextModelWriteProcessor.extract_model_name("Model") == "Model"

    def test_url_raises(self) -> None:
        with pytest.raises(ValueError, match="URL"):
            TextModelWriteProcessor.extract_model_name("https://example.com/model")


class TestGenerateDisplayName:
    """Tests for TextModelWriteProcessor.generate_display_name."""

    def test_hyphens_to_spaces(self) -> None:
        assert TextModelWriteProcessor.generate_display_name("llama-2-7b") == "llama 2 7b"

    def test_underscores_to_spaces(self) -> None:
        assert TextModelWriteProcessor.generate_display_name("my_model_name") == "my model name"

    def test_multiple_spaces_collapsed(self) -> None:
        assert TextModelWriteProcessor.generate_display_name("a--b__c") == "a b c"

    def test_stripped(self) -> None:
        assert TextModelWriteProcessor.generate_display_name("-model-") == "model"
