"""Tests for text_model_duplicates.py — backend-prefixed duplicate management."""

# ruff: noqa: D102

from __future__ import annotations

from typing import Any

from horde_model_reference.text_model_duplicates import TextModelDuplicateManager


class TestGetVariantNames:
    """Tests for TextModelDuplicateManager.get_variant_names."""

    def test_author_slash_model_produces_all_variants(self) -> None:
        variants = TextModelDuplicateManager.get_variant_names("ReadyArt/Broken-Tutu-24B")
        assert "aphrodite/ReadyArt/Broken-Tutu-24B" in variants
        assert "koboldcpp/Broken-Tutu-24B" in variants
        assert len(variants) == 2

    def test_author_slash_model_excludes_base(self) -> None:
        variants = TextModelDuplicateManager.get_variant_names("ReadyArt/Broken-Tutu-24B")
        assert "ReadyArt/Broken-Tutu-24B" not in variants

    def test_simple_name_no_slash(self) -> None:
        variants = TextModelDuplicateManager.get_variant_names("my-model-7B")
        assert "aphrodite/my-model-7B" in variants
        assert "koboldcpp/my-model-7B" in variants
        # No flattened variant when there's no "/" in the base name
        assert len(variants) == 2

    def test_no_duplicate_entries(self) -> None:
        variants = TextModelDuplicateManager.get_variant_names("SimpleModel")
        assert len(variants) == len(set(variants))


class TestGetAllNames:
    """Tests for TextModelDuplicateManager.get_all_names."""

    def test_starts_with_base_name(self) -> None:
        all_names = TextModelDuplicateManager.get_all_names("Author/Model-7B")
        assert all_names[0] == "Author/Model-7B"

    def test_includes_all_variants(self) -> None:
        all_names = TextModelDuplicateManager.get_all_names("Author/Model-7B")
        variants = TextModelDuplicateManager.get_variant_names("Author/Model-7B")
        assert all_names[1:] == variants


class TestGenerateDuplicates:
    """Tests for TextModelDuplicateManager.generate_duplicates."""

    def test_produces_all_variant_keys(self) -> None:
        record = {"name": "Author/Model-7B", "parameters": 7_000_000_000}
        dupes = TextModelDuplicateManager.generate_duplicates("Author/Model-7B", record)
        expected_variants = TextModelDuplicateManager.get_variant_names("Author/Model-7B")
        assert sorted(dupes.keys()) == sorted(expected_variants)

    def test_name_field_updated_on_duplicates(self) -> None:
        record = {"name": "Author/Model-7B", "parameters": 7_000_000_000}
        dupes = TextModelDuplicateManager.generate_duplicates("Author/Model-7B", record)
        for variant_name, variant_record in dupes.items():
            assert variant_record["name"] == variant_name

    def test_duplicates_are_deep_copies(self) -> None:
        record: dict[str, Any] = {"name": "Author/Model-7B", "parameters": 7_000_000_000, "tags": ["chat"]}
        dupes = TextModelDuplicateManager.generate_duplicates("Author/Model-7B", record)
        # Mutating the original should not affect duplicates
        record["tags"].append("mutated")
        for variant_record in dupes.values():
            assert "mutated" not in variant_record["tags"]

    def test_non_name_fields_preserved(self) -> None:
        record = {"name": "Author/Model-7B", "parameters": 7_000_000_000, "style": "chat"}
        dupes = TextModelDuplicateManager.generate_duplicates("Author/Model-7B", record)
        for variant_record in dupes.values():
            assert variant_record["parameters"] == 7_000_000_000
            assert variant_record["style"] == "chat"

    def test_simple_name_produces_two_duplicates(self) -> None:
        record = {"name": "my-model", "parameters": 3_000_000_000}
        dupes = TextModelDuplicateManager.generate_duplicates("my-model", record)
        assert len(dupes) == 2


class TestStripDuplicatesFromData:
    """Tests for TextModelDuplicateManager.strip_duplicates_from_data."""

    def test_removes_prefixed_keeps_base(self) -> None:
        data = {
            "Author/Model": {"name": "Author/Model"},
            "aphrodite/Author/Model": {"name": "aphrodite/Author/Model"},
            "koboldcpp/Model": {"name": "koboldcpp/Model"},
        }
        stripped = TextModelDuplicateManager.strip_duplicates_from_data(data)
        assert list(stripped.keys()) == ["Author/Model"]

    def test_empty_input(self) -> None:
        assert TextModelDuplicateManager.strip_duplicates_from_data({}) == {}


class TestFindExistingVariants:
    """Tests for TextModelDuplicateManager.find_existing_variants."""

    def test_finds_present_variants(self) -> None:
        data: dict[str, Any] = {
            "Author/Model": {},
            "aphrodite/Author/Model": {},
            "koboldcpp/Model": {},
        }
        found = TextModelDuplicateManager.find_existing_variants("Author/Model", data)
        assert "aphrodite/Author/Model" in found
        assert "koboldcpp/Model" in found

    def test_missing_variants_not_returned(self) -> None:
        data: dict[str, Any] = {"Author/Model": {}}
        found = TextModelDuplicateManager.find_existing_variants("Author/Model", data)
        assert found == []
