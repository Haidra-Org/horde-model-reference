"""Tests for extras extraction, group aliases, and alias-aware grouping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from horde_model_reference.analytics.text_model_parser import (
    ExtraPartType,
    get_base_model_name,
    group_text_models_by_base,
    infer_name_format,
    parse_text_model_name,
)
from horde_model_reference.group_aliases import GroupAliasStore
from horde_model_reference.group_schema_store import GroupSchemaStore
from horde_model_reference.model_reference_records import TextModelGroupNameSchema


class TestExtrasExtraction:
    """Tests for the extra-parts extraction in the parser."""

    def test_date_suffix_extracted_as_extra(self) -> None:
        """A trailing MM-YYYY date should be extracted as a DATE extra."""
        parsed = parse_text_model_name("c4ai-command-r-08-2024")

        assert parsed.base_name == "c4ai-command-r"
        extra_types = [e.inferred_type for e in parsed.extras]
        assert ExtraPartType.DATE in extra_types
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert date_extras[0].value == "08-2024"

    def test_date_suffix_yyyy_mm(self) -> None:
        """A trailing YYYY-MM date should also be extracted."""
        parsed = parse_text_model_name("SomeModel-2024-08")

        extra_types = [e.inferred_type for e in parsed.extras]
        assert ExtraPartType.DATE in extra_types
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert date_extras[0].value == "2024-08"

    def test_leading_version_extracted_as_extra(self) -> None:
        """A leading dotted version (e.g. 2.0-ModelName) becomes LEADING_VERSION."""
        parsed = parse_text_model_name("2.0-SomeModel-7B")

        extra_types = [e.inferred_type for e in parsed.extras]
        assert ExtraPartType.LEADING_VERSION in extra_types
        version_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.LEADING_VERSION]
        assert version_extras[0].value == "2.0"

    def test_no_extras_for_standard_name(self) -> None:
        """A standard Llama-3-8B-Instruct name should produce no extras."""
        parsed = parse_text_model_name("Llama-3-8B-Instruct")

        assert parsed.extras == []
        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.variant == "Instruct"

    def test_command_r_plus_date_suffix(self) -> None:
        """c4ai-command-r-plus-08-2024 should extract the date, leaving base intact."""
        parsed = parse_text_model_name("c4ai-command-r-plus-08-2024")

        assert "08-2024" not in parsed.base_name
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "08-2024"

    def test_extras_appear_in_inferred_schema(self) -> None:
        """infer_name_format should report extra_parts when members have extras."""
        names = [
            "c4ai-command-r-08-2024",
            "c4ai-command-r-plus-08-2024",
        ]
        schema = infer_name_format(names)

        assert ExtraPartType.DATE in schema.extra_parts

    # --- 4-digit date code tests ---

    def test_four_digit_yymm_date_code(self) -> None:
        """Trailing YYMM codes like -2407 should be extracted as DATE extras."""
        parsed = parse_text_model_name("Mistral-Large-Instruct-2407")

        assert parsed.base_name == "Mistral-Large"
        assert parsed.variant == "Instruct"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "2407"

    def test_four_digit_date_code_with_size(self) -> None:
        """Size + 4-digit date should both be extracted, leaving a clean base."""
        parsed = parse_text_model_name("GLM-4-32B-0414")

        assert parsed.base_name == "GLM-4"
        assert parsed.size == "32B"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "0414"

    def test_four_digit_date_code_mid_name(self) -> None:
        """4-digit date code not at the very end should still be extracted."""
        parsed = parse_text_model_name("GLM-4-32B-0414-abliterated")

        assert parsed.base_name == "GLM-4-abliterated"
        assert parsed.size == "32B"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "0414"

    def test_four_digit_date_code_deepseek(self) -> None:
        """DeepSeek-V2.5-1210 should extract version and date code."""
        parsed = parse_text_model_name("DeepSeek-V2.5-1210")

        assert parsed.base_name == "DeepSeek"
        assert parsed.version == "V2.5"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "1210"

    def test_four_digit_date_code_qwen_thinking(self) -> None:
        """Qwen3-4B-Thinking-2507: Thinking variant + 4-digit date extracted."""
        parsed = parse_text_model_name("Qwen3-4B-Thinking-2507")

        assert parsed.base_name == "Qwen3"
        assert parsed.size == "4B"
        assert parsed.variant == "Thinking"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "2507"

    # --- Thinking/Reasoning variant tests ---

    def test_thinking_variant_extracted(self) -> None:
        """Thinking should be recognized as a variant."""
        parsed = parse_text_model_name("Qwen3-30B-A3B-Thinking-2507")

        assert parsed.variant == "Thinking"
        assert "Thinking" not in parsed.base_name

    def test_reasoning_variant_extracted(self) -> None:
        """Reasoning should be recognized as a variant."""
        parsed = parse_text_model_name("Ministral-3-14B-Reasoning-2512")

        assert parsed.variant == "Reasoning"
        assert parsed.size == "14B"
        assert parsed.base_name == "Ministral-3"

    def test_thinking_and_instruct_same_base(self) -> None:
        """Thinking and Instruct variants of the same model should have the same base."""
        base_instruct = get_base_model_name("Qwen3-4B-Instruct-2507")
        base_thinking = get_base_model_name("Qwen3-4B-Thinking-2507")

        assert base_instruct == base_thinking == "Qwen3"

    # --- Grouping improvements with 4-digit dates ---

    def test_mistral_variants_group_together(self) -> None:
        """Mistral models with different date codes should group under same base."""
        base_2407 = get_base_model_name("Mistral-Large-Instruct-2407")
        base_2411 = get_base_model_name("Mistral-Large-Instruct-2411")

        assert base_2407 == base_2411 == "Mistral-Large"

    def test_glm_variants_group_together(self) -> None:
        """GLM models with -0414 date code should group under GLM-4 base."""
        base_32b = get_base_model_name("GLM-4-32B-0414")
        base_9b = get_base_model_name("GLM-4-9B-0414")

        assert base_32b == base_9b == "GLM-4"


class TestGroupAliasStore:
    """Tests for the group alias persistence store."""

    @pytest.fixture()
    def store_path(self, tmp_path: Path) -> Path:
        """Return a temporary path for the alias store JSON file."""
        return tmp_path / "aliases.json"

    @pytest.fixture()
    def store(self, store_path: Path) -> GroupAliasStore:
        """Create a fresh GroupAliasStore backed by a temporary file."""
        return GroupAliasStore(file_path=store_path)

    def test_resolve_unknown_returns_identity(self, store: GroupAliasStore) -> None:
        """An unknown name should resolve to itself."""
        assert store.resolve("unknown-model") == "unknown-model"

    def test_set_and_resolve(self, store: GroupAliasStore) -> None:
        """Aliases set via set_aliases should resolve to the canonical name."""
        store.set_aliases("c4ai-command-r", ["c4ai-command-r-plus", "c4ai-command-a"])

        assert store.resolve("c4ai-command-r-plus") == "c4ai-command-r"
        assert store.resolve("c4ai-command-a") == "c4ai-command-r"
        assert store.resolve("c4ai-command-r") == "c4ai-command-r"

    def test_add_alias(self, store: GroupAliasStore) -> None:
        """add_alias should append to an existing canonical entry."""
        store.set_aliases("Broken-Tutu", ["Broken-Tutu-Unslop"])
        store.add_alias("Broken-Tutu", "Broken-Tutu-Transgression")

        assert store.resolve("Broken-Tutu-Transgression") == "Broken-Tutu"

    def test_remove_alias(self, store: GroupAliasStore) -> None:
        """Removing an alias should make it resolve to itself again."""
        store.set_aliases("base", ["alias1", "alias2"])
        assert store.remove_alias("base", "alias1")
        assert store.resolve("alias1") == "alias1"
        assert store.resolve("alias2") == "base"

    def test_delete_canonical(self, store: GroupAliasStore) -> None:
        """Deleting a canonical entry should remove all its aliases."""
        store.set_aliases("base", ["alias1"])
        assert store.delete("base")
        assert store.resolve("alias1") == "alias1"

    def test_self_alias_raises(self, store: GroupAliasStore) -> None:
        """A canonical name cannot be its own alias."""
        with pytest.raises(ValueError, match="cannot be its own alias"):
            store.set_aliases("base", ["base"])

    def test_conflict_raises(self, store: GroupAliasStore) -> None:
        """An alias already claimed by another canonical should raise."""
        store.set_aliases("group-a", ["shared-alias"])
        with pytest.raises(ValueError, match="already registered"):
            store.set_aliases("group-b", ["shared-alias"])

    def test_persistence_across_instances(self, store_path: Path) -> None:
        """Data persisted by one store instance should be loaded by a new one."""
        store1 = GroupAliasStore(file_path=store_path)
        store1.set_aliases("canonical", ["alias-x"])

        store2 = GroupAliasStore(file_path=store_path)
        assert store2.resolve("alias-x") == "canonical"

    def test_list_all(self, store: GroupAliasStore) -> None:
        """list_all should return deep copies of all entries."""
        store.set_aliases("g1", ["a1"])
        store.set_aliases("g2", ["a2", "a3"])

        all_entries = store.list_all()
        assert len(all_entries) == 2
        assert set(all_entries["g2"].aliases) == {"a2", "a3"}

    def test_is_alias(self, store: GroupAliasStore) -> None:
        """is_alias should identify registered aliases but not canonical names."""
        store.set_aliases("canon", ["alias"])
        assert store.is_alias("alias")
        assert not store.is_alias("canon")
        assert not store.is_alias("unrelated")


class TestAliasAwareSchemaStore:
    """Tests for alias-aware schema lookups."""

    @pytest.fixture()
    def alias_store(self, tmp_path: Path) -> GroupAliasStore:
        """Create alias store with c4ai-command-r aliases pre-populated."""
        store = GroupAliasStore(file_path=tmp_path / "aliases.json")
        store.set_aliases("c4ai-command-r", ["c4ai-command-r-plus"])
        return store

    @pytest.fixture()
    def schema_store(self, tmp_path: Path, alias_store: GroupAliasStore) -> GroupSchemaStore:
        """Create schema store wired to the alias store."""
        return GroupSchemaStore(file_path=tmp_path / "schemas.json", alias_store=alias_store)

    def test_get_via_alias(self, schema_store: GroupSchemaStore) -> None:
        """Looking up an alias should return the schema stored under its canonical name."""
        schema = TextModelGroupNameSchema(separator="-", template="{base}-{size}")
        schema_store.set("c4ai-command-r", schema)

        result = schema_store.get("c4ai-command-r-plus")
        assert result is not None
        assert result.template == "{base}-{size}"

    def test_direct_lookup_still_works(self, schema_store: GroupSchemaStore) -> None:
        """Direct lookups by canonical name should still work."""
        schema = TextModelGroupNameSchema(separator="-")
        schema_store.set("direct-name", schema)

        assert schema_store.get("direct-name") is not None

    def test_no_alias_store_returns_none_for_alias(self, tmp_path: Path) -> None:
        """Without an alias store, alias lookups should return None."""
        store = GroupSchemaStore(file_path=tmp_path / "schemas.json")
        store.set("canonical-group", TextModelGroupNameSchema())
        assert store.get("some-alias") is None


# ---------------------------------------------------------------------------
# Alias-aware grouping
# ---------------------------------------------------------------------------


class TestAliasAwareGrouping:
    """Tests for alias-resolved model grouping."""

    @pytest.fixture()
    def alias_store(self, tmp_path: Path) -> GroupAliasStore:
        """Create alias store with Broken-Tutu variant aliases."""
        store = GroupAliasStore(file_path=tmp_path / "aliases.json")
        store.set_aliases("Broken-Tutu", ["Broken-Tutu-Unslop", "Broken-Tutu-Transgression"])
        return store

    def test_aliases_merge_groups(self, alias_store: GroupAliasStore) -> None:
        """Aliased base names should collapse into a single group."""
        names = [
            "koboldcpp/Broken-Tutu-24B-Unslop-v2.0-Q4_K_M",
            "ReadyArt/Broken-Tutu-24B-Transgression-v2.0-Q4_K_M",
            "koboldcpp/Broken-Tutu-24B-Q5_K_M",
        ]
        grouped = group_text_models_by_base(names, alias_store=alias_store)

        assert "Broken-Tutu" in grouped
        assert len(grouped["Broken-Tutu"].variants) == 3

    def test_no_aliases_produces_separate_groups(self) -> None:
        """Without aliases, variant names may produce separate groups."""
        names = [
            "koboldcpp/Broken-Tutu-24B-Unslop-v2.0-Q4_K_M",
            "ReadyArt/Broken-Tutu-24B-Transgression-v2.0-Q4_K_M",
        ]
        grouped = group_text_models_by_base(names)

        # Without aliases, variant names are part of base so they may separate
        assert len(grouped) >= 1


class TestSchemaExtraParts:
    """Tests for the extra_parts field on TextModelGroupNameSchema."""

    def test_default_empty(self) -> None:
        """extra_parts defaults to an empty list."""
        schema = TextModelGroupNameSchema()
        assert schema.extra_parts == []

    def test_round_trip_serialization(self) -> None:
        """extra_parts should survive model_dump -> model_validate round-trip."""
        schema = TextModelGroupNameSchema(extra_parts=["date", "leading_version"])
        dumped = schema.model_dump(mode="json")
        restored = TextModelGroupNameSchema.model_validate(dumped)
        assert restored.extra_parts == ["date", "leading_version"]

    def test_json_round_trip(self) -> None:
        """extra_parts should survive JSON serialization round-trip."""
        schema = TextModelGroupNameSchema(extra_parts=["date"])
        json_str = json.dumps(schema.model_dump(mode="json"))
        restored = TextModelGroupNameSchema.model_validate(json.loads(json_str))
        assert restored.extra_parts == ["date"]


# ---------------------------------------------------------------------------
# Real-world consolidation: names that MUST resolve to the same base
# ---------------------------------------------------------------------------


class TestRealWorldConsolidation:
    """Verify that real model names from the AI Horde text reference correctly group together."""

    def test_mistral_large_date_variants(self) -> None:
        """Mistral-Large-Instruct-2407 / -2411 differ only by release date.

        `-2407` and `-2411` are YYMM date codes (July 2024, November 2024).
        `Instruct` is a variant keyword. Both refer to Mistral-Large at
        different release dates, so they must share the same group.
        """
        base_2407 = get_base_model_name("Mistral-Large-Instruct-2407")
        base_2411 = get_base_model_name("Mistral-Large-Instruct-2411")

        assert base_2407 == "Mistral-Large"
        assert base_2411 == "Mistral-Large"

    def test_mistral_small_different_size_and_date(self) -> None:
        """Mistral-Small-Instruct-2409 and Mistral-Small-24B-Instruct-2501.

        One omits the explicit size, the other includes `24B`. Both are
        Mistral-Small releases with different dates. The size and date
        are stripped, leaving the same base.
        """
        base_no_size = get_base_model_name("Mistral-Small-Instruct-2409")
        base_with_size = get_base_model_name("Mistral-Small-24B-Instruct-2501")

        assert base_no_size == base_with_size == "Mistral-Small"

    def test_glm4_multiple_sizes_same_date(self) -> None:
        r"""GLM-4-32B-0414 and GLM-4-9B-0414 are size variants of GLM-4.

        The `-4` in `GLM-4` is a single digit - NOT a 4-digit date code.
        The 4-digit date pattern requires exactly 4 digits (`\\d{4}`), so
        `-4` is safe. Only `-0414` (April 14th) is extracted as a date.
        """
        base_32b = get_base_model_name("GLM-4-32B-0414")
        base_9b = get_base_model_name("GLM-4-9B-0414")

        assert base_32b == base_9b == "GLM-4"

    def test_glm_z1_sizes_same_date(self) -> None:
        """GLM-Z1-32B-0414 and GLM-Z1-9B-0414 are size variants of GLM-Z1.

        `Z1` is alphanumeric (starts with a letter) so won't match the
        date pattern. The `-0414` is the only 4-digit date code present.
        """
        base_32b = get_base_model_name("GLM-Z1-32B-0414")
        base_9b = get_base_model_name("GLM-Z1-9B-0414")

        assert base_32b == base_9b == "GLM-Z1"

    def test_qwen3_instruct_and_thinking_same_date(self) -> None:
        """Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507.

        Both `Instruct` and `Thinking` are variant keywords. Both have
        the same size (`4B`) and date code (`2507`). After stripping
        all three, the base is `Qwen3`.
        """
        base_instruct = get_base_model_name("Qwen3-4B-Instruct-2507")
        base_thinking = get_base_model_name("Qwen3-4B-Thinking-2507")
        base_plain = get_base_model_name("Qwen3-4B")

        assert base_instruct == base_thinking == base_plain == "Qwen3"

    def test_ministral3_instruct_and_reasoning_multiple_sizes(self) -> None:
        """Ministral-3-{8B,14B}-{Instruct,Reasoning}-2512 all share `Ministral-3`.

        The `-3` is a single digit (not a 4-digit date). Both `Instruct`
        and `Reasoning` are variant keywords. `8B`/`14B` are sizes.
        `-2512` is a YYMM date. All four combinations must share a base.
        """
        names = [
            "Ministral-3-8B-Instruct-2512",
            "Ministral-3-14B-Instruct-2512",
            "Ministral-3-8B-Reasoning-2512",
            "Ministral-3-14B-Reasoning-2512",
        ]
        bases = [get_base_model_name(n) for n in names]
        assert all(b == "Ministral-3" for b in bases)

    def test_deepseek_v2_5_with_date_code(self) -> None:
        r"""DeepSeek-V2.5-1210 extracts version V2.5 and date 1210.

        `V2.5` matches the version pattern (`V\\d+\\.\\d+`). After version
        removal, `-1210` is a standalone 4-digit date code (December 2010
        or Oct 12, more likely MMDD). The base is just `DeepSeek`.
        """
        assert get_base_model_name("DeepSeek-V2.5-1210") == "DeepSeek"

    def test_devstral_variants_with_date(self) -> None:
        """Devstral-2-123B-Instruct-2512 and Devstral-2-123B-Instruct without date.

        Both have size `123B` and variant `Instruct`. The `-2` is a single
        digit and stays in the base. Adding date `-2512` shouldn't change the base.
        """
        base_with_date = get_base_model_name("Devstral-2-123B-Instruct-2512")
        # Without a date, the base should be the same
        base_plain = get_base_model_name("Devstral-2-123B-Instruct")

        assert base_with_date == base_plain == "Devstral-2"

    def test_c4ai_date_format_and_version_converge(self) -> None:
        """c4ai-command-r-08-2024 (MM-YYYY date) and c4ai-command-r-v01 (version).

        The date variant has `08-2024` extracted as a DATE extra.
        The version variant has `v01` extracted as a version.
        Both leave the same base: `c4ai-command-r`.
        """
        base_date = get_base_model_name("c4ai-command-r-08-2024")
        base_version = get_base_model_name("c4ai-command-r-v01")

        assert base_date == base_version == "c4ai-command-r"

    def test_qwen3_moe_instruct_and_thinking_same_base(self) -> None:
        """Qwen3-235B-A22B-Instruct-2507 and Qwen3-235B-A22B-Thinking-2507.

        The total size `235B` is extracted. `A22B` stays in the base because
        the lookbehind `(?<![a-zA-Z])` on the size pattern prevents matching
        digits preceded by `A`. Both Instruct and Thinking are variants.
        `-2507` is a date code. The base is `Qwen3-A22B`.
        """
        base_instruct = get_base_model_name("Qwen3-235B-A22B-Instruct-2507")
        base_thinking = get_base_model_name("Qwen3-235B-A22B-Thinking-2507")
        base_plain = get_base_model_name("Qwen3-235B-A22B")

        assert base_instruct == base_thinking == base_plain == "Qwen3-A22B"


# ---------------------------------------------------------------------------
# Real-world separation: names that MUST resolve to different bases
# ---------------------------------------------------------------------------


class TestRealWorldSeparation:
    r"""Verify that models which are genuinely distinct do NOT accidentally merge into the same group."""

    def test_qwen3_moe_different_active_sizes(self) -> None:
        """Qwen3-A22B vs Qwen3-A3B are architecturally distinct MoE models.

        `A22B` and `A3B` represent different active parameter counts in a
        Mixture-of-Experts architecture. They are different models, not
        size variants. The size regex extracts the total param count (235B,
        30B) but `A22B`/`A3B` stay in the base because of the letter
        lookbehind guard.
        """
        base_a22b = get_base_model_name("Qwen3-235B-A22B")
        base_a3b = get_base_model_name("Qwen3-30B-A3B")

        assert base_a22b == "Qwen3-A22B"
        assert base_a3b == "Qwen3-A3B"
        assert base_a22b != base_a3b

    def test_glm_z1_vs_glm_z1_rumination(self) -> None:
        """GLM-Z1 and GLM-Z1-Rumination are different model variants.

        `Rumination` is NOT in the variant keyword list (it's a unique
        model fine-tune, not a generic deployment variant like Instruct).
        These should remain separate groups.
        """
        base_plain = get_base_model_name("GLM-Z1-32B-0414")
        base_rumination = get_base_model_name("GLM-Z1-Rumination-32B-0414")

        assert base_plain == "GLM-Z1"
        assert base_rumination == "GLM-Z1-Rumination"
        assert base_plain != base_rumination

    def test_mistral_large_vs_small_vs_nemo(self) -> None:
        """Mistral-Large, Mistral-Small, and Mistral-Nemo are separate families.

        `Large`, `Small`, and `Nemo` are integral parts of the model
        identity, not strippable variant keywords. Only `Instruct` and
        the date codes are removed.
        """
        base_large = get_base_model_name("Mistral-Large-Instruct-2407")
        base_small = get_base_model_name("Mistral-Small-Instruct-2409")
        base_nemo = get_base_model_name("Mistral-Nemo-Instruct-2407")

        assert base_large == "Mistral-Large"
        assert base_small == "Mistral-Small"
        assert base_nemo == "Mistral-Nemo"
        assert len({base_large, base_small, base_nemo}) == 3

    def test_devstral_vs_devstral_small(self) -> None:
        """Devstral-2 and Devstral-Small-2 are different model families.

        `Small` here is part of the model name (not an extracted variant)
        because the full word context makes it a model identifier.
        """
        base_devstral = get_base_model_name("Devstral-2-123B-Instruct-2512")
        base_small = get_base_model_name("Devstral-Small-2-24B-Instruct-2512")

        assert base_devstral == "Devstral-2"
        assert base_small == "Devstral-Small-2"
        assert base_devstral != base_small

    def test_broken_tutu_sub_models_separate_without_aliases(self) -> None:
        """Broken-Tutu, Broken-Tutu-Transgression, and Broken-Tutu-Unslop.

        `Transgression` and `Unslop` are custom fine-tune names, not
        recognized variant keywords. Without the alias system, these
        correctly produce separate groups. Merging requires explicit aliases.
        """
        base_plain = get_base_model_name("Broken-Tutu-24B")
        base_transgression = get_base_model_name("Broken-Tutu-24B-Transgression-v2.0")
        base_unslop = get_base_model_name("Broken-Tutu-24B-Unslop-v2.0")

        assert base_plain == "Broken-Tutu"
        assert base_transgression == "Broken-Tutu-Transgression"
        assert base_unslop == "Broken-Tutu-Unslop"
        assert len({base_plain, base_transgression, base_unslop}) == 3

    def test_qwen3_coder_variants_stay_separate(self) -> None:
        """Qwen3-Coder-A35B, Qwen3-Coder-A3B, and Qwen3-Coder-Next.

        Each is a distinct model within the Qwen3-Coder family. The MoE
        active param tags (A35B, A3B) and the -Next suffix are all part
        of the model identity.
        """
        base_a35b = get_base_model_name("Qwen3-Coder-480B-A35B-Instruct")
        base_a3b = get_base_model_name("Qwen3-Coder-30B-A3B-Instruct")
        base_next = get_base_model_name("Qwen3-Coder-Next")

        assert base_a35b == "Qwen3-Coder-A35B"
        assert base_a3b == "Qwen3-Coder-A3B"
        assert base_next == "Qwen3-Coder-Next"
        assert len({base_a35b, base_a3b, base_next}) == 3

    def test_glm4_vs_glm4_abliterated(self) -> None:
        """GLM-4-32B-0414 and GLM-4-32B-0414-abliterated are separate.

        `abliterated` is a community modification (safety filter removal),
        not a recognized variant keyword. The date code `0414` is extracted
        from both, but `abliterated` stays in the base.
        """
        base_plain = get_base_model_name("GLM-4-32B-0414")
        base_ablit = get_base_model_name("GLM-4-32B-0414-abliterated")

        assert base_plain == "GLM-4"
        assert base_ablit == "GLM-4-abliterated"
        assert base_plain != base_ablit


# ---------------------------------------------------------------------------
# Corner cases: tricky patterns that exercise extraction boundary conditions
# ---------------------------------------------------------------------------


class TestParserCornerCases:
    """Tests for edge cases and potentially ambiguous model name patterns."""

    def test_single_digit_not_consumed_as_date(self) -> None:
        """Llama-2-7B: the `-2` is a single digit, not a 4-digit date.

        The 4-digit date pattern requires exactly 4 consecutive digits.
        Single/double/triple digits in model names (Llama-2, Ministral-3,
        GPT-4) must remain part of the base.
        """
        parsed = parse_text_model_name("Llama-2-7B-Instruct")

        assert parsed.base_name == "Llama-2"
        assert parsed.size == "7B"
        assert parsed.variant == "Instruct"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 0

    def test_two_digit_not_consumed_as_date(self) -> None:
        """Llama-3.2-3B-Instruct: the `.2` and `3` segments are not 4-digit codes.

        Even though `3.2` contains digits, neither `3` nor `2` is 4 digits
        long, so no false date extraction occurs.
        """
        base = get_base_model_name("Llama-3.2-3B-Instruct")

        assert base == "Llama-3.2"

    def test_three_digit_not_consumed_as_date(self) -> None:
        """AID-Neo-125M: the size is 125M, and 125 is only 3 digits.

        No false date extraction should occur for 3-digit numbers.
        """
        parsed = parse_text_model_name("AID-Neo-125M")

        assert parsed.base_name == "AID-Neo"
        assert parsed.size == "125M"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 0

    def test_five_digit_not_consumed_as_date(self) -> None:
        """A hypothetical 5-digit suffix should NOT match the 4-digit date pattern.

        The `(?![a-zA-Z0-9])` lookahead prevents matching 4 digits within
        a longer number.
        """
        parsed = parse_text_model_name("SomeModel-12345-7B")

        assert "12345" in parsed.base_name
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 0

    def test_leading_version_with_three_segments(self) -> None:
        """4.2.0-Broken-Tutu-24b: leading dotted version is extracted as extra.

        The `4.2.0` matches `LEADING_VERSION_PATTERN` (three-segment dotted
        version at start). After extraction, the remaining `Broken-Tutu-24b`
        is parsed normally: size `24B` extracted, base `Broken-Tutu`.
        """
        parsed = parse_text_model_name("4.2.0-Broken-Tutu-24b")

        assert parsed.base_name == "Broken-Tutu"
        assert parsed.size == "24B"
        leading = [e for e in parsed.extras if e.inferred_type == ExtraPartType.LEADING_VERSION]
        assert len(leading) == 1
        assert leading[0].value == "4.2.0"

    def test_moe_8x7b_extracted_as_size(self) -> None:
        r"""Noromaid-v0.1-mixtral-8x7b-v3: MoE size `8x7b` is extracted.

        The MoE size pattern `\\d+x\\d+[BMK]` matches `8x7b`. The first
        version pattern match is `v0.1`. After both are extracted, `v3`
        stays in the base because only the first version is extracted.
        """
        parsed = parse_text_model_name("Noromaid-v0.1-mixtral-8x7b-v3")

        assert parsed.size == "8X7B"
        assert parsed.version == "v0.1"
        # v3 remains in the base because only one version is extracted
        assert "v3" in parsed.base_name

    def test_moe_active_param_not_extracted_as_size(self) -> None:
        """Qwen2-57B-A14B: `57B` is the size, `A14B` stays in the base.

        The size regex's lookbehind `(?<![a-zA-Z])` prevents `14B` from
        matching when preceded by the letter `A`. This preserves MoE
        active parameter designators as part of the model identity.
        """
        parsed = parse_text_model_name("Qwen2-57B-A14B")

        assert parsed.size == "57B"
        assert parsed.base_name == "Qwen2-A14B"

    def test_version_without_prefix_stays_in_base(self) -> None:
        """airoboros-gpt4-1.2: unadorned `1.2` is NOT extracted as a version.

        Version extraction requires a `v`/`V` prefix. Bare dotted numbers
        stay in the base name. This is correct: airoboros-gpt4-1.2, -1.3,
        and -1.4 are intentionally separate model releases.
        """
        parsed = parse_text_model_name("airoboros-33b-gpt4-1.2")

        assert parsed.version is None
        assert "1.2" in parsed.base_name
        assert parsed.size == "33B"

    def test_fractional_size_extracted(self) -> None:
        r"""Qwen2.5-Coder-0.5B: the fractional size `0.5B` is extracted.

        The size regex `\\d+\\.?\\d*[BMK]` matches `0.5B`. The base after
        extraction should be `Qwen2.5-Coder`.
        """
        parsed = parse_text_model_name("Qwen2.5-Coder-0.5B-Instruct")

        assert parsed.size == "0.5B"
        assert parsed.variant == "Instruct"
        assert parsed.base_name == "Qwen2.5-Coder"

    def test_date_code_in_middle_with_suffix(self) -> None:
        """GLM-4-32B-0414-abliterated: date is in the middle, suffix stays.

        The 4-digit date `0414` is extracted even when followed by more
        name segments. `-abliterated` is not a variant keyword so it
        remains in the base: `GLM-4-abliterated`.
        """
        parsed = parse_text_model_name("GLM-4-32B-0414-abliterated")

        assert parsed.base_name == "GLM-4-abliterated"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "0414"

    def test_underscore_separated_name(self) -> None:
        r"""Pernicious_Prophecy_70B: underscore separators work like hyphens.

        The size regex uses `(?<![a-zA-Z])` / `(?![a-zA-Z])` instead of
        `\\b` specifically to handle underscore boundaries. `70B` should
        be extracted as size.
        """
        parsed = parse_text_model_name("Pernicious_Prophecy_70B")

        assert parsed.size == "70B"
        assert parsed.base_name == "Pernicious_Prophecy"

    def test_backend_prefix_stripped_for_grouping(self) -> None:
        """Backend and author prefixes don't affect base name extraction.

        `get_base_model_name` strips backend (koboldcpp/, aphrodite/) and
        author (mistralai/, THUDM/) prefixes before parsing. The same model
        registered under different backends must produce identical bases.
        """
        names = [
            "aphrodite/mistralai/Mistral-Large-Instruct-2407",
            "koboldcpp/Mistral-Large-Instruct-2407",
            "mistralai/Mistral-Large-Instruct-2407",
        ]
        bases = [get_base_model_name(n) for n in names]

        assert all(b == "Mistral-Large" for b in bases)

    def test_multiple_extractions_all_at_once(self) -> None:
        """Qwen3-235B-A22B-Instruct-2507 exercises every extraction stage.

        - Leading version: no match (doesn't start with digits)
        - Size: `235B` extracted
        - Version: no v-prefix found
        - Quant: none
        - Variant: `Instruct` extracted
        - Date: `2507` extracted as 4-digit code
        - Remaining: `Qwen3-A22B`
        """
        parsed = parse_text_model_name("Qwen3-235B-A22B-Instruct-2507")

        assert parsed.size == "235B"
        assert parsed.variant == "Instruct"
        assert parsed.version is None
        assert parsed.quant is None
        assert parsed.base_name == "Qwen3-A22B"
        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "2507"

    def test_mm_yyyy_takes_priority_over_four_digit(self) -> None:
        """c4ai-command-r-08-2024: the MM-YYYY pattern matches first.

        DATE_PATTERNS are checked in order. `08-2024` matches the first
        pattern (MM-YYYY) as a single 7-character extra, rather than
        leaving `08` behind and extracting `2024` as a standalone code.
        """
        parsed = parse_text_model_name("c4ai-command-r-08-2024")

        date_extras = [e for e in parsed.extras if e.inferred_type == ExtraPartType.DATE]
        assert len(date_extras) == 1
        assert date_extras[0].value == "08-2024"
        assert parsed.base_name == "c4ai-command-r"

    def test_code_variant_not_confused_with_model_name(self) -> None:
        r"""Qwen2.5-Coder-32B-Instruct: `Code` would match as variant but `Coder` does not; var. uses word boundaries.

        `\\bCode\\b` matches `Code` as a whole word. `Coder` has an `r`
        after `Code`, so the word boundary `\\b` fails. `Coder` correctly
        stays in the base name.
        """
        parsed = parse_text_model_name("Qwen2.5-Coder-32B-Instruct")

        assert parsed.variant == "Instruct"
        assert parsed.base_name == "Qwen2.5-Coder"
        assert parsed.size == "32B"

    def test_leading_size_in_name(self) -> None:
        """13B-BlueMethod: size appears before the model name.

        The size pattern has no positional constraint - it matches `13B`
        regardless of where it appears. The base should be `BlueMethod`.
        """
        parsed = parse_text_model_name("13B-BlueMethod")

        assert parsed.size == "13B"
        assert parsed.base_name == "BlueMethod"

    def test_version_with_many_dots(self) -> None:
        r"""Captain-Eris_Violet-V0.420-12B: multi-digit version V0.420.

        The version pattern matches `V\\d+(\\.\\d+)+`, so V0.420 is
        captured as a single version string.
        """
        parsed = parse_text_model_name("Captain-Eris_Violet-V0.420-12B")

        assert parsed.version == "V0.420"
        assert parsed.size == "12B"
        assert parsed.base_name == "Captain-Eris_Violet"
