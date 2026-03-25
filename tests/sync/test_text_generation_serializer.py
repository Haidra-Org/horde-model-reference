"""Tests for the text generation CSV-mediated serialization pipeline.

These tests validate that the serializer produces output matching the upstream
convert.py's behavior. The test strategy avoids tautological repetition of the
implementation by using independent reference implementations, known-good fixture
data, and semantic property checks.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference.sync.text_generation_serializer import (
    TEXT_CSV_FIELDNAMES,
    TextGenerationSerializer,
    _format_parameters_bn,
)

# ---------------------------------------------------------------------------
# Reference implementation: a standalone transliteration of convert.py logic.
# This is intentionally NOT shared with the serializer code. Tests compare
# the serializer's output against this independent reference.
# ---------------------------------------------------------------------------


def _convert_py_reference(
    csv_rows: list[dict[str, str]],
    defaults: dict[str, Any],
    generation_params: dict[str, Any],
) -> dict[str, Any]:
    """Independent reference implementation of upstream convert.py.

    Transliterated directly from the upstream repository's convert.py.
    Used as a comparison oracle — the serializer must produce identical output.
    """
    data: dict[str, Any] = {}

    for csv_row in csv_rows:
        row: dict[str, Any] = dict(csv_row)
        name = row.pop("name")
        model_name = name.split("/")[1] if "/" in name else name

        params_str = row.pop("parameters_bn")
        params_f = float(params_str)
        row["parameters"] = int(params_f * 1_000_000_000)

        tags = set([t.strip() for t in row["tags"].split(",")] if row["tags"] else [])
        if style := row.get("style"):
            tags.add(style)
        tags.add(f"{round(params_f, 0):.0f}B")
        row["tags"] = sorted(tags)

        row["settings"] = json.loads(row["settings"]) if row["settings"] else {}

        if not row.get("display_name"):
            row["display_name"] = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

        row = {k: v for k, v in row.items() if v}

        for key_format in ["{name}", "aphrodite/{name}", "koboldcpp/{model_name}"]:
            key = key_format.format(name=name, model_name=model_name)
            data[key] = {"name": key, "model_name": model_name, **defaults, **row}

    return data


@pytest.fixture()
def serializer() -> TextGenerationSerializer:
    """Create a fresh serializer instance."""
    return TextGenerationSerializer()


@pytest.fixture()
def sample_csv_rows() -> list[dict[str, str]]:
    """CSV rows covering diverse field combinations.

    Includes: full fields, minimal fields, fractional params, instruct_format,
    settings, custom display_name, multiple tags.
    """
    return [
        {
            "name": "acrastt/Marx-3B-V3",
            "parameters_bn": "3",
            "display_name": "Marx 3B V3",
            "url": "https://huggingface.co/acrastt/Marx-3B-V3",
            "baseline": "StableLM-3B-4E1T",
            "description": "StableLM 3B finetuned on EverythingLM Data V3.",
            "style": "",
            "tags": "",
            "instruct_format": "",
            "settings": '{"temperature": 0.7, "top_p": 0.1}',
        },
        {
            "name": "anthracite-org/magnum-12b-v2",
            "parameters_bn": "12",
            "display_name": "Magnum 12B V2",
            "url": "https://huggingface.co/anthracite-org/magnum-v2-12b",
            "baseline": "mistralai/Mistral-Nemo-Base-2407",
            "description": "Very diverse and creative model.",
            "style": "roleplay",
            "tags": "popular,story",
            "instruct_format": "ChatML",
            "settings": '{"temperature": 0.75, "min_p": 0.1}',
        },
        {
            "name": "Aeala/Enterredaas-33b",
            "parameters_bn": "33",
            "display_name": "",
            "url": "https://huggingface.co/Aeala/Enterredaas-33b",
            "baseline": "",
            "description": "",
            "style": "",
            "tags": "",
            "instruct_format": "Long Alpaca",
            "settings": "",
        },
        {
            "name": "smallmodel/tiny",
            "parameters_bn": "0.56",
            "display_name": "",
            "url": "",
            "baseline": "",
            "description": "",
            "style": "",
            "tags": "",
            "instruct_format": "",
            "settings": "",
        },
    ]


@pytest.fixture()
def sample_primary_records() -> dict[str, dict[str, Any]]:
    """PRIMARY API-style records (base models, no backend prefixes).

    These represent what the sync script fetches from the v1 API.
    """
    return {
        "acrastt/Marx-3B-V3": {
            "name": "acrastt/Marx-3B-V3",
            "model_name": "Marx-3B-V3",
            "parameters": 3_000_000_000,
            "baseline": "StableLM-3B-4E1T",
            "description": "StableLM 3B finetuned on EverythingLM Data V3.",
            "version": "1",
            "style": "generalist",
            "nsfw": False,
            "display_name": "Marx 3B V3",
            "url": "https://huggingface.co/acrastt/Marx-3B-V3",
            "tags": ["3B"],
            "settings": {"temperature": 0.7, "top_p": 0.1},
        },
        "anthracite-org/magnum-12b-v2": {
            "name": "anthracite-org/magnum-12b-v2",
            "model_name": "magnum-12b-v2",
            "parameters": 12_000_000_000,
            "baseline": "mistralai/Mistral-Nemo-Base-2407",
            "description": "Very diverse and creative model.",
            "version": "1",
            "style": "roleplay",
            "nsfw": False,
            "display_name": "Magnum 12B V2",
            "url": "https://huggingface.co/anthracite-org/magnum-v2-12b",
            "tags": ["12B", "popular", "roleplay", "story"],
            "instruct_format": "ChatML",
            "settings": {"temperature": 0.75, "min_p": 0.1},
        },
        "Aeala/Enterredaas-33b": {
            "name": "Aeala/Enterredaas-33b",
            "model_name": "Enterredaas-33b",
            "parameters": 33_000_000_000,
            "baseline": "",
            "description": "",
            "version": "1",
            "style": "generalist",
            "nsfw": False,
            "display_name": "Enterredaas 33b",
            "url": "https://huggingface.co/Aeala/Enterredaas-33b",
            "tags": ["33B"],
            "instruct_format": "Long Alpaca",
        },
    }


class TestForwardConversion:
    """Verify the forward conversion produces output identical to convert.py."""

    def test_forward_convert_matches_convert_py_reference(
        self,
        serializer: TextGenerationSerializer,
        sample_csv_rows: list[dict[str, str]],
    ) -> None:
        """The serializer's forward conversion must produce byte-identical JSON to convert.py.

        Uses an independent reference implementation (not shared code) as oracle.
        This is the single most important correctness test.
        """
        reference_output = _convert_py_reference(
            csv_rows=sample_csv_rows,
            defaults=dict(serializer._defaults),
            generation_params=dict(serializer._generation_params),
        )

        serializer_output = serializer._forward_convert(sample_csv_rows)

        reference_json = json.dumps(reference_output, indent=4) + "\n"
        serializer_json = json.dumps(serializer_output, indent=4) + "\n"

        assert serializer_json == reference_json, (
            "Forward conversion output differs from convert.py reference. "
            "This means db.json would not match the upstream."
        )

    def test_db_json_field_ordering(
        self,
        serializer: TextGenerationSerializer,
        sample_csv_rows: list[dict[str, str]],
    ) -> None:
        """Each record's keys must follow the exact order produced by convert.py's dict merge.

        The order is: name, model_name, {defaults keys}, {remaining row-only keys}.
        """
        db_dict = serializer._forward_convert(sample_csv_rows)

        defaults_keys = list(serializer._defaults.keys())

        for entry_name, record in db_dict.items():
            keys = list(record.keys())
            assert keys[0] == "name", f"{entry_name}: first key must be 'name'"
            assert keys[1] == "model_name", f"{entry_name}: second key must be 'model_name'"

            # Keys 2..N should start with defaults keys in order
            remaining_keys = keys[2:]
            defaults_position = 0
            for key in remaining_keys:
                if defaults_position < len(defaults_keys) and key == defaults_keys[defaults_position]:
                    defaults_position += 1

            assert defaults_position == len(defaults_keys), (
                f"{entry_name}: defaults keys not in expected order. "
                f"Expected {defaults_keys} at positions 2+, got {remaining_keys}"
            )

    def test_three_entries_per_model(
        self,
        serializer: TextGenerationSerializer,
    ) -> None:
        """Each base model must produce exactly 3 entries: base, aphrodite/, koboldcpp/."""
        csv_rows = [
            {
                "name": "org/Model-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        db_dict = serializer._forward_convert(csv_rows)

        assert "org/Model-7B" in db_dict
        assert "aphrodite/org/Model-7B" in db_dict
        assert "koboldcpp/Model-7B" in db_dict
        assert len(db_dict) == 3

        # model_name is consistent across all entries
        for record in db_dict.values():
            assert record["model_name"] == "Model-7B"

        # name field matches the entry key
        for key, record in db_dict.items():
            assert record["name"] == key

    def test_defaults_always_present(
        self,
        serializer: TextGenerationSerializer,
        sample_csv_rows: list[dict[str, str]],
    ) -> None:
        """Every record must contain all defaults.json keys regardless of CSV content."""
        db_dict = serializer._forward_convert(sample_csv_rows)
        defaults_keys = set(serializer._defaults.keys())

        for entry_name, record in db_dict.items():
            missing = defaults_keys - set(record.keys())
            assert not missing, f"{entry_name}: missing defaults keys: {missing}"

    def test_empty_settings_not_in_output(
        self,
        serializer: TextGenerationSerializer,
    ) -> None:
        """Empty settings dict is falsy and stripped by convert.py's empty-value filter."""
        csv_rows = [
            {
                "name": "org/NoSettings-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        db_dict = serializer._forward_convert(csv_rows)

        for record in db_dict.values():
            assert "settings" not in record, "Empty settings should be absent from output"

    def test_instruct_format_in_output(
        self,
        serializer: TextGenerationSerializer,
    ) -> None:
        """instruct_format field appears in db.json when present in CSV."""
        csv_rows = [
            {
                "name": "org/Chat-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "ChatML",
                "settings": "",
            },
        ]

        db_dict = serializer._forward_convert(csv_rows)

        for record in db_dict.values():
            assert record.get("instruct_format") == "ChatML"


# ---------------------------------------------------------------------------
# CSV reverse conversion
# ---------------------------------------------------------------------------


class TestReverseConversion:
    """Verify PRIMARY records are correctly converted to CSV row format."""

    def test_parameters_bn_conversion(self, serializer: TextGenerationSerializer) -> None:
        """Parameters integer is converted to minimal billions string."""
        cases = [
            (7_000_000_000, "7"),
            (560_000_000, "0.56"),
            (123_000_000_000, "123"),
            (3_000_000_000, "3"),
            (12_000_000_000, "12"),
        ]
        for params, expected_bn in cases:
            record: dict[str, Any] = {"parameters": params}
            row = serializer._record_to_csv_row(name="test/model", record=record)
            assert row["parameters_bn"] == expected_bn, (
                f"parameters={params} should produce parameters_bn='{expected_bn}', got '{row['parameters_bn']}'"
            )

    def test_auto_generated_tags_stripped(self, serializer: TextGenerationSerializer) -> None:
        """Style and size tags are auto-generated by convert.py, so they must not appear in CSV."""
        record: dict[str, Any] = {
            "parameters": 3_000_000_000,
            "style": "roleplay",
            "tags": ["3B", "roleplay", "story", "popular"],
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)

        csv_tags = {t.strip() for t in row["tags"].split(",") if t.strip()}
        assert "3B" not in csv_tags, "Size tag should be stripped"
        assert "roleplay" not in csv_tags, "Style tag should be stripped"
        assert csv_tags == {"story", "popular"}

    def test_auto_generated_display_name_omitted(self, serializer: TextGenerationSerializer) -> None:
        """display_name matching auto-generated value should be empty in CSV."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "display_name": "My Model 7B",
        }
        row = serializer._record_to_csv_row(name="org/My-Model-7B", record=record)
        assert row["display_name"] == "", "Auto-generated display_name should be omitted"

    def test_custom_display_name_preserved(self, serializer: TextGenerationSerializer) -> None:
        """display_name that differs from auto-generated should be kept."""
        record: dict[str, Any] = {
            "parameters": 12_000_000_000,
            "display_name": "Magnum 12B V2",
        }
        row = serializer._record_to_csv_row(name="anthracite-org/magnum-12b-v2", record=record)
        assert row["display_name"] == "Magnum 12B V2"

    def test_default_only_generalist_style_stripped_in_csv(self, serializer: TextGenerationSerializer) -> None:
        """A style matching the default that was never explicitly set (not in tags) must be stripped.

        convert.py adds explicit styles to tags before applying defaults. So style="generalist"
        with "generalist" absent from tags means it was only injected by defaults.json — writing
        it to CSV would cause the next forward conversion to add a spurious "generalist" tag.
        """
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "style": "generalist",
            "tags": ["7B"],
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)
        assert row["style"] == "", "Default-only style should be stripped to prevent tag leakage"

    def test_explicit_generalist_style_preserved_in_csv(self, serializer: TextGenerationSerializer) -> None:
        """An explicitly set generalist style (present in tags) must be preserved in CSV."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "style": "generalist",
            "tags": ["7B", "generalist"],
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)
        assert row["style"] == "generalist", "Explicit generalist style must be preserved"

    def test_generalist_tag_survives_roundtrip(self, serializer: TextGenerationSerializer) -> None:
        """A model with explicit style=generalist must have 'generalist' in its db.json tags after round-trip."""
        records: dict[str, dict[str, Any]] = {
            "org/model": {
                "parameters": 27_000_000_000,
                "style": "generalist",
                "tags": ["27B", "generalist"],
            },
        }
        artifacts = serializer.serialize(primary_base_records=records)
        db = json.loads(artifacts.json_content)
        base_entry = db["org/model"]
        assert "generalist" in base_entry["tags"], "generalist tag must survive round-trip"
        assert base_entry.get("style") == "generalist"

    def test_no_style_model_does_not_acquire_generalist_tag(self, serializer: TextGenerationSerializer) -> None:
        """A model with no explicit style must not acquire a 'generalist' tag after roundtrip.

        This is the core regression test. The pipeline:
        1. PRIMARY has style="generalist" from defaults, tags=["3B"] (no "generalist" tag)
        2. Reverse to CSV: style must be stripped to ""
        3. Forward from CSV: defaults add style="generalist", tags stay ["3B"]
        """
        records: dict[str, dict[str, Any]] = {
            "acrastt/Marx-3B-V3": {
                "parameters": 3_000_000_000,
                "style": "generalist",
                "tags": ["3B"],
            },
        }
        artifacts = serializer.serialize(primary_base_records=records)
        db = json.loads(artifacts.json_content)
        base_entry = db["acrastt/Marx-3B-V3"]
        assert "generalist" not in base_entry["tags"], (
            "Model without explicit generalist style must not acquire generalist tag"
        )
        # Style is still present from defaults (forward conversion always adds it)
        assert base_entry.get("style") == "generalist"

    def test_multi_pass_stability_no_tag_accumulation(self, serializer: TextGenerationSerializer) -> None:
        """Multiple roundtrips must not accumulate tags. Tags must be identical after pass 1 and 2."""
        records: dict[str, dict[str, Any]] = {
            "acrastt/Marx-3B-V3": {
                "parameters": 3_000_000_000,
                "style": "generalist",
                "tags": ["3B"],
            },
            "Aeala/Enterredaas-33b": {
                "parameters": 33_000_000_000,
                "style": "generalist",
                "tags": ["33B"],
            },
        }

        # Pass 1
        art1 = serializer.serialize(primary_base_records=records)
        db1 = json.loads(art1.json_content)

        # Parse pass 1 CSV back and re-serialize (simulating the next sync cycle)
        reader = csv.DictReader(io.StringIO(art1.csv_content))
        csv_rows_pass1 = list(reader)
        db2 = serializer._forward_convert(csv_rows_pass1)

        for name in records:
            tags1 = db1[name]["tags"]
            tags2 = db2[name]["tags"]
            assert tags1 == tags2, f"Tags changed for {name} between passes: {tags1} → {tags2}"

    def test_instruct_format_preserved_in_csv(self, serializer: TextGenerationSerializer) -> None:
        """instruct_format is passed through to CSV."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "instruct_format": "ChatML",
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)
        assert row["instruct_format"] == "ChatML"

    def test_v2_only_fields_ignored(self, serializer: TextGenerationSerializer) -> None:
        """Fields that only exist in the v2 internal format should not appear in CSV."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "config": {"download": []},
            "metadata": {"schema_version": "1"},
            "model_classification": {"domain": "text", "purpose": "generation"},
            "record_type": "text_generation",
            "text_model_group": "llama2",
            "version": "1",
            "nsfw": False,
            "model_name": "model",
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)
        csv_keys = set(row.keys())
        assert csv_keys == set(TEXT_CSV_FIELDNAMES), (
            f"CSV row should only contain CSV fields, got extra: {csv_keys - set(TEXT_CSV_FIELDNAMES)}"
        )

    def test_settings_serialized_to_json_string(self, serializer: TextGenerationSerializer) -> None:
        """Settings dict is serialized as a JSON string in CSV."""
        settings = {"temperature": 0.7, "top_p": 0.1}
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "settings": settings,
        }
        row = serializer._record_to_csv_row(name="org/model", record=record)
        parsed_back = json.loads(row["settings"])
        assert parsed_back == settings


class TestApplyChanges:
    """Verify that merging PRIMARY changes into existing CSV preserves ordering."""

    def test_preserves_unchanged_row_order(self, serializer: TextGenerationSerializer) -> None:
        """Existing row order is kept; removed models are dropped; new models appended."""
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "3",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            {
                "name": "org/B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            {
                "name": "org/C",
                "parameters_bn": "13",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "3",
                "display_name": "Updated A",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            "org/B": {
                "name": "org/B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            "org/D": {
                "name": "org/D",
                "parameters_bn": "24",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)

        names = [row["name"] for row in result]
        # org/C is preserved from existing CSV even though it's absent from PRIMARY
        assert names == ["org/A", "org/B", "org/C", "org/D"], f"Expected [A, B, C, D], got {names}"
        assert result[0]["display_name"] == "Updated A", "A should be updated"

    def test_merge_preserves_instruct_format(self, serializer: TextGenerationSerializer) -> None:
        """Existing CSV instruct_format is kept when PRIMARY doesn't provide it."""
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "llama",
                "description": "A model",
                "style": "",
                "tags": "",
                "instruct_format": "alpaca",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "llama",
                "description": "A model",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        assert result[0]["instruct_format"] == "alpaca", "instruct_format should be preserved from existing CSV"

    def test_merge_preserves_absent_models(self, serializer: TextGenerationSerializer) -> None:
        """Models in existing CSV but absent from PRIMARY are preserved during transition."""
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "ChatML",
                "settings": "",
            },
            {
                "name": "org/B",
                "parameters_bn": "13",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        names = [row["name"] for row in result]
        assert "org/B" in names, "Model absent from PRIMARY should be preserved"
        assert len(result) == 2

    def test_merge_updates_modified_fields(self, serializer: TextGenerationSerializer) -> None:
        """PRIMARY non-empty values overwrite existing CSV values."""
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "Old description",
                "style": "",
                "tags": "",
                "instruct_format": "alpaca",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "https://example.com",
                "baseline": "llama",
                "description": "New description",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        merged = result[0]
        assert merged["description"] == "New description", "PRIMARY non-empty value should win"
        assert merged["url"] == "https://example.com", "PRIMARY non-empty value should win"
        assert merged["baseline"] == "llama", "PRIMARY non-empty value should win"
        assert merged["instruct_format"] == "alpaca", "Empty PRIMARY value should fall back to existing"

    def test_all_new_models_when_no_existing_csv(self, serializer: TextGenerationSerializer) -> None:
        """When no existing CSV exists, all PRIMARY records appear in insertion order."""
        primary_csv_rows = {
            "org/X": {
                "name": "org/X",
                "parameters_bn": "3",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            "org/Y": {
                "name": "org/Y",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=[], primary_csv_rows=primary_csv_rows)

        names = [row["name"] for row in result]
        assert names == ["org/X", "org/Y"]


# ---------------------------------------------------------------------------
# End-to-end serialization
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline: PRIMARY records → serialize() → CSV + JSON."""

    def test_serialize_produces_valid_csv_and_json(
        self,
        serializer: TextGenerationSerializer,
        sample_primary_records: dict[str, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """The serialize() entry point produces well-formed CSV and JSON."""
        artifacts = serializer.serialize(
            primary_base_records=sample_primary_records,
            existing_csv_path=None,
        )

        # CSV is parseable
        reader = csv.DictReader(io.StringIO(artifacts.csv_content))
        csv_rows = list(reader)
        assert len(csv_rows) == len(sample_primary_records)

        # JSON is parseable
        db_dict = json.loads(artifacts.json_content)
        assert len(db_dict) == len(sample_primary_records) * 3

    def test_serialize_with_existing_csv(
        self,
        serializer: TextGenerationSerializer,
        sample_primary_records: dict[str, dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """serialize() reads existing CSV and preserves row order."""
        csv_path = tmp_path / "models.csv"
        header = "name,parameters_bn,display_name,url,baseline,description,style,tags,instruct_format,settings"
        row_1 = "Aeala/Enterredaas-33b,33,,https://huggingface.co/Aeala/Enterredaas-33b,,,,,Long Alpaca,"
        row_2 = "acrastt/Marx-3B-V3,3,Marx 3B V3,https://huggingface.co/acrastt/Marx-3B-V3,StableLM-3B-4E1T,,"
        csv_path.write_text(f"{header}\n{row_1}\n{row_2}\n", encoding="utf-8")

        artifacts = serializer.serialize(
            primary_base_records=sample_primary_records,
            existing_csv_path=csv_path,
        )

        reader = csv.DictReader(io.StringIO(artifacts.csv_content))
        csv_rows = list(reader)
        names = [row["name"] for row in csv_rows]

        # Existing order preserved: Aeala first, then acrastt, then new model appended
        assert names[0] == "Aeala/Enterredaas-33b"
        assert names[1] == "acrastt/Marx-3B-V3"

    def test_csv_roundtrip_produces_identical_json(
        self,
        serializer: TextGenerationSerializer,
        sample_primary_records: dict[str, dict[str, Any]],
    ) -> None:
        """CSV→JSON is idempotent: serializing twice from the same data gives the same result."""
        artifacts_1 = serializer.serialize(
            primary_base_records=sample_primary_records,
            existing_csv_path=None,
        )

        # Parse the CSV back and re-forward-convert
        reader = csv.DictReader(io.StringIO(artifacts_1.csv_content))
        csv_rows = list(reader)
        db_dict_2 = serializer._forward_convert(csv_rows)
        json_2 = json.dumps(db_dict_2, indent=4) + "\n"

        assert json_2 == artifacts_1.json_content, "Round-trip through CSV must produce identical JSON"

    def test_backend_prefixed_input_stripped(
        self,
        serializer: TextGenerationSerializer,
    ) -> None:
        """Backend-prefixed entries in input are ignored; only base records are serialized."""
        records: dict[str, dict[str, Any]] = {
            "org/Model-7B": {
                "parameters": 7_000_000_000,
            },
            "aphrodite/org/Model-7B": {
                "parameters": 7_000_000_000,
            },
            "koboldcpp/Model-7B": {
                "parameters": 7_000_000_000,
            },
        }

        artifacts = serializer.serialize(
            primary_base_records=records,
            existing_csv_path=None,
        )

        reader = csv.DictReader(io.StringIO(artifacts.csv_content))
        csv_rows = list(reader)
        assert len(csv_rows) == 1, "Only the base record should be in CSV"
        assert csv_rows[0]["name"] == "org/Model-7B"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


class TestCSVLineEndingsFix:
    r"""Verify CSV output uses LF endings, not the CRLF that csv.DictWriter emits.

    Python's csv module always writes ``\r\n`` as the record terminator
    (per RFC 4180), regardless of platform. The upstream GitHub repos store
    CSV with git-normalized LF endings. If our output contains CRLF, every
    line appears as changed in the diff (``^M`` artefact), obscuring real
    changes and bloating PRs.

    The fix is a ``.replace("\r\n", "\n")`` on the rendered CSV string.
    These tests verify the fix at both the serializer level and the
    end-to-end level.
    """

    def test_render_csv_uses_lf(self, serializer: TextGenerationSerializer) -> None:
        r"""_render_csv output must use LF, never CRLF.

        We check for ``\r`` anywhere in the output rather than just
        ``\r\n`` — there is no legitimate reason for ``\r`` to appear
        in model reference CSV data.
        """
        csv_rows = [
            {
                "name": "org/Model-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]
        result = serializer._render_csv(csv_rows)
        assert "\r" not in result, "CSV output must not contain any carriage returns"
        assert "\n" in result, "CSV output must contain at least one newline"

    def test_end_to_end_csv_uses_lf(
        self,
        serializer: TextGenerationSerializer,
        sample_primary_records: dict[str, dict[str, Any]],
    ) -> None:
        """The full serialize() pipeline must produce LF-only CSV content.

        This catches regressions where the fix is applied in _render_csv
        but a later step (e.g., file write or string concatenation)
        reintroduces CRLF.
        """
        artifacts = serializer.serialize(
            primary_base_records=sample_primary_records,
            existing_csv_path=None,
        )
        assert "\r" not in artifacts.csv_content


class TestTagMergeFix:
    """Verify that empty PRIMARY tags overwrite existing CSV tags.

    The serializer strips auto-generated tags (style + size bucket) before
    writing to CSV. For a model whose tags are ALL auto-generated, the
    stripped result is ``""``. Under the old merge rule ("PRIMARY wins if
    non-empty"), this empty string was falsy, so the merge fell back to
    the existing CSV's tags — which could contain user-added tags like
    ``"popular"`` that were never in the PRIMARY data.

    The fix introduces ``_PRIMARY_AUTHORITATIVE_FIELDS``: a set of fields
    where the serializer always produces a definitive value. For these
    fields, even an empty PRIMARY value overwrites the existing CSV.

    ``instruct_format`` is intentionally NOT in this set because it is
    CSV-only metadata that PRIMARY may not carry.
    """

    def test_empty_primary_tags_overwrite_existing(self, serializer: TextGenerationSerializer) -> None:
        """When PRIMARY says tags is empty, existing CSV 'popular' must not leak through.

        This is the core regression: a model with only auto-generated tags
        (e.g., "7B", "chat") gets stripped to tags="". If merge falls back
        to existing CSV, "popular" re-appears in the output db.json even
        though PRIMARY never had it.
        """
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "chat",
                "tags": "popular",
                "instruct_format": "",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "chat",
                "tags": "",  # All auto-tags stripped → empty
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        assert result[0]["tags"] == "", (
            "Empty PRIMARY tags must overwrite existing CSV tags. "
            "If this fails, 'popular' is leaking through the merge."
        )

    def test_instruct_format_still_preserved_from_existing(self, serializer: TextGenerationSerializer) -> None:
        """instruct_format must still fall back to existing CSV when PRIMARY is empty.

        This is the counterpart to the tags test: instruct_format is
        CSV-only metadata that PRIMARY genuinely may not carry. The fix
        must NOT break this preservation.
        """
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "ChatML",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        assert result[0]["instruct_format"] == "ChatML", (
            "instruct_format must be preserved from existing CSV when PRIMARY is empty"
        )

    def test_nonempty_primary_tags_still_win(self, serializer: TextGenerationSerializer) -> None:
        """Non-empty PRIMARY tags must still overwrite existing CSV tags.

        Sanity check that the fix didn't break the normal case where
        PRIMARY has real tag data.
        """
        existing = [
            {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "old_tag",
                "instruct_format": "",
                "settings": "",
            },
        ]

        primary_csv_rows = {
            "org/A": {
                "name": "org/A",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "new_tag",
                "instruct_format": "",
                "settings": "",
            },
        }

        result = serializer._apply_changes(existing_rows=existing, primary_csv_rows=primary_csv_rows)
        assert result[0]["tags"] == "new_tag"

    def test_popular_tag_absent_after_full_roundtrip(
        self,
        serializer: TextGenerationSerializer,
        tmp_path: Path,
    ) -> None:
        """End-to-end: a model with only auto-tags must NOT gain 'popular' from existing CSV.

        Reproduces the full pipeline: PRIMARY record → serialize with
        existing CSV containing 'popular' → verify db.json does NOT
        have 'popular' in the output tags.

        This is the definitive regression test for the popular-tag leak.
        """
        # PRIMARY record: tags are ["7B", "chat"] (all auto-generated)
        primary_records: dict[str, dict[str, Any]] = {
            "org/Model-7B": {
                "parameters": 7_000_000_000,
                "style": "chat",
                "tags": ["7B", "chat"],
                "display_name": "Model 7B",
            },
        }

        # Existing CSV has "popular" in tags
        csv_path = tmp_path / "models.csv"
        csv_path.write_text(
            "name,parameters_bn,display_name,url,baseline,description,style,tags,instruct_format,settings\n"
            "org/Model-7B,7,,,,,,popular,,\n",
            encoding="utf-8",
        )

        artifacts = serializer.serialize(
            primary_base_records=primary_records,
            existing_csv_path=csv_path,
        )

        db = json.loads(artifacts.json_content)
        base_tags = db["org/Model-7B"]["tags"]
        assert "popular" not in base_tags, f"'popular' must NOT appear in tags after roundtrip. Got: {base_tags}"


class TestURLNameGuard:
    """Verify that URL-shaped model names are rejected in forward conversion.

    The upstream convert.py uses ``name.split("/")[1]`` to derive model_name.
    For ``"https://huggingface.co/Org/Model"``, this produces ``""``
    because ``split("/")`` gives ``["https:", "", "huggingface.co", ...]``
    and index 1 is the empty segment. This creates:

    - A base key that is the full URL (wrong)
    - ``"koboldcpp/"`` with no model identifier (collides with other models)

    These tests verify the guard that skips URL-shaped names with a warning.
    """

    def test_url_name_skipped_in_forward_convert(self, serializer: TextGenerationSerializer) -> None:
        """_forward_convert must skip rows whose name contains ``://``.

        Without this guard, the URL becomes a top-level key in db.json
        and the koboldcpp entry is just ``"koboldcpp/"``.
        """
        csv_rows = [
            {
                "name": "https://huggingface.co/Org/Model",
                "parameters_bn": "7",
                "display_name": "",
                "url": "https://huggingface.co/Org/Model",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
            {
                "name": "Org/Good-Model-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        result = serializer._forward_convert(csv_rows)

        # URL entry must be absent
        assert not any("huggingface" in key for key in result), (
            f"URL-shaped name must be skipped. Keys: {list(result.keys())}"
        )
        assert "koboldcpp/" not in result, "Empty koboldcpp/ key must not exist"

        # Normal entry must be present
        assert "Org/Good-Model-7B" in result
        assert "koboldcpp/Good-Model-7B" in result

    def test_url_name_skipped_in_end_to_end(self, serializer: TextGenerationSerializer) -> None:
        """Full pipeline: if PRIMARY has a URL-named record, it must not appear in output.

        This can happen if a model was added to PRIMARY via the API with
        the URL as the model name by mistake.
        """
        primary_records: dict[str, dict[str, Any]] = {
            "https://huggingface.co/Org/Model": {
                "parameters": 7_000_000_000,
                "url": "https://huggingface.co/Org/Model",
            },
            "Org/Good-Model-7B": {
                "parameters": 7_000_000_000,
            },
        }

        artifacts = serializer.serialize(
            primary_base_records=primary_records,
            existing_csv_path=None,
        )

        db = json.loads(artifacts.json_content)
        assert "koboldcpp/" not in db
        assert not any("huggingface" in key for key in db)
        assert "Org/Good-Model-7B" in db


class TestFormatParametersBn:
    """Verify the minimal float formatter."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (3.0, "3"),
            (0.56, "0.56"),
            (123.0, "123"),
            (7.0, "7"),
            (1.5, "1.5"),
            (0.1, "0.1"),
        ],
    )
    def test_format(self, value: float, expected: str) -> None:
        """Whole numbers have no decimal; fractional numbers preserve precision."""
        assert _format_parameters_bn(value) == expected


# ---------------------------------------------------------------------------
# Cross-validation with upstream repo data (optional, skipped if not available)
# ---------------------------------------------------------------------------


class TestUpstreamCrossValidation:
    """Validate our forward conversion against the actual upstream db.json.

    These tests read the real models.csv and db.json from the upstream repo
    and verify our serializer produces identical output. Skipped in CI where
    the upstream repo isn't checked out.
    """

    def test_forward_convert_matches_upstream_db_json(self) -> None:
        """Our forward conversion of the upstream models.csv must produce the upstream db.json."""
        from horde_model_reference import horde_model_reference_settings

        remote_repo_db_file = horde_model_reference_settings.text_github_repo.compose_full_file_url("db.json")
        remote_repo_csv_file = horde_model_reference_settings.text_github_repo.compose_full_file_url("models.csv")

        import requests

        try:
            response_db = requests.get(remote_repo_db_file)
            response_csv = requests.get(remote_repo_csv_file)
            response_db.raise_for_status()
            response_csv.raise_for_status()
        except Exception as e:
            pytest.skip(f"Upstream repo not accessible: {e}")

        upstream_json = response_db.text
        upstream_csv = response_csv.text

        serializer = TextGenerationSerializer()

        rows: list[dict[str, str]] = []
        reader = csv.DictReader(io.StringIO(upstream_csv))
        for row in reader:
            rows.append(row)

        db_dict = serializer._forward_convert(rows)
        our_json = json.dumps(db_dict, indent=4) + "\n"

        assert our_json == upstream_json, (
            "Forward conversion of upstream models.csv does not match upstream db.json. "
            "This means our convert.py replication has a bug."
        )
