"""Tests for the shared csv_rows_to_legacy_dict conversion function.

Verifies that the canonical CSV-to-legacy-dict conversion replicates convert.py exactly:
field ordering, defaults.json merging, instruct_format preservation, empty-value
filtering, tag generation, and backend prefix duplication.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference.legacy.text_csv_utils import (
    TextCSVRow,
    csv_rows_to_legacy_dict,
    legacy_record_to_csv_row,
    parse_legacy_text_csv,
    parse_legacy_text_csv_file,
    write_legacy_text_csv,
)
from horde_model_reference.text_model_write_processor import TextModelWriteProcessor, _get_defaults


def _make_row(
    *,
    name: str = "TestOrg/TestModel-7B",
    parameters_bn: float = 7.0,
    style: str = "",
    tags: list[str] | None = None,
    instruct_format: str = "",
    settings: dict[str, Any] | None = None,
    display_name: str = "",
    url: str = "",
    baseline: str = "",
    description: str = "",
) -> TextCSVRow:
    """Build a TextCSVRow with sensible defaults for testing."""
    return TextCSVRow(
        name=name,
        parameters_bn=parameters_bn,
        parameters=int(parameters_bn * 1_000_000_000),
        style=style,
        tags=tags or [],
        instruct_format=instruct_format,
        settings=settings,
        display_name=display_name,
        url=url,
        baseline=baseline,
        description=description,
        version="",
        nsfw=False,
    )


def _convert_py_reference(
    csv_rows: list[dict[str, str]],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Replicate upstream convert.py as an independent test oracle.

    This is a direct transliteration of the upstream convert.py logic,
    used as a test oracle to verify our shared function.
    """
    data: dict[str, Any] = {}

    for csv_row in csv_rows:
        row = dict(csv_row)
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


class TestFieldOrdering:
    """Verify field ordering matches convert.py's {**defaults, **row} merge."""

    def test_field_order_matches_convert_py(self) -> None:
        """Verify name, model_name, then defaults keys appear in order."""
        row = _make_row(
            style="chat",
            url="https://example.com",
            baseline="llama",
            description="A model",
            instruct_format="alpaca",
            settings={"temperature": 0.7},
        )
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=True)
        record = result["TestOrg/TestModel-7B"]

        keys = list(record.keys())
        assert keys[0] == "name"
        assert keys[1] == "model_name"
        defaults = _get_defaults()
        defaults_keys = list(defaults.keys())
        for i, dk in enumerate(defaults_keys):
            assert keys[2 + i] == dk, f"Expected defaults key {dk} at position {2 + i}, got {keys[2 + i]}"

    def test_row_keys_override_defaults(self) -> None:
        """Verify CSV row values override defaults.json values."""
        row = _make_row(baseline="llama", style="chat", description="Custom desc")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["baseline"] == "llama"
        assert record["style"] == "chat"
        assert record["description"] == "Custom desc"


class TestDefaultsMerging:
    """Verify defaults.json values are always present."""

    def test_defaults_always_present(self) -> None:
        """Verify every defaults.json key appears in the output record."""
        row = _make_row()
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        defaults = _get_defaults()
        for key in defaults:
            assert key in record, f"Defaults key '{key}' missing from record"

    def test_version_from_defaults(self) -> None:
        """Verify version defaults to '1' from defaults.json."""
        row = _make_row()
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["version"] == "1"

    def test_nsfw_from_defaults(self) -> None:
        """Verify nsfw defaults to False from defaults.json."""
        row = _make_row()
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["nsfw"] is False

    def test_style_defaults_to_generalist(self) -> None:
        """Verify empty style defaults to 'generalist' from defaults.json."""
        row = _make_row(style="")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["style"] == "generalist"


class TestInstructFormat:
    """Verify instruct_format is preserved through the conversion."""

    def test_instruct_format_present_in_output(self) -> None:
        """Verify non-empty instruct_format appears in all 3 entries."""
        row = _make_row(instruct_format="alpaca")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=True)

        for key, record in result.items():
            assert "instruct_format" in record, f"instruct_format missing from {key}"
            assert record["instruct_format"] == "alpaca"

    def test_empty_instruct_format_stripped(self) -> None:
        """Verify empty instruct_format is removed by the empty-value filter."""
        row = _make_row(instruct_format="")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "instruct_format" not in record


class TestBackendPrefixes:
    """Verify backend prefix generation matches convert.py."""

    def test_three_entries_per_model(self) -> None:
        """Verify base, aphrodite/, and koboldcpp/ entries are generated."""
        row = _make_row(name="ReadyArt/Broken-Tutu-24B")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=True)

        assert "ReadyArt/Broken-Tutu-24B" in result
        assert "aphrodite/ReadyArt/Broken-Tutu-24B" in result
        assert "koboldcpp/Broken-Tutu-24B" in result
        assert len(result) == 3

    def test_one_entry_without_prefixes(self) -> None:
        """Verify only the base entry is generated without prefixes."""
        row = _make_row(name="ReadyArt/Broken-Tutu-24B")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)

        assert "ReadyArt/Broken-Tutu-24B" in result
        assert len(result) == 1

    def test_name_field_matches_key(self) -> None:
        """Verify each entry's name field matches its dict key."""
        row = _make_row(name="Org/Model-7B")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=True)

        assert result["Org/Model-7B"]["name"] == "Org/Model-7B"
        assert result["aphrodite/Org/Model-7B"]["name"] == "aphrodite/Org/Model-7B"
        assert result["koboldcpp/Model-7B"]["name"] == "koboldcpp/Model-7B"

    def test_model_name_consistent_across_entries(self) -> None:
        """Verify model_name is the same across all 3 entries."""
        row = _make_row(name="Org/Model-7B")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=True)

        for record in result.values():
            assert record["model_name"] == "Model-7B"


class TestEmptyValueFiltering:
    """Verify empty values are stripped matching convert.py's filter."""

    def test_empty_settings_not_in_output(self) -> None:
        """Verify None settings are excluded from the output."""
        row = _make_row(settings=None)
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "settings" not in record

    def test_empty_url_not_in_output(self) -> None:
        """Verify empty string url is excluded from the output."""
        row = _make_row(url="")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "url" not in record

    def test_nonempty_settings_preserved(self) -> None:
        """Verify non-empty settings dict is preserved."""
        row = _make_row(settings={"temperature": 0.7})
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["settings"] == {"temperature": 0.7}


class TestTagGeneration:
    """Verify tags include style + size bucket, matching convert.py."""

    def test_size_tag_added(self) -> None:
        """Verify parameter-based size tag (e.g. '7B') is added."""
        row = _make_row(parameters_bn=7.0)
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "7B" in record["tags"]

    def test_style_tag_added(self) -> None:
        """Verify style value is added as a tag."""
        row = _make_row(style="chat")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "chat" in record["tags"]

    def test_existing_tags_preserved(self) -> None:
        """Verify pre-existing CSV tags are preserved in the output."""
        row = _make_row(tags=["roleplay", "story"])
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert "roleplay" in record["tags"]
        assert "story" in record["tags"]

    def test_tags_sorted(self) -> None:
        """Verify tags list is sorted alphabetically."""
        row = _make_row(tags=["zebra", "alpha"], style="chat")
        result = csv_rows_to_legacy_dict([row], with_backend_prefixes=False)
        record = result["TestOrg/TestModel-7B"]

        assert record["tags"] == sorted(record["tags"])


class TestCrossValidation:
    """Cross-validate against an independent convert.py reference implementation."""

    def test_matches_convert_py_reference(self, tmp_path: Path) -> None:
        """Compare shared function output against independent convert.py transliteration."""
        csv_path = tmp_path / "models.csv"
        fieldnames = [
            "name",
            "parameters_bn",
            "display_name",
            "url",
            "baseline",
            "description",
            "style",
            "tags",
            "instruct_format",
            "settings",
        ]
        raw_rows = [
            {
                "name": "Org/Model-7B",
                "parameters_bn": "7",
                "display_name": "",
                "url": "https://example.com",
                "baseline": "llama",
                "description": "A test model",
                "style": "chat",
                "tags": "roleplay,story",
                "instruct_format": "alpaca",
                "settings": '{"temperature": 0.7}',
            },
            {
                "name": "Another/Small-Model",
                "parameters_bn": "0.56",
                "display_name": "Custom Display",
                "url": "",
                "baseline": "",
                "description": "",
                "style": "",
                "tags": "",
                "instruct_format": "",
                "settings": "",
            },
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)

        # Reference: independent convert.py transliteration
        defaults = dict(_get_defaults())
        expected = _convert_py_reference(raw_rows, defaults)

        # Under test: our shared function via parse + convert
        parsed_rows, issues = parse_legacy_text_csv_file(csv_path)
        assert not issues
        actual = csv_rows_to_legacy_dict(parsed_rows, with_backend_prefixes=True)

        assert set(actual.keys()) == set(expected.keys()), (
            f"Key mismatch.\nExtra: {set(actual.keys()) - set(expected.keys())}\n"
            f"Missing: {set(expected.keys()) - set(actual.keys())}"
        )

        for key in expected:
            assert actual[key] == expected[key], (
                f"Record mismatch for '{key}':\n"
                f"Expected: {json.dumps(expected[key], indent=2)}\n"
                f"Actual:   {json.dumps(actual[key], indent=2)}"
            )

    def test_matches_upstream_db_json(self) -> None:
        """Cross-validate against actual upstream repo files if available."""
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

        expected = response_db.json()
        upstream_csv = response_csv.text

        parsed_rows, issues = parse_legacy_text_csv(io.StringIO(upstream_csv))
        assert not issues, f"Parse issues: {issues}"
        actual = csv_rows_to_legacy_dict(parsed_rows, with_backend_prefixes=True)

        assert set(actual.keys()) == set(expected.keys()), f"Key count: actual={len(actual)}, expected={len(expected)}"

        mismatches: list[str] = []
        for key in expected:
            if actual[key] != expected[key]:
                mismatches.append(key)

        assert not mismatches, (
            f"{len(mismatches)} records differ. First 5: {mismatches[:5]}\n"
            f"Example diff for '{mismatches[0]}':\n"
            f"Expected: {json.dumps(expected[mismatches[0]], indent=2)}\n"
            f"Actual:   {json.dumps(actual[mismatches[0]], indent=2)}"
        )


class TestWriteLegacyTextCsvRoundtrip:
    """Verify CSV write→read roundtrip preserves data."""

    def test_roundtrip_preserves_rows(self, tmp_path: Path) -> None:
        """Write rows to CSV, read them back, verify identical legacy dict output."""
        rows = [
            _make_row(
                name="Org/Model-7B",
                parameters_bn=7.0,
                style="chat",
                tags=["roleplay", "story"],
                instruct_format="alpaca",
                settings={"temperature": 0.7},
                url="https://example.com",
                baseline="llama",
                description="A test model",
                display_name="Custom Display",
            ),
            _make_row(
                name="Another/Small-Model",
                parameters_bn=0.56,
                display_name="",
            ),
        ]

        csv_path = tmp_path / "models.csv"
        write_legacy_text_csv(rows, csv_path)

        parsed_rows, issues = parse_legacy_text_csv_file(csv_path)
        assert not issues, f"Parse issues: {issues}"

        original_dict = csv_rows_to_legacy_dict(rows, with_backend_prefixes=True)
        roundtrip_dict = csv_rows_to_legacy_dict(parsed_rows, with_backend_prefixes=True)

        assert set(original_dict.keys()) == set(roundtrip_dict.keys())
        for key in original_dict:
            assert original_dict[key] == roundtrip_dict[key], (
                f"Roundtrip mismatch for '{key}':\n"
                f"Original:  {json.dumps(original_dict[key], indent=2)}\n"
                f"Roundtrip: {json.dumps(roundtrip_dict[key], indent=2)}"
            )

    def test_written_csv_has_correct_header(self, tmp_path: Path) -> None:
        """CSV file starts with the canonical header line."""
        rows = [_make_row()]
        csv_path = tmp_path / "models.csv"
        write_legacy_text_csv(rows, csv_path)

        first_line = csv_path.read_text(encoding="utf-8").split("\n")[0].strip()
        expected = "name,parameters_bn,display_name,url,baseline,description,style,tags,instruct_format,settings"
        assert first_line == expected


class TestLegacyRecordToCsvRow:
    """Verify reverse conversion from db.json record to TextCSVRow."""

    def test_strips_auto_generated_tags(self) -> None:
        """Size tag and style tag are stripped since convert.py adds them."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "style": "roleplay",
            "tags": ["7B", "roleplay", "story", "popular"],
        }
        row = legacy_record_to_csv_row("org/Model-7B", record)
        assert "7B" not in row.tags
        assert "roleplay" not in row.tags
        assert set(row.tags) == {"story", "popular"}

    def test_strips_auto_generated_display_name(self) -> None:
        """display_name matching the auto-generated value is cleared."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "display_name": "Model 7B",
        }
        row = legacy_record_to_csv_row("org/Model-7B", record)
        assert row.display_name == ""

    def test_preserves_custom_display_name(self) -> None:
        """display_name differing from auto-generated is kept."""
        record: dict[str, Any] = {
            "parameters": 12_000_000_000,
            "display_name": "Magnum 12B V2",
        }
        row = legacy_record_to_csv_row("anthracite-org/magnum-12b-v2", record)
        assert row.display_name == "Magnum 12B V2"

    def test_reverses_parameters(self) -> None:
        """Integer parameters are converted back to float billions."""
        record: dict[str, Any] = {"parameters": 7_000_000_000}
        row = legacy_record_to_csv_row("org/Model-7B", record)
        assert row.parameters_bn == 7.0
        assert row.parameters == 7_000_000_000

    def test_preserves_instruct_format(self) -> None:
        """instruct_format is preserved through reverse conversion."""
        record: dict[str, Any] = {
            "parameters": 7_000_000_000,
            "instruct_format": "ChatML",
        }
        row = legacy_record_to_csv_row("org/Model-7B", record)
        assert row.instruct_format == "ChatML"

    def test_roundtrip_through_legacy_dict(self) -> None:
        """TextCSVRow → legacy dict → legacy_record_to_csv_row → legacy dict is identical."""
        original_row = _make_row(
            name="Org/Model-7B",
            parameters_bn=7.0,
            style="chat",
            tags=["roleplay", "story"],
            instruct_format="alpaca",
            settings={"temperature": 0.7},
            url="https://example.com",
            baseline="llama",
            description="A test model",
        )

        legacy_dict = csv_rows_to_legacy_dict([original_row], with_backend_prefixes=False)
        record = legacy_dict["Org/Model-7B"]

        roundtrip_row = legacy_record_to_csv_row("Org/Model-7B", record)
        roundtrip_dict = csv_rows_to_legacy_dict([roundtrip_row], with_backend_prefixes=False)

        assert legacy_dict == roundtrip_dict

    def test_strips_default_style_when_not_in_tags(self) -> None:
        """A style that matches defaults.json but is absent from tags was only injected by defaults."""
        defaults = _get_defaults()
        default_style = defaults.get("style", "generalist")

        record: dict[str, Any] = {
            "parameters": 3_000_000_000,
            "style": default_style,
            "tags": ["3B"],
        }
        row = legacy_record_to_csv_row("acrastt/Marx-3B-V3", record)
        assert row.style == "", f"Default-only style '{default_style}' should be stripped when absent from tags"

    def test_preserves_explicit_generalist_style(self) -> None:
        """An explicit generalist style (present in tags) must be preserved."""
        defaults = _get_defaults()
        default_style = defaults.get("style", "generalist")

        record: dict[str, Any] = {
            "parameters": 14_000_000_000,
            "style": default_style,
            "tags": ["14B", default_style, "agentic"],
        }
        row = legacy_record_to_csv_row("mistralai/Ministral-3-14B-Instruct-2512", record)
        assert row.style == default_style, "Explicit generalist style (present in tags) should be preserved"

    def test_no_style_model_does_not_acquire_generalist_tag_on_roundtrip(self) -> None:
        """Multi-pass stability: models without style must not accumulate a generalist tag.

        This is the core regression test for the bug where:
        1. CSV has no style → defaults inject style="generalist" into db.json
        2. Reverse conversion writes "generalist" to CSV style column
        3. Next forward conversion adds "generalist" to tags
        """
        original_row = _make_row(name="acrastt/Marx-3B-V3", parameters_bn=3.0, style="", tags=[])

        # Pass 1: CSV → legacy dict (simulates convert.py)
        dict_pass1 = csv_rows_to_legacy_dict([original_row], with_backend_prefixes=False)
        record1 = dict_pass1["acrastt/Marx-3B-V3"]
        tags_pass1 = set(record1["tags"])

        # Reverse: legacy dict → CSV row
        csv_row_pass1 = legacy_record_to_csv_row("acrastt/Marx-3B-V3", record1)

        # Pass 2: CSV → legacy dict again
        dict_pass2 = csv_rows_to_legacy_dict([csv_row_pass1], with_backend_prefixes=False)
        record2 = dict_pass2["acrastt/Marx-3B-V3"]
        tags_pass2 = set(record2["tags"])

        assert tags_pass1 == tags_pass2, (
            f"Tags changed after roundtrip: {tags_pass1} → {tags_pass2}. "
            "A defaulted style is leaking into the CSV and then into tags."
        )

    def test_explicit_style_stable_across_roundtrips(self) -> None:
        """Models with an explicit style remain stable across multiple roundtrips."""
        original_row = _make_row(
            name="Org/Chat-Model-8B",
            parameters_bn=8.0,
            style="chat",
            tags=["roleplay"],
        )

        dict_pass1 = csv_rows_to_legacy_dict([original_row], with_backend_prefixes=False)
        record1 = dict_pass1["Org/Chat-Model-8B"]

        csv_row = legacy_record_to_csv_row("Org/Chat-Model-8B", record1)
        dict_pass2 = csv_rows_to_legacy_dict([csv_row], with_backend_prefixes=False)
        record2 = dict_pass2["Org/Chat-Model-8B"]

        assert record1["tags"] == record2["tags"]
        assert record1["style"] == record2["style"]

    def test_multiple_no_style_models_stable(self) -> None:
        """Batch of no-style models all remain stable through roundtrip."""
        names_and_params = [
            ("acrastt/Marx-3B-V3", 3.0),
            ("Aeala/Enterredaas-33b", 33.0),
            ("aetherwiing/MN-12B-Starcannon-v3", 12.0),
            ("ai21labs/AI21-Jamba-1.5-Mini", 26.0),
        ]
        rows = [_make_row(name=n, parameters_bn=p, style="", tags=[]) for n, p in names_and_params]

        dict_pass1 = csv_rows_to_legacy_dict(rows, with_backend_prefixes=False)

        csv_rows_pass1 = [legacy_record_to_csv_row(n, dict_pass1[n]) for n, _ in names_and_params]
        dict_pass2 = csv_rows_to_legacy_dict(csv_rows_pass1, with_backend_prefixes=False)

        for name, _ in names_and_params:
            assert dict_pass1[name]["tags"] == dict_pass2[name]["tags"], (
                f"Tags changed for {name}: {dict_pass1[name]['tags']} → {dict_pass2[name]['tags']}"
            )


class TestURLNameRejection:
    """Guard against URL-shaped model names producing broken entries.

    The upstream convert.py uses ``name.split("/")[1]`` to extract model_name.
    For a normal name like ``"Org/Model-7B"``, this yields ``"Model-7B"``.
    For a URL like ``"https://huggingface.co/Org/Model"``, it yields ``""``
    (the empty segment between ``https:`` and ``huggingface.co``), which
    produces broken dict keys: the full URL as a base key, and ``"koboldcpp/"``
    with no model identifier.

    These tests verify the guard rejects such names rather than silently
    producing corrupt output.
    """

    def test_url_name_skipped_in_csv_rows_to_legacy_dict(self) -> None:
        """A CSV row whose name is a URL must be silently skipped.

        If allowed through, ``split("/")[1]`` produces ``""`` and the
        koboldcpp key becomes ``"koboldcpp/"`` — a collision hazard that
        overwrites unrelated models and breaks API lookups.
        """
        url_row = _make_row(
            name="https://huggingface.co/Org/Model-7B",
            url="https://huggingface.co/Org/Model-7B",
        )
        normal_row = _make_row(name="Org/Model-7B")

        result = csv_rows_to_legacy_dict([url_row, normal_row], with_backend_prefixes=True)

        # The URL-named model must not appear
        assert "https://huggingface.co/Org/Model-7B" not in result
        assert "koboldcpp/" not in result

        # The normal model must still be present
        assert "Org/Model-7B" in result
        assert "koboldcpp/Model-7B" in result

    def test_url_name_skipped_without_backend_prefixes(self) -> None:
        """URL guard also applies when backend prefixes are disabled."""
        url_row = _make_row(name="https://example.com/org/model")
        result = csv_rows_to_legacy_dict([url_row], with_backend_prefixes=False)
        assert len(result) == 0

    def test_extract_model_name_rejects_url(self) -> None:
        """TextModelWriteProcessor.extract_model_name raises on URL input.

        This is the underlying utility; callers that use it for display_name
        generation or model_name extraction need a clear signal rather than
        a silently empty string.
        """
        with pytest.raises(ValueError, match="URL-shaped key"):
            TextModelWriteProcessor.extract_model_name("https://huggingface.co/Org/Model")

    def test_extract_model_name_accepts_normal_names(self) -> None:
        """Sanity: normal org/model names still work after the guard."""
        assert TextModelWriteProcessor.extract_model_name("Org/Model-7B") == "Model-7B"
        assert TextModelWriteProcessor.extract_model_name("Model-7B") == "Model-7B"


class TestCSVLineEndings:
    r"""Verify CSV output uses LF line endings, not CRLF.

    Python's ``csv.DictWriter`` always emits ``\r\n`` as the record
    terminator (per RFC 4180). The upstream GitHub repositories store CSV
    files with LF endings (git-normalized). If we write CRLF, every line
    in the file shows as changed in the diff (the ``^M`` artefact),
    obscuring real changes.

    These tests verify the fix: both write functions must strip ``\r``
    before the content reaches disk or the caller.
    """

    def test_write_legacy_text_csv_uses_lf(self, tmp_path: Path) -> None:
        r"""The on-disk CSV must contain only ``\n``, never ``\r\n``.

        Checked at the byte level to rule out any OS-level newline
        translation that string-level checks might miss.
        """
        rows = [_make_row(name="Org/Model-7B")]
        csv_path = tmp_path / "models.csv"
        write_legacy_text_csv(rows, csv_path)

        raw_bytes = csv_path.read_bytes()
        assert b"\r\n" not in raw_bytes, "CSV file must use LF, not CRLF"
        assert b"\n" in raw_bytes, "CSV file must contain at least one newline"

    def test_written_csv_round_trips_through_parse(self, tmp_path: Path) -> None:
        r"""A CSV written with LF endings must parse back identically.

        Guards against the possibility that the ``\r`` stripping corrupts
        quoted fields or embedded commas.
        """
        rows = [
            _make_row(
                name="Org/Model-7B",
                tags=["roleplay", "story"],
                settings={"temperature": 0.7},
                description="A test, with commas",
            ),
        ]
        csv_path = tmp_path / "models.csv"
        write_legacy_text_csv(rows, csv_path)

        parsed, issues = parse_legacy_text_csv_file(csv_path)
        assert not issues
        assert len(parsed) == 1
        assert parsed[0].name == "Org/Model-7B"
        assert set(parsed[0].tags) == {"roleplay", "story"}
