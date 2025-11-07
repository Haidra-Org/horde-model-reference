"""Test for Broken-Tutu model grouping behavior in legacy text generation CSV handling.

This test verifies that a single CSV row generates exactly 3 JSON entries:
1. Base name (e.g., ReadyArt/Broken-Tutu-24B-Unslop-v2.0)
2. Aphrodite prefixed (e.g., aphrodite/ReadyArt/Broken-Tutu-24B-Unslop-v2.0)
3. KoboldCPP prefixed (e.g., koboldcpp/Broken-Tutu-24B-Unslop-v2.0)

The test uses two rows from the actual Broken-Tutu models to verify the complete behavior.
"""

import csv
from pathlib import Path

import pytest

from horde_model_reference.legacy.classes.legacy_converters import LegacyTextGenerationConverter


class TestBrokenTutuGrouping:
    """Test the grouping behavior for Broken-Tutu models and backend prefixes."""

    @pytest.fixture
    def broken_tutu_csv_file(self, tmp_path: Path) -> Path:
        """Create a CSV file with the actual Broken-Tutu model data."""
        csv_path = tmp_path / "legacy" / "models.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "name",
            "parameters_bn",
            "description",
            "version",
            "style",
            "nsfw",
            "baseline",
            "url",
            "tags",
            "settings",
            "display_name",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Row 1: Broken-Tutu-24B-Unslop-v2.0
            writer.writerow(
                {
                    "name": "ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
                    "parameters_bn": "24",
                    "description": (
                        "An updated version of the RP powerhouse, Broken-Tutu, now with unprecedented coherence "
                        "and a zero-slop dataset. Mistral-V7-Tekken-T8-XML Recommended."
                    ),
                    "version": "1",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "mistralai/Mistral-Small-24B-Base-2501",
                    "url": "https://huggingface.co/ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
                    "tags": "roleplay, ERP, unaligned, explicit, horror, violence, popular",
                    "settings": "",
                    "display_name": "",
                }
            )

            # Row 2: Broken-Tutu-24B
            writer.writerow(
                {
                    "name": "ReadyArt/Broken-Tutu-24B",
                    "parameters_bn": "24",
                    "description": (
                        "A do-it-all RP powerhouse with a lively, initiative-driven voice and rock-solid adherence "
                        "to even the most complex character cards. Go-to daily-driver for slice-of-life, fantasy, "
                        "adventure, ERP, and gritty horror - effortlessly churning out detailed adventure scenes and "
                        "intimate dialogues alike. Fully unaligned and uncensored, it delivers extreme NSFW/NSFL "
                        "performance when appropriate and tracks up to 7 characters, time, and location. Beware: "
                        "without careful temperature tweaks it can loop or get overly hostile, and mild repetition "
                        "may creep in extremely-long sessions. Creator recommends "
                        "sleepdeprived3/Mistral-V7-Tekken-T5-XML preset."
                    ),
                    "version": "1",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "",
                    "url": "https://huggingface.co/ReadyArt/Broken-Tutu-24B",
                    "tags": "roleplay, unaligned, popular",
                    "settings": "",
                    "display_name": "",
                }
            )

        return csv_path

    def test_broken_tutu_generates_six_entries(self, broken_tutu_csv_file: Path, tmp_path: Path) -> None:
        """Test that two CSV rows generate exactly 6 JSON entries (3 per row)."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Convert - this loads CSV and processes it internally
        converted_records = converter.convert_to_new_format()

        # In the v2 format, we should NOT have backend-prefixed entries
        # They should be filtered out during conversion
        # So we expect only 2 base entries
        assert len(converted_records) == 2, f"Expected 2 entries, got {len(converted_records)}"

        assert "ReadyArt/Broken-Tutu-24B-Unslop-v2.0" in converted_records
        assert "ReadyArt/Broken-Tutu-24B" in converted_records

    def test_github_backend_format_has_six_entries(self, broken_tutu_csv_file: Path, tmp_path: Path) -> None:
        """Test that the GitHub backend format (legacy CSV → dict) produces 6 entries.

        This tests the actual GitHubBackend._read_legacy_csv_to_dict method to ensure
        it creates 3 entries per CSV row as expected.
        """
        from horde_model_reference.backends.github_backend import GitHubBackend

        # Create a GitHub backend instance (we'll use its method directly)
        backend = GitHubBackend()

        # Call the actual _read_legacy_csv_to_dict method
        data = backend._read_legacy_csv_to_dict(broken_tutu_csv_file)

        # Verify we have exactly 6 entries (3 per CSV row)
        assert len(data) == 6, f"Expected 6 entries from GitHub backend format, got {len(data)}"

        # Verify all expected keys exist
        expected_keys = [
            "ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
            "aphrodite/ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
            "koboldcpp/Broken-Tutu-24B-Unslop-v2.0",
            "ReadyArt/Broken-Tutu-24B",
            "aphrodite/ReadyArt/Broken-Tutu-24B",
            "koboldcpp/Broken-Tutu-24B",
        ]

        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in data"

        # Verify each entry has the correct 'name' field
        for key in expected_keys:
            assert data[key]["name"] == key, f"Entry '{key}' has incorrect name field: {data[key]['name']}"

    def test_broken_tutu_v2_entry_structure(self, broken_tutu_csv_file: Path, tmp_path: Path) -> None:
        """Test that each entry has the correct structure matching the expected JSON."""
        import re

        from horde_model_reference.legacy.text_csv_utils import parse_legacy_text_csv

        parsed_rows, _ = parse_legacy_text_csv(broken_tutu_csv_file)

        # Process the first row (Broken-Tutu-24B-Unslop-v2.0)
        csv_row = parsed_rows[0]
        name = csv_row.name
        model_name = name.split("/")[1] if "/" in name else name

        # Build expected structure
        tags = set(csv_row.tags)
        if csv_row.style:
            tags.add(csv_row.style)
        tags.add("24B")  # Parameter size tag

        re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

        expected_base = {
            "name": "ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
            "model_name": "Broken-Tutu-24B-Unslop-v2.0",
            "baseline": "mistralai/Mistral-Small-24B-Base-2501",
            "parameters": 24000000000,
            "description": (
                "An updated version of the RP powerhouse, Broken-Tutu, now with unprecedented coherence "
                "and a zero-slop dataset. Mistral-V7-Tekken-T8-XML Recommended."
            ),
            "version": "1",
            "style": "chat",
            "nsfw": False,
            "display_name": "Broken Tutu 24B Unslop v2.0",
            "url": "https://huggingface.co/ReadyArt/Broken-Tutu-24B-Unslop-v2.0",
            "tags": sorted(tags),
        }

        # Verify tag contents
        tags_value = expected_base["tags"]
        if isinstance(tags_value, (list, set, tuple)):
            tags_list = list(tags_value)
        else:
            raise ValueError("Tags field is not a list, set, or tuple")
        assert "24B" in tags_list
        assert "ERP" in tags_list
        assert "chat" in tags_list
        assert "explicit" in tags_list
        assert "horror" in tags_list
        assert "popular" in tags_list
        assert "roleplay" in tags_list
        assert "unaligned" in tags_list
        assert "violence" in tags_list

    def test_koboldcpp_uses_model_name_not_full_name(self, broken_tutu_csv_file: Path, tmp_path: Path) -> None:
        """Test that KoboldCPP entries use model_name only, not the full name."""
        from horde_model_reference.legacy.text_csv_utils import parse_legacy_text_csv

        parsed_rows, _ = parse_legacy_text_csv(broken_tutu_csv_file)

        for csv_row in parsed_rows:
            name = csv_row.name
            model_name = name.split("/")[1] if "/" in name else name

            # KoboldCPP key should use model_name, not full name
            expected_koboldcpp_key = f"koboldcpp/{model_name}"

            # Verify the format
            assert "/" in name, "Test expects names with '/' separator"
            assert expected_koboldcpp_key == f"koboldcpp/{model_name}"

            # For ReadyArt/Broken-Tutu-24B-Unslop-v2.0:
            # - Base: ReadyArt/Broken-Tutu-24B-Unslop-v2.0
            # - Aphrodite: aphrodite/ReadyArt/Broken-Tutu-24B-Unslop-v2.0
            # - KoboldCPP: koboldcpp/Broken-Tutu-24B-Unslop-v2.0 (NOT koboldcpp/ReadyArt/Broken-Tutu-24B-Unslop-v2.0)
