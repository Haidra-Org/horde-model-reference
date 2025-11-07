"""Tests to verify text_generation file paths and formats.

This test module verifies that:
1. Legacy folder contains models.csv in CSV format
2. Base folder contains text_generation.json in JSON format
3. Both files have correct structure and content
"""

import csv
import json
from pathlib import Path

from horde_model_reference import horde_model_reference_paths
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class TestTextGenerationFilePaths:
    """Test that text_generation files exist in correct locations with correct formats."""

    def test_v2_text_generation_is_json_format(self, tmp_path: Path) -> None:
        """Verify that the v2 text_generation file is JSON format, not CSV.

        The v2 format file should be:
        - Located at: {base_path}/text_generation.json
        - Format: JSON
        - Content: Dict with model records
        """
        # Get the expected v2 file path
        v2_file_path = horde_model_reference_paths.get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )

        assert v2_file_path is not None
        assert v2_file_path.name == "text_generation.json", (
            f"V2 file should be text_generation.json, not {v2_file_path.name}"
        )

        # Create a sample JSON file
        sample_data = {
            "test-model": {
                "name": "test-model",
                "parameters": 7_000_000_000,
                "description": "Test model",
            }
        }

        v2_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(v2_file_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)

        # Verify it's valid JSON
        with open(v2_file_path, encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data == sample_data
        assert "test-model" in loaded_data

    def test_legacy_text_generation_is_csv_format(self, tmp_path: Path) -> None:
        """Verify that the legacy text_generation file is CSV format.

        The legacy format file should be:
        - Located at: {base_path}/legacy/models.csv
        - Format: CSV
        - Content: CSV with proper headers and rows
        """
        # Get the expected legacy file path
        legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )

        assert legacy_file_path is not None
        assert legacy_file_path.name == "models.csv", f"Legacy file should be models.csv, not {legacy_file_path.name}"

        # Create a sample CSV file
        legacy_file_path.parent.mkdir(parents=True, exist_ok=True)

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

        with open(legacy_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "name": "test-model",
                    "parameters_bn": "7.0",
                    "description": "Test model",
                    "version": "1.0",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "test",
                    "url": "https://example.com",
                    "tags": "tag1,tag2",
                    "settings": "{}",
                    "display_name": "Test Model",
                }
            )

        # Verify it's valid CSV
        with open(legacy_file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["name"] == "test-model"
        assert rows[0]["parameters_bn"] == "7.0"

    def test_both_files_can_coexist(self, tmp_path: Path) -> None:
        """Verify that both v2 JSON and legacy CSV can exist simultaneously."""
        # Get both paths
        v2_file_path = horde_model_reference_paths.get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )
        legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )

        assert v2_file_path is not None
        assert legacy_file_path is not None

        # Verify they're different files
        assert v2_file_path != legacy_file_path
        assert v2_file_path.name == "text_generation.json"
        assert legacy_file_path.name == "models.csv"

        # Create both files
        v2_file_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_file_path.parent.mkdir(parents=True, exist_ok=True)

        # V2 JSON
        v2_data = {"model1": {"name": "model1", "parameters": 7_000_000_000}}
        with open(v2_file_path, "w", encoding="utf-8") as f:
            json.dump(v2_data, f)

        # Legacy CSV
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
        with open(legacy_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "name": "model1",
                    "parameters_bn": "7.0",
                    "description": "",
                    "version": "",
                    "style": "",
                    "nsfw": "false",
                    "baseline": "",
                    "url": "",
                    "tags": "",
                    "settings": "{}",
                    "display_name": "",
                }
            )

        # Verify both exist and are readable
        assert v2_file_path.exists()
        assert legacy_file_path.exists()

        with open(v2_file_path, encoding="utf-8") as f:
            v2_loaded = json.load(f)
        assert "model1" in v2_loaded

        with open(legacy_file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "model1"

    def test_path_consts_filenames_are_correct(self) -> None:
        """Verify that path_consts has correct filenames configured."""
        # V2 format should be text_generation.json
        v2_filename = horde_model_reference_paths.model_reference_filenames.get(
            MODEL_REFERENCE_CATEGORY.text_generation
        )
        assert v2_filename == "text_generation.json", (
            f"V2 filename should be 'text_generation.json', got '{v2_filename}'"
        )

        # Legacy format should use models.csv (this is what gets downloaded from GitHub)
        # The legacy filename is built by get_legacy_model_reference_file_path
        # which currently uses the same filename as v2, but should use models.csv for text_generation
