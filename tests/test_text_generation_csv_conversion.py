"""Tests for text_generation CSV to JSON conversion.

This test module verifies that:
1. Legacy CSV files are correctly read and parsed
2. CSV data is converted to v2 JSON format
3. Converter handles edge cases (empty files, malformed data, etc.)
4. Round-trip conversion preserves data integrity

IMPORTANT NOTES:
- text_generation is the ONLY category that uses CSV format for legacy files
- Legacy path: {base}/legacy/models.csv (CSV format)
- V2 path: {base}/text_generation.json (JSON format)
- CSV columns: name, parameters_bn, description, version, style, nsfw, baseline,
  url, tags (comma-separated), settings (JSON string), display_name
- Parameters conversion: parameters_bn (billions) * 1,000,000,000 = parameters (integer)
- Settings field constraint: Only flat dicts allowed, no nested dicts. Type is:
  dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None
- Output is ALWAYS JSON, never CSV (critical regression test included)
"""

import csv
import json
from pathlib import Path

import pytest

from horde_model_reference import horde_model_reference_paths
from horde_model_reference.legacy.classes.legacy_converters import LegacyTextGenerationConverter
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import TextGenerationModelRecord


class TestTextGenerationCSVConversion:
    """Test CSV to JSON conversion for text_generation category."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path: Path) -> Path:
        """Create a sample CSV file with test data."""
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
            writer.writerow(
                {
                    "name": "test-model-1",
                    "parameters_bn": "7.0",
                    "description": "A test model",
                    "version": "1.0",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "test-baseline",
                    "url": "https://example.com/model1",
                    "tags": "tag1,tag2,tag3",
                    "settings": '{"temperature": 0.7}',
                    "display_name": "Test Model 1",
                }
            )
            writer.writerow(
                {
                    "name": "test-model-2",
                    "parameters_bn": "13.0",
                    "description": "Another test model",
                    "version": "2.0",
                    "style": "instruct",
                    "nsfw": "true",
                    "baseline": "test-baseline-2",
                    "url": "https://example.com/model2",
                    "tags": "tag4,tag5",
                    "settings": '{"top_p": 0.9}',
                    "display_name": "Test Model 2",
                }
            )

        return csv_path

    @pytest.fixture
    def empty_csv_file(self, tmp_path: Path) -> Path:
        """Create an empty CSV file."""
        csv_path = tmp_path / "legacy" / "models.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.touch()
        return csv_path

    def test_converter_reads_csv_file(self, sample_csv_file: Path, tmp_path: Path) -> None:
        """Test that the converter can read CSV files."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Trigger conversion
        converted_records = converter.convert_to_new_format()

        # Verify records were loaded
        assert len(converted_records) >= 2
        assert "test-model-1" in converted_records or "Test Model 1" in converted_records
        assert "test-model-2" in converted_records or "Test Model 2" in converted_records

    def test_converter_writes_json_file(self, sample_csv_file: Path, tmp_path: Path) -> None:
        """Test that the converter writes JSON format, not CSV."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Convert
        converter.convert_to_new_format()

        # Check output file
        output_file = horde_model_reference_paths.get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )

        assert output_file is not None
        assert output_file.exists()
        assert output_file.suffix == ".json", f"Output file should be JSON, got {output_file.suffix}"

        # Verify it's valid JSON
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert len(data) >= 2

    def test_csv_to_json_data_integrity(self, sample_csv_file: Path, tmp_path: Path) -> None:
        """Test that CSV data is correctly converted to JSON."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Convert
        converted_records = converter.convert_to_new_format()

        # Verify data integrity - find the first model
        model_record = None
        for record in converted_records.values():
            if "test-model-1" in record.name.lower() or record.name == "test-model-1":
                model_record = record
                break

        assert model_record is not None, "Could not find test-model-1 in converted records"
        assert isinstance(model_record, TextGenerationModelRecord)

        # Check fields
        assert model_record.description == "A test model"
        assert model_record.version == "1.0"
        assert model_record.style == "chat"
        assert model_record.nsfw is False
        assert model_record.baseline == "test-baseline"
        assert model_record.url == "https://example.com/model1"
        assert "tag1" in model_record.tags
        assert "tag2" in model_record.tags
        assert "tag3" in model_record.tags
        assert model_record.settings == {"temperature": 0.7}
        assert model_record.display_name == "Test Model 1"

        # Check parameter conversion (7.0 billion = 7,000,000,000)
        assert model_record.parameters_count == 7_000_000_000

    def test_empty_csv_file_handling(self, empty_csv_file: Path, tmp_path: Path) -> None:
        """Test that empty CSV files are handled gracefully."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # This should not raise an error
        converted_records = converter.convert_to_new_format()

        # Should return empty dict
        assert isinstance(converted_records, dict)
        assert len(converted_records) == 0

    def test_csv_with_missing_optional_fields(self, tmp_path: Path) -> None:
        """Test that CSV with missing optional fields is handled correctly."""
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
            writer.writerow(
                {
                    "name": "minimal-model",
                    "parameters_bn": "7.0",
                    "description": "",
                    "version": "",
                    "style": "",
                    "nsfw": "false",
                    "baseline": "",
                    "url": "",
                    "tags": "",
                    "settings": "",
                    "display_name": "",
                }
            )

        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Should not raise an error
        converted_records = converter.convert_to_new_format()

        assert len(converted_records) >= 1
        model = None
        for record in converted_records.values():
            if record.name == "minimal-model":
                model = record
                break

        assert model is not None
        assert model.name == "minimal-model"
        assert model.parameters_count == 7_000_000_000
        assert model.tags == [] or model.tags is None
        assert model.settings is None or model.settings == {}

    def test_csv_with_complex_settings(self, tmp_path: Path) -> None:
        """Test that complex JSON settings are preserved during conversion."""
        csv_path = tmp_path / "legacy" / "models.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        complex_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048,
            "stop_sequences": ["</s>", "[DONE]"],
            "enabled": True,
        }

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
            writer.writerow(
                {
                    "name": "complex-model",
                    "parameters_bn": "7.0",
                    "description": "Model with complex settings",
                    "version": "1.0",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "test",
                    "url": "https://example.com",
                    "tags": "",
                    "settings": json.dumps(complex_settings),
                    "display_name": "Complex Model",
                }
            )

        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        converted_records = converter.convert_to_new_format()

        model = None
        for _key, record in converted_records.items():
            if record.name == "complex-model" or "complex" in record.name.lower():
                model = record
                break

        assert model is not None, f"Could not find complex-model. Records: {list(converted_records.keys())}"
        assert model.settings == complex_settings
        assert model.settings["temperature"] == 0.7
        assert model.settings["stop_sequences"] == ["</s>", "[DONE]"]
        assert model.settings["enabled"] is True

    def test_nonexistent_csv_file(self, tmp_path: Path) -> None:
        """Test that converter handles missing CSV file gracefully."""
        # Don't create the CSV file
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        # Should not raise an error
        converted_records = converter.convert_to_new_format()

        # Should return empty dict
        assert isinstance(converted_records, dict)
        assert len(converted_records) == 0

    def test_output_file_is_json_not_csv(self, sample_csv_file: Path, tmp_path: Path) -> None:
        """Regression test: Ensure output is always JSON, never CSV."""
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        converter.convert_to_new_format()

        output_file = horde_model_reference_paths.get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=tmp_path,
        )

        # Read the file and verify it's JSON, not CSV
        with open(output_file, encoding="utf-8") as f:
            content = f.read()

        # Should be valid JSON
        data = json.loads(content)
        assert isinstance(data, dict)

        # Should NOT be CSV (would fail JSON parsing or not be a dict)
        # Also check that it doesn't have CSV headers
        assert not content.startswith("name,parameters_bn,description")

    def test_large_parameter_values(self, tmp_path: Path) -> None:
        """Test that large parameter values are correctly converted."""
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
            # Test various parameter sizes
            writer.writerow(
                {
                    "name": "large-model",
                    "parameters_bn": "70.0",  # 70 billion
                    "description": "Large model",
                    "version": "1.0",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "",
                    "url": "",
                    "tags": "",
                    "settings": "",
                    "display_name": "",
                }
            )
            writer.writerow(
                {
                    "name": "small-model",
                    "parameters_bn": "0.5",  # 500 million
                    "description": "Small model",
                    "version": "1.0",
                    "style": "chat",
                    "nsfw": "false",
                    "baseline": "",
                    "url": "",
                    "tags": "",
                    "settings": "",
                    "display_name": "",
                }
            )

        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path,
            debug_mode=False,
        )

        converted_records = converter.convert_to_new_format()

        # Find models
        large_model = None
        small_model = None
        for record in converted_records.values():
            if record.name == "large-model":
                large_model = record
            elif record.name == "small-model":
                small_model = record

        assert large_model is not None
        assert large_model.parameters_count == 70_000_000_000

        assert small_model is not None
        assert small_model.parameters_count == 500_000_000
