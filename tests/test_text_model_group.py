"""Test text_model_group field in TextGenerationModelRecord."""

from pathlib import Path
from typing import cast

from horde_model_reference.analytics.text_model_parser import group_text_models_by_base
from horde_model_reference.legacy.classes.legacy_converters import LegacyTextGenerationConverter
from horde_model_reference.model_reference_records import TextGenerationModelRecord


class TestTextModelGroup:
    """Test text_model_group field in converted records."""

    def test_text_model_group_field_exists(self) -> None:
        """Test that TextGenerationModelRecord has text_model_group field."""
        record = TextGenerationModelRecord(
            name="test-model",
            parameters=7000000000,
        )
        assert hasattr(record, "text_model_group")
        assert record.text_model_group is None  # Default value

    def test_text_model_group_can_be_set(self) -> None:
        """Test that text_model_group can be set."""
        record = TextGenerationModelRecord(
            name="Llama-3-8B-Instruct",
            parameters=8000000000,
            text_model_group="Llama-3",
        )
        assert record.text_model_group == "Llama-3"

    def test_converter_populates_text_model_group(self, tmp_path: Path) -> None:
        """Test that converter populates text_model_group field."""
        import csv

        # Create the proper directory structure
        legacy_path = tmp_path / "legacy"
        legacy_path.mkdir(parents=True, exist_ok=True)
        csv_path = legacy_path / "models.csv"

        # Create a CSV file with test data using proper CSV writer
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
                    "name": "Llama-3-8B-Instruct",
                    "parameters_bn": "8.0",
                    "description": "Test model",
                    "version": "1.0",
                    "style": "",
                    "nsfw": "false",
                    "baseline": "llama",
                    "url": "",
                    "tags": "test-tag",
                    "settings": "",
                    "display_name": "",
                }
            )
            writer.writerow(
                {
                    "name": "Llama-3-8B-Instruct-Q4",
                    "parameters_bn": "8.0",
                    "description": "Test quantized model",
                    "version": "1.0",
                    "style": "",
                    "nsfw": "false",
                    "baseline": "llama",
                    "url": "",
                    "tags": "test-tag",
                    "settings": "",
                    "display_name": "",
                }
            )
            writer.writerow(
                {
                    "name": "Mistral-7B-v0.1",
                    "parameters_bn": "7.0",
                    "description": "Different model",
                    "version": "1.0",
                    "style": "",
                    "nsfw": "false",
                    "baseline": "mistral",
                    "url": "",
                    "tags": "test-tag",
                    "settings": "",
                    "display_name": "",
                }
            )

        # Create converter and convert
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=tmp_path,
            target_file_folder=tmp_path / "output",
        )
        converted_records = converter.convert_to_new_format()

        # Verify text_model_group was populated
        assert "Llama-3-8B-Instruct" in converted_records
        assert "Llama-3-8B-Instruct-Q4" in converted_records
        assert "Mistral-7B-v0.1" in converted_records

        llama_record = cast(TextGenerationModelRecord, converted_records["Llama-3-8B-Instruct"])
        llama_quant_record = cast(TextGenerationModelRecord, converted_records["Llama-3-8B-Instruct-Q4"])
        mistral_record = cast(TextGenerationModelRecord, converted_records["Mistral-7B-v0.1"])

        # Verify they are TextGenerationModelRecords
        assert isinstance(llama_record, TextGenerationModelRecord)
        assert isinstance(llama_quant_record, TextGenerationModelRecord)
        assert isinstance(mistral_record, TextGenerationModelRecord)

        # Both Llama variants should have the same group
        assert llama_record.text_model_group == "Llama-3"
        assert llama_quant_record.text_model_group == "Llama-3"

        # Mistral should have its own group
        assert mistral_record.text_model_group == "Mistral-v0.1"

    def test_grouping_logic_matches_parser(self) -> None:
        """Test that the grouping logic matches the parser function."""
        model_names = [
            "Llama-3-8B-Instruct",
            "Llama-3-8B-Instruct-Q4",
            "Llama-3-70B-Instruct",
            "Mistral-7B-v0.1",
        ]

        grouped = group_text_models_by_base(model_names)

        # Verify grouping structure
        assert "Llama-3" in grouped
        assert "Mistral-v0.1" in grouped

        assert grouped["Llama-3"].base_name == "Llama-3"
        assert len(grouped["Llama-3"].variants) == 3
        assert set(grouped["Llama-3"].variants) == {
            "Llama-3-8B-Instruct",
            "Llama-3-8B-Instruct-Q4",
            "Llama-3-70B-Instruct",
        }

        assert grouped["Mistral-v0.1"].base_name == "Mistral-v0.1"
        assert len(grouped["Mistral-v0.1"].variants) == 1
