"""Tests for LegacyTextValidator ensuring convert.py compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from horde_model_reference.sync.legacy_text_validator import LegacyRecordDict, LegacyTextValidator


@pytest.fixture
def validator() -> LegacyTextValidator:
    """Create a LegacyTextValidator instance for testing."""
    return LegacyTextValidator()


@pytest.fixture
def minimal_text_record() -> dict[str, Any]:
    """Minimal valid text generation record."""
    return {
        "name": "test-model",
        "parameters": 7000000000,  # 7B
        "description": "Test model",
        "version": "1.0",
        "style": "generalist",
        "config": {
            "files": [],
            "download": [
                {
                    "file_name": "model.safetensors",
                    "file_url": "https://example.com/model.safetensors",
                    "file_path": "",
                }
            ],
        },
    }


@pytest.fixture
def text_record_with_settings() -> dict[str, Any]:
    """Text record with valid settings."""
    return {
        "name": "test-model-settings",
        "parameters": 13000000000,  # 13B
        "description": "Test model with settings",
        "version": "1.0",
        "style": "chat",
        "settings": {
            "temperature": 0.7,
            "max_length": 512,
            "top_p": 0.9,
        },
        "config": {
            "files": [],
            "download": [
                {
                    "file_name": "model.safetensors",
                    "file_url": "https://example.com/model.safetensors",
                    "file_path": "",
                }
            ],
        },
    }


def test_validator_loads_generation_params(validator: LegacyTextValidator) -> None:
    """Test that validator loads generation_params.json."""
    assert validator.generation_params is not None
    assert isinstance(validator.generation_params, dict)
    # Check some expected keys from generation_params.json
    assert "temperature" in validator.generation_params
    assert "max_length" in validator.generation_params
    assert "top_p" in validator.generation_params


def test_validator_loads_defaults(validator: LegacyTextValidator) -> None:
    """Test that validator loads defaults.json."""
    assert validator.defaults is not None
    assert isinstance(validator.defaults, dict)
    # Check some expected keys from defaults.json
    assert "baseline" in validator.defaults
    assert "nsfw" in validator.defaults


def test_validate_minimal_record(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test validation of minimal valid record."""
    result = validator.validate_and_transform({"test-model": minimal_text_record})

    # Should have base record + 2 backend prefix duplicates
    assert len(result) == 3
    assert "test-model" in result
    assert "aphrodite/test-model" in result
    assert "koboldcpp/test-model" in result  # model_name is same as base name

    # Check base record
    base = result["test-model"]
    assert base["name"] == "test-model"
    assert base["parameters"] == 7000000000

    # Check defaults were applied
    assert "baseline" in base
    assert "nsfw" in base

    # Check tags were auto-generated
    assert "tags" in base
    assert isinstance(base["tags"], list)
    assert "generalist" in base["tags"]  # style tag
    assert "7B" in base["tags"]  # parameter size tag

    # Check display_name was auto-generated
    assert base["display_name"] == "test model"

    # Check model_name was set
    assert base["model_name"] == "test-model"


def test_validate_record_with_settings(
    validator: LegacyTextValidator,
    text_record_with_settings: dict[str, Any],
) -> None:
    """Test validation of record with settings."""
    result = validator.validate_and_transform({"test-model-settings": text_record_with_settings})

    assert len(result) == 3
    base = result["test-model-settings"]

    # Settings should be preserved
    assert "settings" in base
    assert isinstance(base["settings"], dict)
    assert base["settings"]["temperature"] == 0.7
    assert base["settings"]["max_length"] == 512
    assert base["settings"]["top_p"] == 0.9


def test_invalid_settings_key_raises_error(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that invalid settings keys raise ValueError."""
    minimal_text_record["settings"] = {
        "temperature": 0.7,
        "invalid_key_not_in_generation_params": 123,
    }

    with pytest.raises(ValueError, match="settings contains invalid keys"):
        validator.validate_and_transform({"test-model": minimal_text_record})


def test_missing_parameters_raises_error(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that missing parameters field raises ValueError."""
    del minimal_text_record["parameters"]

    with pytest.raises(ValueError, match="'parameters' field is required"):
        validator.validate_and_transform({"test-model": minimal_text_record})


def test_backend_prefix_duplicates(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that backend prefix duplicates are created correctly."""
    result = validator.validate_and_transform({"test-model": minimal_text_record})

    # Check aphrodite prefix
    aphrodite = result["aphrodite/test-model"]
    assert aphrodite["name"] == "aphrodite/test-model"
    assert aphrodite["parameters"] == minimal_text_record["parameters"]
    assert aphrodite["tags"] == result["test-model"]["tags"]

    # Check koboldcpp prefix (uses model_name)
    koboldcpp = result["koboldcpp/test-model"]
    assert koboldcpp["name"] == "koboldcpp/test-model"
    assert koboldcpp["parameters"] == minimal_text_record["parameters"]


def test_existing_backend_prefixes_are_skipped(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that existing backend-prefixed entries in input are skipped."""
    # Add a backend-prefixed entry to input
    input_data = {
        "test-model": minimal_text_record,
        "aphrodite/test-model": minimal_text_record.copy(),
    }

    result = validator.validate_and_transform(input_data)

    # Should still only have 3 entries (base + 2 regenerated prefixes)
    assert len(result) == 3
    assert "test-model" in result
    assert "aphrodite/test-model" in result
    assert "koboldcpp/test-model" in result


def test_tags_include_style_and_size(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that tags include style and parameter size."""
    minimal_text_record["style"] = "instruction"
    minimal_text_record["parameters"] = 70000000000  # 70B

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    assert isinstance(base["tags"], list)
    assert "instruction" in base["tags"]
    assert "70B" in base["tags"]


def test_tags_preserve_existing_tags(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that existing tags are preserved."""
    minimal_text_record["tags"] = ["custom-tag", "another-tag"]

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    assert isinstance(base["tags"], list)
    assert "custom-tag" in base["tags"]
    assert "another-tag" in base["tags"]
    assert "generalist" in base["tags"]  # style
    assert "7B" in base["tags"]  # size


def test_display_name_generation(validator: LegacyTextValidator) -> None:
    """Test display name generation."""
    test_cases = [
        ("llama-2-7b-chat", "llama 2 7b chat"),
        ("gpt_2_medium", "gpt 2 medium"),
        ("mixtral-8x7b", "mixtral 8x7b"),
        ("model__with___multiple___underscores", "model with multiple underscores"),
    ]

    for model_name, expected_display in test_cases:
        assert validator._generate_display_name(model_name) == expected_display


def test_display_name_not_overwritten_if_provided(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that provided display_name is not overwritten."""
    minimal_text_record["display_name"] = "Custom Display Name"

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    assert base["display_name"] == "Custom Display Name"


def test_model_with_slash_in_name(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test handling of model name with slash (e.g., 'org/model')."""
    minimal_text_record["name"] = "huggingface/llama-2-7b"

    result = validator.validate_and_transform({"huggingface/llama-2-7b": minimal_text_record})

    base = result["huggingface/llama-2-7b"]
    assert base["model_name"] == "llama-2-7b"

    # Koboldcpp should use the extracted model_name (raw, not with spaces)
    assert "koboldcpp/llama-2-7b" in result


def test_defaults_are_applied(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that defaults.json fields are applied."""
    # Remove fields that would normally come from defaults
    minimal_text_record.pop("description", None)
    minimal_text_record.pop("version", None)

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    # Defaults should be applied
    assert "baseline" in base
    assert "nsfw" in base
    assert "style" in base
    assert "version" in base


def test_record_overrides_defaults(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that record values override defaults."""
    minimal_text_record["nsfw"] = True
    minimal_text_record["baseline"] = "custom-baseline"

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    # Record values should override defaults
    assert base["nsfw"] is True
    assert base["baseline"] == "custom-baseline"


def test_multiple_models_validation(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test validation of multiple models at once."""
    model1 = minimal_text_record.copy()
    model1["name"] = "model-1"

    model2 = minimal_text_record.copy()
    model2["name"] = "model-2"
    model2["parameters"] = 13000000000  # 13B

    input_data = {
        "model-1": model1,
        "model-2": model2,
    }

    result = validator.validate_and_transform(input_data)

    # Should have 2 base models + 4 prefix duplicates = 6 total
    assert len(result) == 6

    # Check both base models exist
    assert "model-1" in result
    assert "model-2" in result

    # Check parameter size tags are correct
    assert isinstance(result["model-1"]["tags"], list)
    assert isinstance(result["model-2"]["tags"], list)
    assert "7B" in result["model-1"]["tags"]
    assert "13B" in result["model-2"]["tags"]


def test_tags_handle_comma_separated_string(validator: LegacyTextValidator) -> None:
    """Test that tags can be provided as comma-separated string (legacy format)."""
    record: LegacyRecordDict = {
        "name": "test-model",
        "parameters": 7000000000,
        "style": "generalist",
        "tags": "tag1, tag2, tag3",
        "config": {"files": [], "download": []},
    }

    result = validator.validate_and_transform({"test-model": record})
    base = result["test-model"]

    assert isinstance(base["tags"], list)
    assert "tag1" in base["tags"]
    assert "tag2" in base["tags"]
    assert "tag3" in base["tags"]


def test_parameter_size_rounding(validator: LegacyTextValidator) -> None:
    """Test that parameter size tags are rounded correctly."""
    test_cases = [
        (6_700_000_000, "7B"),  # 6.7B rounds to 7B
        (13_200_000_000, "13B"),  # 13.2B rounds to 13B
        (70_000_000_000, "70B"),  # 70B exactly
        (175_000_000_000, "175B"),  # 175B
    ]

    for params, expected_tag in test_cases:
        record: LegacyRecordDict = {
            "name": "test-model",
            "parameters": params,
            "config": {"files": [], "download": []},
        }

        result = validator.validate_and_transform({"test-model": record})
        base = result["test-model"]

        assert isinstance(base["tags"], list)
        assert expected_tag in base["tags"], f"Expected {expected_tag} for {params} parameters"


def test_validator_with_custom_paths(tmp_path: Path) -> None:
    """Test validator with custom generation_params and defaults paths."""
    # Create custom files
    gen_params_path = tmp_path / "gen_params.json"
    gen_params_path.write_text(json.dumps({"custom_param": 123}))

    defaults_path = tmp_path / "defaults.json"
    defaults_path.write_text(json.dumps({"custom_default": "value"}))

    validator = LegacyTextValidator(
        generation_params_path=gen_params_path,
        defaults_path=defaults_path,
    )

    assert validator.generation_params == {"custom_param": 123}
    assert validator.defaults == {"custom_default": "value"}


def test_validator_raises_on_missing_files(tmp_path: Path) -> None:
    """Test that validator raises error if required files are missing."""
    with pytest.raises(FileNotFoundError, match="Required file not found"):
        LegacyTextValidator(
            generation_params_path=tmp_path / "nonexistent.json",
            defaults_path=tmp_path / "defaults.json",
        )


def test_settings_none_is_valid(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that settings can be None or empty."""
    minimal_text_record["settings"] = None

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]
    assert "settings" not in base

    # Also test with empty dict
    minimal_text_record["settings"] = {}
    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]
    assert "settings" not in base


def test_settings_json_string_parsed(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that JSON string settings are parsed like convert.py."""
    minimal_text_record["settings"] = json.dumps({"temperature": 0.42})

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    assert base["settings"] == {"temperature": 0.42}


def test_settings_invalid_json_raises_error(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that invalid JSON string settings raise ValueError."""
    minimal_text_record["settings"] = "{invalid"

    with pytest.raises(ValueError, match="settings must be valid JSON"):
        validator.validate_and_transform({"test-model": minimal_text_record})


def test_style_from_defaults_not_added_to_tags(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that default style does not create a style tag."""
    record = minimal_text_record.copy()
    record.pop("style", None)

    result = validator.validate_and_transform({"test-model": record})
    base = result["test-model"]

    assert base["style"] == validator.defaults["style"]
    assert base["tags"] == ["7B"]


def test_none_values_use_defaults(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that falsey values are replaced by defaults."""
    record = minimal_text_record.copy()
    record["description"] = None
    record["baseline"] = ""

    result = validator.validate_and_transform({"test-model": record})
    base = result["test-model"]

    assert base["description"] == validator.defaults["description"]
    assert base["baseline"] == validator.defaults["baseline"]


def test_tags_are_sorted(
    validator: LegacyTextValidator,
    minimal_text_record: dict[str, Any],
) -> None:
    """Test that tags are returned in sorted order."""
    minimal_text_record["tags"] = ["zebra", "apple", "banana"]
    minimal_text_record["style"] = "instruction"

    result = validator.validate_and_transform({"test-model": minimal_text_record})
    base = result["test-model"]

    # Tags should be sorted alphabetically
    expected_order = ["7B", "apple", "banana", "instruction", "zebra"]
    assert base["tags"] == expected_order
