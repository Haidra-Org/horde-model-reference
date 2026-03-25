"""Validation and transformation for text model write operations.

Enforces the same rules that scripts/legacy_text/convert.py applies:
- Settings keys validated against generation_params.json
- Tags auto-generated from style + parameter size
- display_name auto-generated if not provided
- model_name extracted from entry key
- parameters normalized to int

Used by both the API write path and the GitHub sync validator.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

from loguru import logger

# Type aliases (shared with LegacyTextValidator)
type SettingsValue = int | float | str | list[int] | list[float] | list[str] | bool
type SettingsDict = dict[str, SettingsValue]
type LegacyRecordValue = str | int | float | bool | list[int] | list[float] | list[str] | SettingsDict | None
type LegacyRecordDict = dict[str, LegacyRecordValue]
type GenerationParamsDict = dict[str, int | float | str | bool | list[int]]
type GenerationDefaultsDict = dict[str, LegacyRecordValue]


def _load_bundled_json(filename: str) -> dict[str, LegacyRecordValue]:
    """Load a bundled JSON data file from the package data directory.

    Falls back to scripts/legacy_text/ relative to the repository root
    when running from a source checkout.

    Args:
        filename: The JSON filename to load (e.g., "generation_params.json").

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file cannot be located in either location.
    """
    # Try package data first (installed package)
    try:
        data_files = resources.files("horde_model_reference") / "data"
        data_path = data_files / filename
        content = data_path.read_text(encoding="utf-8")
        data_return_value: dict[str, LegacyRecordValue] = json.loads(content)
        return data_return_value
    except (FileNotFoundError, TypeError, ModuleNotFoundError):
        pass

    # Fall back to scripts/legacy_text/ relative to repo root
    repo_root = Path(__file__).parent.parent.parent
    scripts_path = repo_root / "scripts" / "legacy_text" / filename
    if scripts_path.exists():
        with open(scripts_path, encoding="utf-8") as f:
            repo_data_return_value: dict[str, LegacyRecordValue] = json.load(f)
            return repo_data_return_value

    raise FileNotFoundError(
        f"Cannot find {filename} in package data or scripts/legacy_text/. "
        "Ensure the package is installed correctly or running from the repository root."
    )


# Module-level singletons loaded once
_generation_params: GenerationParamsDict | None = None
_defaults: GenerationDefaultsDict | None = None


def _get_generation_params() -> GenerationParamsDict:
    """Get the generation_params.json data, loading on first access."""
    global _generation_params
    if _generation_params is None:
        raw = _load_bundled_json("generation_params.json")
        # generation_params.json values are int | float | str | bool | list[int]
        validated: GenerationParamsDict = {}
        for k, v in raw.items():
            if isinstance(v, (int, float, str)):
                validated[k] = v
            elif isinstance(v, list):
                validated[k] = [x for x in v if isinstance(x, int)]
        _generation_params = validated
        logger.debug(f"Loaded generation_params.json with {len(_generation_params)} valid setting keys")
    return _generation_params


def _get_defaults() -> GenerationDefaultsDict:
    """Get the defaults.json data, loading on first access."""
    global _defaults
    if _defaults is None:
        _defaults = _load_bundled_json("defaults.json")
        logger.debug(f"Loaded defaults.json with {len(_defaults)} default fields")
    return _defaults


def get_valid_settings_keys() -> list[str]:
    """Return the list of valid settings keys from generation_params.json.

    Useful for frontend validation hints or API documentation.

    Returns:
        Sorted list of valid setting key names.
    """
    return sorted(_get_generation_params().keys())


class TextModelWriteProcessor:
    """Validates and transforms text model records on write operations.

    Enforces convert.py rules:
    1. Settings keys must exist in generation_params.json
    2. Parameters must be a positive integer
    3. Tags auto-generated: existing + style + size bucket (e.g., "7B")
    4. display_name auto-generated from model name if not provided
    5. model_name field populated by splitting entry key on "/"
    """

    def __init__(self) -> None:
        """Initialize the processor with bundled validation data."""
        self.generation_params = _get_generation_params()
        self.defaults = _get_defaults()

    def validate_and_transform(
        self,
        entry_key: str,
        record: LegacyRecordDict,
        *,
        apply_defaults: bool = True,
    ) -> LegacyRecordDict:
        """Validate and transform a single text model record.

        Args:
            entry_key: The model name / dictionary key (used for error messages and naming).
            record: The record data to validate and transform.
            apply_defaults: Whether to apply defaults.json values for missing fields.

        Returns:
            Validated and transformed record.

        Raises:
            ValueError: If validation fails (invalid settings keys, missing parameters, etc.)
        """
        result = dict(record)

        original_style = result.get("style") if result.get("style") else None
        existing_tags = result.get("tags")

        # Normalize parameters
        parameters_value = result.get("parameters")
        normalized_parameters = self.normalize_parameters(entry_key, parameters_value)
        result["parameters"] = normalized_parameters

        # Validate and normalize settings
        if "settings" in result:
            normalized_settings = self.normalize_settings(entry_key, result.get("settings"))
            if normalized_settings is None:
                result.pop("settings", None)
            else:
                result["settings"] = normalized_settings

        # Auto-generate tags
        result["tags"] = self.generate_tags(
            parameters=normalized_parameters,
            existing_tags=existing_tags,
            style_for_tag=original_style,
        )

        # Ensure name field matches the key
        result["name"] = entry_key

        # Auto-generate display_name if not provided
        if not result.get("display_name"):
            display_source = self.extract_model_name(entry_key)
            result["display_name"] = self.generate_display_name(display_source)

        # Populate model_name field
        result["model_name"] = self.extract_model_name(entry_key)

        # Remove empty values (matching convert.py semantics)
        final_result: LegacyRecordDict = {key: value for key, value in result.items() if value}

        # Apply defaults for missing fields
        if apply_defaults:
            for key, value in self.defaults.items():
                if key not in final_result:
                    final_result[key] = value

        return final_result

    def normalize_parameters(self, entry_key: str, value: LegacyRecordValue) -> int:
        """Ensure the parameters field is present and a positive integer.

        Args:
            entry_key: Model name for error messages.
            value: The raw parameters value.

        Returns:
            Normalized integer parameter count.

        Raises:
            ValueError: If parameters is missing, non-numeric, or non-positive.
        """
        if value is None:
            raise ValueError(f"{entry_key}: 'parameters' field is required")

        if isinstance(value, bool):
            raise ValueError(f"{entry_key}: 'parameters' must be numeric")

        if isinstance(value, (int, float)):
            int_value = int(value)
            if int_value <= 0:
                raise ValueError(f"{entry_key}: 'parameters' must be positive")
            return int_value

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"{entry_key}: 'parameters' field is required")
            try:
                int_value = int(stripped)
            except ValueError:
                try:
                    int_value = int(float(stripped))
                except ValueError as exc:
                    raise ValueError(f"{entry_key}: 'parameters' must be numeric") from exc
            if int_value <= 0:
                raise ValueError(f"{entry_key}: 'parameters' must be positive")
            return int_value

        raise ValueError(f"{entry_key}: 'parameters' must be numeric")

    def normalize_settings(self, entry_key: str, value: LegacyRecordValue) -> SettingsDict | None:
        """Validate and normalize the settings dict.

        All keys must exist in generation_params.json.

        Args:
            entry_key: Model name for error messages.
            value: The raw settings value (dict, JSON string, or None).

        Returns:
            Normalized settings dict, or None if empty/absent.

        Raises:
            ValueError: If settings is not a dict or contains invalid keys.
        """
        if value is None:
            return None

        parsed_from_json = False

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                value = json.loads(stripped)
                parsed_from_json = True
            except json.JSONDecodeError as exc:
                raise ValueError(f"{entry_key}: settings must be valid JSON") from exc

        if value is None:
            if parsed_from_json:
                raise ValueError(f"{entry_key}: settings must be a JSON dictionary")
            return None

        if not isinstance(value, dict):
            raise ValueError(f"{entry_key}: settings must be a JSON dictionary")

        invalid_keys = [key for key in value if key not in self.generation_params]
        if invalid_keys:
            raise ValueError(
                f"{entry_key}: settings contains invalid keys: {invalid_keys}. "
                f"Valid keys are: {sorted(self.generation_params.keys())}"
            )

        return value

    def generate_tags(
        self,
        *,
        parameters: int,
        existing_tags: LegacyRecordValue,
        style_for_tag: LegacyRecordValue,
    ) -> list[str]:
        """Generate tags following convert.py rules.

        Includes existing tags, the style (if provided), and a size tag
        derived from the parameter count.

        Args:
            parameters: The parameter count (used to derive size tag like "7B").
            existing_tags: Tags from the incoming record (list or comma-separated string).
            style_for_tag: Style value from the incoming record (added as tag if truthy).

        Returns:
            Sorted list of unique tags.
        """
        tags_set: set[str] = set()

        if existing_tags:
            if isinstance(existing_tags, list):
                for tag in existing_tags:
                    if tag and str(tag).strip():
                        tags_set.add(str(tag).strip())
            elif isinstance(existing_tags, str):
                tags_set.update(t.strip() for t in existing_tags.split(",") if t and t.strip())
            else:
                raise ValueError("tags must be provided as a list or comma-separated string")

        if style_for_tag:
            tags_set.add(str(style_for_tag))

        params_bn = float(parameters) / 1_000_000_000
        size_tag = f"{round(params_bn, 0):.0f}B"
        tags_set.add(size_tag)

        return sorted(tags_set)

    @staticmethod
    def extract_model_name(entry_key: str) -> str:
        """Extract model_name following convert.py's splitting logic.

        For "Author/ModelName", returns "ModelName".
        For "ModelName", returns "ModelName".

        Args:
            entry_key: The full model name / dict key.

        Returns:
            The model name portion (after the first "/" if present).

        Raises:
            ValueError: If ``entry_key`` looks like a URL (contains ``://``).
        """
        if "://" in entry_key:
            raise ValueError(
                f"Cannot extract model_name from URL-shaped key: {entry_key!r}. "
                "Model names should use 'Author/ModelName' format, not full URLs."
            )
        if "/" in entry_key:
            return entry_key.split("/")[1]
        return entry_key

    @staticmethod
    def generate_display_name(model_name: str) -> str:
        """Generate a human-readable display name from a model name.

        Replaces hyphens and underscores with spaces, normalizes whitespace.

        Args:
            model_name: The raw model name to convert.

        Returns:
            Human-readable display name.

        Example:
            >>> TextModelWriteProcessor.generate_display_name("llama-2-7b-chat")
            'llama 2 7b chat'
        """
        display_name = re.sub(r"[-_]", " ", model_name)
        display_name = re.sub(r" +", " ", display_name)
        return display_name.strip()
