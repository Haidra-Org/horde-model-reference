"""Validator for legacy text generation format to ensure convert.py compatibility.

This module ensures that legacy text generation data synced to GitHub matches the
validation rules and transformations applied by scripts/legacy_text/convert.py.

The legacy GitHub repository had a CI process that ran convert.py to validate
models.csv and db.json consistency. Now that the PRIMARY instance with
canonical_format='legacy' is the source of truth, we need to ensure the data
we sync to GitHub would pass those same validation checks.

Key Responsibilities:
    - Validate settings keys against generation_params.json
    - Apply defaults from defaults.json to all records
    - Auto-generate tags (style tag + parameter size tag, e.g., "7B")
    - Generate backend prefix duplicate entries (aphrodite/, koboldcpp/)
    - Auto-generate display_name if not provided
    - Ensure all records have required fields from defaults.json

This maintains backwards compatibility with the old GitHub CI validation while
allowing the PRIMARY instance to be the authoritative source.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeAlias

from loguru import logger

from horde_model_reference.meta_consts import (
    _TEXT_LEGACY_CONVERT_BACKEND_PREFIXES,
    TEXT_BACKENDS,
)

# Type aliases for legacy text generation record structures
SettingsValue: TypeAlias = int | float | str | list[int] | list[float] | list[str] | bool
SettingsDict: TypeAlias = dict[str, SettingsValue]
# Legacy record values include all possible types from defaults, settings, and other fields
LegacyRecordValue: TypeAlias = str | int | float | bool | list[int] | list[float] | list[str] | SettingsDict | None
LegacyRecordDict: TypeAlias = dict[str, LegacyRecordValue]
# Generation params and defaults are subsets of LegacyRecordValue
GenerationParamsDict: TypeAlias = dict[str, int | float | str | bool | list[int]]
GenerationDefaultsDict: TypeAlias = dict[str, LegacyRecordValue]


class LegacyTextValidator:
    """Validator for legacy text generation format ensuring convert.py compatibility.

    This validator enforces the same rules that scripts/legacy_text/convert.py used to enforce:
    1. Settings keys must exist in generation_params.json
    2. All records get fields from defaults.json merged in
    3. Tags are auto-generated: style tag + parameter size tag (e.g., "7B")
    4. Backend prefix duplicates are created (base, aphrodite/, koboldcpp/)
    5. Display names are auto-generated if missing
    """

    def __init__(
        self,
        *,
        generation_params_path: str | Path | None = None,
        defaults_path: str | Path | None = None,
    ) -> None:
        """Initialize the legacy text validator.

        Args:
            generation_params_path: Path to generation_params.json. If None, uses
                scripts/legacy_text/generation_params.json from the repository root.
            defaults_path: Path to defaults.json. If None, uses
                scripts/legacy_text/defaults.json from the repository root.
        """
        # Default paths relative to repository root
        if generation_params_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            generation_params_path = repo_root / "scripts" / "legacy_text" / "generation_params.json"

        if defaults_path is None:
            repo_root = Path(__file__).parent.parent.parent.parent
            defaults_path = repo_root / "scripts" / "legacy_text" / "defaults.json"

        self.generation_params_path = Path(generation_params_path)
        self.defaults_path = Path(defaults_path)

        # Load validation data
        self.generation_params = self._load_json(self.generation_params_path)
        self.defaults = self._load_json(self.defaults_path)

        logger.debug(
            f"Initialized LegacyTextValidator with "
            f"generation_params.json ({len(self.generation_params)} keys), "
            f"defaults.json ({len(self.defaults)} keys)"
        )

    def _load_json(self, path: Path) -> GenerationParamsDict | GenerationDefaultsDict:
        """Load and parse a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            Parsed JSON data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file isn't valid JSON.
        """
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data: GenerationParamsDict | GenerationDefaultsDict = json.load(f)
            return data

    def validate_and_transform(
        self,
        data: dict[str, LegacyRecordDict],
    ) -> dict[str, LegacyRecordDict]:
        """Validate and transform legacy text generation data.

        This method applies convert.py validation rules and regenerates backend prefix duplicates:
        1. Validates settings against generation_params.json
        2. Applies defaults.json to all records
        3. Ensures tags include style and parameter size
        4. Auto-generates display_name if missing

        Args:
            data: Dictionary of model records in legacy format.

        Returns:
            Transformed dictionary with validation rules applied, including regenerated backend prefix duplicates.

        Raises:
            ValueError: If validation fails (invalid settings keys, missing required fields, etc.)
        """
        logger.debug(f"Validating {len(data)} legacy text generation records (grouped format)")

        # Validate and transform each base record
        result: dict[str, LegacyRecordDict] = {}
        backend_duplicates: dict[str, LegacyRecordDict] = {}
        for model_name, record in data.items():
            # Skip backend-prefixed entries in input (not stored internally anymore)
            if self._has_backend_prefix(model_name):
                logger.debug(
                    f"Skipping backend-prefixed entry {model_name} (backend prefixes are not stored internally)"
                )
                continue

            try:
                validated_record = self._validate_single_record(model_name, record)
                result[model_name] = validated_record
                backend_duplicates.update(self._generate_backend_duplicates(model_name, validated_record))
            except ValueError as e:
                logger.error(f"Validation failed for {model_name}: {e}")
                raise

        combined_result = dict(result)
        for duplicate_name, duplicate_record in backend_duplicates.items():
            if duplicate_name in combined_result:
                logger.warning(f"Skipping duplicate entry {duplicate_name} to avoid overriding an existing record")
                continue
            combined_result[duplicate_name] = duplicate_record

        logger.debug(
            "Validated %d base records and generated %d backend duplicates",
            len(result),
            len(backend_duplicates),
        )

        return combined_result

    def _has_backend_prefix(self, model_name: str) -> bool:
        """Check if a model name has a backend prefix.

        Args:
            model_name: The model name to check.

        Returns:
            True if the model name has a backend prefix (aphrodite/ or koboldcpp/).
        """
        return any(model_name.startswith(prefix) for prefix in _TEXT_LEGACY_CONVERT_BACKEND_PREFIXES.values())

    def _validate_single_record(
        self,
        entry_key: str,
        record: LegacyRecordDict,
    ) -> LegacyRecordDict:
        """Validate and transform a single record.

        Args:
            entry_key: The dictionary key for the record (used for error messages and naming).
            record: The record data to validate.

        Returns:
            Validated and transformed record.

        Raises:
            ValueError: If validation fails.
        """
        result = dict(record)

        # Keep track of original field values to mirror convert.py semantics
        original_style_value = result.get("style")
        original_style = original_style_value if original_style_value else None
        existing_tags_value = result.get("tags")

        # Normalize parameters before they are used anywhere else
        parameters_value = result.get("parameters")
        normalized_parameters = self._normalize_parameters(entry_key, parameters_value)
        result["parameters"] = normalized_parameters

        # Normalize settings to align with convert.py behaviour
        if "settings" in result:
            settings_value = result.get("settings")
            normalized_settings = self._normalize_settings(entry_key, settings_value)
            if normalized_settings is None:
                result.pop("settings", None)
            else:
                result["settings"] = normalized_settings

        # Generate tags using only the style supplied in the input (not defaults)
        result["tags"] = self._generate_tags(
            parameters=normalized_parameters,
            existing_tags=existing_tags_value,
            style_for_tag=original_style,
        )

        # Ensure name field matches the key
        result["name"] = entry_key

        # Auto-generate display_name if not provided (uses extracted model name)
        if not result.get("display_name"):
            display_source = self._extract_model_name(entry_key)
            result["display_name"] = self._generate_display_name(display_source)

        # Remove empty values before defaults are applied, matching convert.py
        result = self._remove_empty_values(result)

        # Apply defaults for any missing fields
        for key, value in self.defaults.items():
            if key not in result:
                result[key] = value

        # Compute model_name using the same rule as convert.py
        result["model_name"] = self._extract_model_name(entry_key)

        return result

    def _generate_tags(
        self,
        *,
        parameters: int,
        existing_tags: LegacyRecordValue,
        style_for_tag: LegacyRecordValue,
    ) -> list[str]:
        """Generate tags for a record following convert.py rules.

        Tags include existing tags, the original style (if provided), and the
        rounded parameter size tag.

        Args:
            parameters: The parameter count used to derive the size tag.
            existing_tags: Tags provided on the incoming record (list or comma-separated string).
            style_for_tag: Style value supplied on the incoming record (ignored if falsey).

        Returns:
            Sorted list of tags.
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

    def _normalize_parameters(self, entry_key: str, value: LegacyRecordValue) -> int:
        """Ensure the parameters field is present and numeric."""
        if value is None:
            raise ValueError(f"{entry_key}: 'parameters' field is required")

        if isinstance(value, bool):
            raise ValueError(f"{entry_key}: 'parameters' must be numeric")

        if isinstance(value, (int, float)):
            return int(value)

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"{entry_key}: 'parameters' field is required")

            try:
                return int(stripped)
            except ValueError:
                try:
                    return int(float(stripped))
                except ValueError as exc:
                    raise ValueError(f"{entry_key}: 'parameters' must be numeric") from exc

        raise ValueError(f"{entry_key}: 'parameters' must be numeric")

    def _normalize_settings(self, entry_key: str, value: LegacyRecordValue) -> SettingsDict | None:
        """Normalize the settings field to match convert.py expectations."""
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
                f"Valid keys are: {list(self.generation_params.keys())}"
            )

        return value

    def _remove_empty_values(self, record: LegacyRecordDict) -> LegacyRecordDict:
        """Remove falsey values to mimic convert.py's filtering semantics."""
        return {key: value for key, value in record.items() if value}

    def _extract_model_name(self, entry_key: str) -> str:
        """Extract the model_name following convert.py's splitting logic."""
        if "/" in entry_key:
            return entry_key.split("/")[1]
        return entry_key

    def _generate_display_name(self, model_name: str) -> str:
        """Generate display name following convert.py rules.

        Converts underscores and hyphens to spaces, then normalizes whitespace.

        Args:
            model_name: The model name to generate display name from.

        Returns:
            Generated display name.

        Example:
            "llama-2-7b-chat" -> "llama 2 7b chat"
            "gpt_2_medium" -> "gpt 2 medium"
        """
        # Replace hyphens and underscores with spaces
        display_name = re.sub(r"[-_]", " ", model_name)
        # Normalize multiple spaces to single space
        display_name = re.sub(r" +", " ", display_name)
        return display_name.strip()

    def _generate_backend_duplicates(
        self,
        base_name: str,
        record: LegacyRecordDict,
    ) -> dict[str, LegacyRecordDict]:
        """Generate backend prefix duplicate entries.

        Creates entries for aphrodite/ and koboldcpp/ prefixes following convert.py rules.

        Args:
            base_name: The base model name (without prefix).
            record: The base record data.

        Returns:
            Dictionary with backend-prefixed entries.

        Example:
            Input: "llama-2-7b-chat", {...}
            Output: {
                "aphrodite/llama-2-7b-chat": {..., "name": "aphrodite/llama-2-7b-chat"},
                "koboldcpp/llama-2-7b-chat": {..., "name": "koboldcpp/llama-2-7b-chat"}
            }
            Note: koboldcpp uses the raw model_name, not with spaces
        """
        result = {}

        # Get model_name for koboldcpp prefix (raw, not with spaces)
        model_name = record.get("model_name", base_name)

        # Generate entries with backend prefixes
        for backend, prefix in _TEXT_LEGACY_CONVERT_BACKEND_PREFIXES.items():
            if backend == TEXT_BACKENDS.aphrodite:
                # aphrodite uses base_name
                prefixed_name = f"{prefix}{base_name}"
            elif backend == TEXT_BACKENDS.koboldcpp:
                # koboldcpp uses model_name (raw, not display_name)
                prefixed_name = f"{prefix}{model_name}"
            else:
                continue

            # Create a copy of the record with the prefixed name
            prefixed_record = dict(record)
            prefixed_record["name"] = prefixed_name

            result[prefixed_name] = prefixed_record

        return result
