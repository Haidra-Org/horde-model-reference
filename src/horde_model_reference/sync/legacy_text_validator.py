"""Validator for legacy text generation format to ensure convert.py compatibility.

This module ensures that legacy text generation data synced to GitHub matches the
validation rules and transformations applied by scripts/legacy_text/convert.py.

Delegates record-level validation to TextModelWriteProcessor and duplicate generation
to TextModelDuplicateManager. This class orchestrates the batch operation:
stripping incoming duplicates, validating base records, and re-generating duplicates.
"""

from __future__ import annotations

from typing import TypeAlias

from loguru import logger

from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix
from horde_model_reference.text_model_duplicates import TextModelDuplicateManager
from horde_model_reference.text_model_write_processor import TextModelWriteProcessor

# Re-export type aliases for backwards compatibility with callers
SettingsValue: TypeAlias = int | float | str | list[int] | list[float] | list[str] | bool
SettingsDict: TypeAlias = dict[str, SettingsValue]
LegacyRecordValue: TypeAlias = str | int | float | bool | list[int] | list[float] | list[str] | SettingsDict | None
LegacyRecordDict: TypeAlias = dict[str, LegacyRecordValue]
GenerationParamsDict: TypeAlias = dict[str, int | float | str | bool | list[int]]
GenerationDefaultsDict: TypeAlias = dict[str, LegacyRecordValue]


class LegacyTextValidator:
    """Validator for legacy text generation format ensuring convert.py compatibility.

    Delegates per-record validation to TextModelWriteProcessor and
    duplicate generation to TextModelDuplicateManager.
    """

    def __init__(
        self,
        *,
        generation_params_path: str | None = None,
        defaults_path: str | None = None,
    ) -> None:
        """Initialize the legacy text validator.

        Args:
            generation_params_path: Ignored, kept for API compatibility.
            defaults_path: Ignored, kept for API compatibility.
        """
        self._processor = TextModelWriteProcessor()

        logger.debug(
            f"Initialized LegacyTextValidator with "
            f"generation_params.json ({len(self._processor.generation_params)} keys), "
            f"defaults.json ({len(self._processor.defaults)} keys)"
        )

    @property
    def generation_params(self) -> GenerationParamsDict:
        """Expose processor's generation_params for external access."""
        return self._processor.generation_params

    @property
    def defaults(self) -> GenerationDefaultsDict:
        """Expose processor's defaults for external access."""
        return self._processor.defaults

    def validate_and_transform(
        self,
        data: dict[str, LegacyRecordDict],
    ) -> dict[str, LegacyRecordDict]:
        """Validate and transform legacy text generation data.

        Strips incoming backend-prefixed entries, validates base records,
        and re-generates backend duplicates.

        Args:
            data: Dictionary of model records in legacy format.

        Returns:
            Transformed dictionary with validation rules applied,
            including regenerated backend prefix duplicates.

        Raises:
            ValueError: If validation fails (invalid settings keys, missing required fields, etc.)
        """
        logger.debug(f"Validating {len(data)} legacy text generation records (grouped format)")

        result: dict[str, LegacyRecordDict] = {}
        backend_duplicates: dict[str, LegacyRecordDict] = {}
        for model_name, record in data.items():
            if has_legacy_text_backend_prefix(model_name):
                logger.debug(
                    f"Skipping backend-prefixed entry {model_name} (backend prefixes are not stored internally)"
                )
                continue

            try:
                validated_record = self._processor.validate_and_transform(model_name, record)
                result[model_name] = validated_record
                backend_duplicates.update(
                    TextModelDuplicateManager.generate_duplicates(model_name, validated_record)
                )
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
