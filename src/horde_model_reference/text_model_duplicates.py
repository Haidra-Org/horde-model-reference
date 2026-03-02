"""Text model backend duplicate management.

Handles generation and cleanup of backend-prefixed duplicate entries
(aphrodite/, koboldcpp/) in the legacy format. These duplicates exist
for backwards compatibility with the Horde API.
"""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from horde_model_reference.meta_consts import TEXT_BACKENDS
from horde_model_reference.text_backend_names import (
    TEXT_LEGACY_BACKEND_PREFIXES,
    has_legacy_text_backend_prefix,
)


class TextModelDuplicateManager:
    """Generates and manages backend-prefixed duplicate entries for text models.

    In the legacy format, each text model has duplicate entries with aphrodite/
    and koboldcpp/ prefixes. This class handles:
    - Generating those duplicates from a base model entry
    - Computing variant names for a given base model
    - Cleaning up duplicates when a base model is deleted
    """

    @staticmethod
    def get_variant_names(base_name: str) -> list[str]:
        """Get all backend-prefixed variant names for a base model.

        Does NOT include the base name itself in the returned list.

        Args:
            base_name: The canonical model name (e.g., "Author/ModelName" or "ModelName").

        Returns:
            List of backend-prefixed variant names.

        Example:
            >>> TextModelDuplicateManager.get_variant_names("ReadyArt/Broken-Tutu-24B")
            ['aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B']
        """
        variants: list[str] = []

        # model_name is the part after "/" (or the full name if no "/")
        model_name_only = base_name.split("/", 1)[1] if "/" in base_name else base_name

        for backend, prefix in TEXT_LEGACY_BACKEND_PREFIXES.items():
            if backend == TEXT_BACKENDS.aphrodite:
                variants.append(f"{prefix}{base_name}")
            elif backend == TEXT_BACKENDS.koboldcpp:
                prefixed = f"{prefix}{model_name_only}"
                variants.append(prefixed)
                # Also include the flattened variant if the base name contains "/"
                sanitized = base_name.replace("/", "_")
                if sanitized not in {base_name, model_name_only}:
                    variants.append(f"{prefix}{sanitized}")

        return variants

    @staticmethod
    def get_all_names(base_name: str) -> list[str]:
        """Get base name plus all backend-prefixed variant names.

        Args:
            base_name: The canonical model name.

        Returns:
            List starting with base_name followed by all variant names.

        Example:
            >>> TextModelDuplicateManager.get_all_names("Author/Model-7B")
            ['Author/Model-7B', 'aphrodite/Author/Model-7B', 'koboldcpp/Model-7B', 'koboldcpp/Author_Model-7B']
        """
        return [base_name, *TextModelDuplicateManager.get_variant_names(base_name)]

    @staticmethod
    def generate_duplicates(
        base_name: str,
        record: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Generate backend-prefixed duplicate entries from a base model record.

        Each duplicate is a copy of the base record with the "name" field updated
        to the prefixed variant.

        Args:
            base_name: The canonical model name (e.g., "ReadyArt/Broken-Tutu-24B").
            record: The base model record dict.

        Returns:
            Dictionary mapping prefixed names to their record dicts.

        Example:
            >>> dupes = TextModelDuplicateManager.generate_duplicates(
            ...     "ReadyArt/Broken-Tutu-24B",
            ...     {"name": "ReadyArt/Broken-Tutu-24B", "parameters": 24000000000}
            ... )
            >>> sorted(dupes.keys())
            ['aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B']
        """
        result: dict[str, dict[str, Any]] = {}

        model_name_only = base_name.split("/", 1)[1] if "/" in base_name else base_name

        for backend, prefix in TEXT_LEGACY_BACKEND_PREFIXES.items():
            if backend == TEXT_BACKENDS.aphrodite:
                prefixed_name = f"{prefix}{base_name}"
            elif backend == TEXT_BACKENDS.koboldcpp:
                prefixed_name = f"{prefix}{model_name_only}"
            else:
                continue

            prefixed_record = copy.deepcopy(record)
            prefixed_record["name"] = prefixed_name
            result[prefixed_name] = prefixed_record

        logger.trace(
            f"Generated {len(result)} backend duplicates for '{base_name}': {sorted(result.keys())}"
        )
        return result

    @staticmethod
    def strip_duplicates_from_data(data: dict[str, Any]) -> dict[str, Any]:
        """Remove all backend-prefixed entries from a data dict, keeping only base models.

        Args:
            data: Dictionary of model records (name → record).

        Returns:
            New dictionary with only base (non-prefixed) model entries.
        """
        return {
            name: record
            for name, record in data.items()
            if not has_legacy_text_backend_prefix(name)
        }

    @staticmethod
    def find_existing_variants(
        base_name: str,
        data: dict[str, Any],
    ) -> list[str]:
        """Find which backend-prefixed variants of a base model exist in the data.

        Args:
            base_name: The canonical model name.
            data: Dictionary of all model records.

        Returns:
            List of variant names that exist in the data dict.
        """
        variant_names = TextModelDuplicateManager.get_variant_names(base_name)
        return [name for name in variant_names if name in data]
