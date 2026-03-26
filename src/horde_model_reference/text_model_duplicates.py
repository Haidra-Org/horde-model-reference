"""Text model backend duplicate management.

Handles generation and cleanup of backend-prefixed duplicate entries
(aphrodite/, koboldcpp/) in the legacy format. These duplicates exist
for backwards compatibility with the Horde API.

Variant name computation is delegated to ``text_backend_names.get_model_name_variants``
which is the single source of truth for the naming rules.
"""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

from horde_model_reference.text_backend_names import (
    get_model_name_variants,
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
        Delegates to ``get_model_name_variants`` which is the single
        source of truth for variant naming rules.

        Args:
            base_name: The canonical model name (e.g., "Author/ModelName" or "ModelName").

        Returns:
            List of backend-prefixed variant names.

        Example:
            >>> TextModelDuplicateManager.get_variant_names("ReadyArt/Broken-Tutu-24B")
            ['aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B', 'koboldcpp/ReadyArt_Broken-Tutu-24B']

        """
        all_variants = get_model_name_variants(base_name)
        return all_variants[1:]

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
        return get_model_name_variants(base_name)

    @staticmethod
    def generate_duplicates(
        base_name: str,
        record: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Generate backend-prefixed duplicate entries from a base model record.

        Each duplicate is a deep copy of the base record with the ``name`` field
        updated to the prefixed variant. Uses ``get_variant_names`` so all
        variant rules (including the legacy flattened koboldcpp form) are applied.

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
            ['aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B', 'koboldcpp/ReadyArt_Broken-Tutu-24B']

        """
        result: dict[str, dict[str, Any]] = {}

        for variant_name in TextModelDuplicateManager.get_variant_names(base_name):
            prefixed_record = copy.deepcopy(record)
            prefixed_record["name"] = variant_name
            result[variant_name] = prefixed_record

        logger.trace(f"Generated {len(result)} backend duplicates for '{base_name}': {sorted(result.keys())}")
        return result

    @staticmethod
    def strip_duplicates_from_data(data: dict[str, Any]) -> dict[str, Any]:
        """Remove all backend-prefixed entries from a data dict, keeping only base models.

        Args:
            data: Dictionary of model records (name → record).

        Returns:
            New dictionary with only base (non-prefixed) model entries.

        """
        return {name: record for name, record in data.items() if not has_legacy_text_backend_prefix(name)}

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
