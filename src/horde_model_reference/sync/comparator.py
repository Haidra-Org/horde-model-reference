"""Comparison engine for detecting differences between PRIMARY and GitHub model references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from horde_model_reference import MODEL_REFERENCE_CATEGORY


@dataclass
class ModelReferenceDiff:
    """Represents the complete diff between PRIMARY and GitHub model references."""

    category: MODEL_REFERENCE_CATEGORY

    added_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Models present in PRIMARY but not in GitHub."""

    removed_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Models present in GitHub but not in PRIMARY."""

    modified_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Models present in both but with different content. Contains PRIMARY version."""

    def has_changes(self) -> bool:
        """Return True if there are any differences between PRIMARY and GitHub."""
        return bool(self.added_models or self.removed_models or self.modified_models)

    def total_changes(self) -> int:
        """Return the total number of changes (added + removed + modified)."""
        return len(self.added_models) + len(self.removed_models) + len(self.modified_models)

    def summary(self) -> str:
        """Return a human-readable summary of the diff."""
        if not self.has_changes():
            return f"No differences detected for {self.category}"

        lines = [f"Differences for {self.category}:"]

        if self.added_models:
            lines.append(f"  Added: {len(self.added_models)} models")
            for model_name in sorted(self.added_models.keys())[:5]:
                lines.append(f"    + {model_name}")
            if len(self.added_models) > 5:
                lines.append(f"    ... and {len(self.added_models) - 5} more")

        if self.removed_models:
            lines.append(f"  Removed: {len(self.removed_models)} models")
            for model_name in sorted(self.removed_models.keys())[:5]:
                lines.append(f"    - {model_name}")
            if len(self.removed_models) > 5:
                lines.append(f"    ... and {len(self.removed_models) - 5} more")

        if self.modified_models:
            lines.append(f"  Modified: {len(self.modified_models)} models")
            for model_name in sorted(self.modified_models.keys())[:5]:
                lines.append(f"    ~ {model_name}")
            if len(self.modified_models) > 5:
                lines.append(f"    ... and {len(self.modified_models) - 5} more")

        return "\n".join(lines)


class ModelReferenceComparator:
    """Compares PRIMARY and GitHub model references to detect differences."""

    def compare_categories(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY,
        primary_data: dict[str, dict[str, Any]],
        github_data: dict[str, dict[str, Any]],
    ) -> ModelReferenceDiff:
        """Compare PRIMARY and GitHub data for a specific category.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being compared.
            primary_data: The model reference data from PRIMARY (v1 API).
            github_data: The model reference data from GitHub (legacy format).

        Returns:
            A ModelReferenceDiff object containing all detected differences.
        """
        logger.debug(f"Comparing {category}: {len(primary_data)} PRIMARY models vs {len(github_data)} GitHub models")

        diff = ModelReferenceDiff(category=category)

        primary_keys = set(primary_data.keys())
        github_keys = set(github_data.keys())

        diff.added_models = {key: primary_data[key] for key in primary_keys - github_keys}
        logger.debug(f"Found {len(diff.added_models)} added models in {category}")
        logger.debug(f"Added models: {list(diff.added_models.keys())}")
        diff.removed_models = {key: github_data[key] for key in github_keys - primary_keys}
        logger.debug(f"Found {len(diff.removed_models)} removed models in {category}")
        logger.debug(f"Removed models: {list(diff.removed_models.keys())}")

        for model_name in primary_keys & github_keys:
            primary_model = primary_data[model_name]
            github_model = github_data[model_name]

            if primary_model != github_model:
                diff.modified_models[model_name] = primary_model
                logger.debug(f"Model modified: {model_name}")
                logger.debug(f"PRIMARY version:\n{primary_model}\n\n")
                logger.debug(f"GitHub version:\n{github_model}\n\n")

        logger.debug(f"Found {len(diff.modified_models)} modified models in {category}")
        logger.debug(f"Modified models: {list(diff.modified_models.keys())}")

        logger.info(f"Comparison complete: {diff.total_changes()} total changes")
        return diff
