from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator

Severity = Literal["error", "warning"]
Cardinality = Literal["single", "list"]


@dataclass(frozen=True)
class FieldPolicy:
    """Per-field validation policy for a category."""

    cardinality: Cardinality = "single"
    severity: Severity = "warning"


@dataclass
class KindPolicy:
    """Collection of field policies for a category."""

    field_policies: dict[str, FieldPolicy] = field(default_factory=dict)


class KindPolicyRegistry:
    """Registry for category-specific validation policies."""

    def __init__(self) -> None:
        """Initialize an empty KindPolicyRegistry."""
        self._policies: dict[str, KindPolicy] = {}

    def register(self, category: str, policy: KindPolicy) -> None:
        """Register a KindPolicy for a specific category.

        Args:
            category: The model reference category to associate with the policy.
            policy: The KindPolicy instance containing field validation rules.

        Raises:
            ValueError: If a policy is already registered for the given category.
        """
        if category in self._policies:
            raise ValueError(f"Policy already registered for {category!r}")
        self._policies[category] = policy

    def get(self, category: str) -> KindPolicy | None:
        """Retrieve the KindPolicy for a given category, or None if no policy is registered.

        Args:
            category: The model reference category to look up.

        Returns:
            The KindPolicy associated with the category, or None if not found.
        """
        return self._policies.get(category)


def category_key(category: str | Enum) -> str:
    """Normalize category identifiers to a string key for the registry."""
    return str(category)


kind_policy_registry = KindPolicyRegistry()


def _strip_value(value: str) -> str:
    return value.strip()


NormalizedModelStyle = Annotated[str, AfterValidator(_strip_value)]
NormalizedTag = Annotated[str, AfterValidator(_strip_value)]
