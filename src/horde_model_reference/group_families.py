"""File-backed persistence for related-group families.

Provides a lightweight association layer between distinct model groups that
share a conceptual relationship - for example, Mistral-Large, Mistral-Small,
and Mistral-Nemo all belong to the "Mistral" family even though they are
architecturally distinct models with their own groups.

Families are purely informational metadata. They do not affect base name
extraction, alias resolution, or group membership. They exist so that UIs
and analytics can surface "see also" relationships between groups.

Public API:
    ``GroupFamily``       - Pydantic model representing a single family.
    ``GroupFamilyStore``  - Thread-safe, file-backed CRUD for families.
    ``detect_families``   - Heuristic prefix-based family suggestion from group names.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from threading import RLock

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference.util import atomic_write_json


class GroupFamily(BaseModel):
    """Represents a named family of related model groups.

    Attributes:
        family_name: Human-readable identifier for this family (e.g., ``"Mistral"``).
        members: Group base-names that belong to this family.

    """

    model_config = ConfigDict(extra="forbid")

    family_name: str
    members: list[str] = Field(default_factory=list)


class GroupFamilyStore:
    """Thread-safe, file-backed store for related-group families.

    Each family maps a descriptive name to a set of group base-names.
    A group may belong to at most one family - assigning it to a second
    family raises ``ValueError``.

    The reverse index (``_group_to_family``) enables O(1) lookups from
    any group name to its family.

    Attributes:
        _file_path: Filesystem location for JSON persistence.
        _families: family_name -> ``GroupFamily`` mapping.
        _group_to_family: group_name -> family_name reverse index.

    """

    def __init__(self, *, file_path: Path) -> None:
        """Construct the store and load persisted state if the file exists.

        Args:
            file_path: Path to the JSON file used for persistence.

        """
        self._file_path = file_path
        self._lock = RLock()
        self._families: dict[str, GroupFamily] = {}
        self._group_to_family: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Read persisted families from disk, rebuilding the reverse index."""
        if not self._file_path.exists():
            return

        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Failed to load group families from {self._file_path}")
            return

        for family_name, entry in raw.items():
            family = GroupFamily.model_validate(entry)
            self._families[family_name] = family
            for member in family.members:
                self._group_to_family[member] = family_name

    def _persist(self) -> None:
        """Write current state to disk atomically."""
        payload = {
            family_name: family.model_dump(mode="json") for family_name, family in sorted(self._families.items())
        }
        atomic_write_json(self._file_path, payload)

    def get_family(self, family_name: str) -> GroupFamily | None:
        """Return a family by name, or ``None`` if not found.

        Args:
            family_name: The family identifier to look up.

        Returns:
            The matching ``GroupFamily`` or ``None``.

        """
        with self._lock:
            family = self._families.get(family_name)
            if family is None:
                return None
            return family.model_copy(deep=True)

    def get_family_for_group(self, group_name: str) -> GroupFamily | None:
        """Return the family that *group_name* belongs to, or ``None``.

        Args:
            group_name: A model group base-name.

        Returns:
            A deep copy of the family, or ``None`` if the group has no family.

        """
        with self._lock:
            family_name = self._group_to_family.get(group_name)
            if family_name is None:
                return None
            family = self._families.get(family_name)
            if family is None:
                return None
            return family.model_copy(deep=True)

    def list_all(self) -> dict[str, GroupFamily]:
        """Return a deep copy of every family.

        Returns:
            Mapping of family_name -> ``GroupFamily`` (copies).

        """
        with self._lock:
            return {family_name: family.model_copy(deep=True) for family_name, family in self._families.items()}

    def set_family(self, family_name: str, member_groups: list[str]) -> None:
        """Create or replace a family, assigning the given members.

        Any previous members of this family that are not in *member_groups*
        are released. New members must not already belong to a different family.

        Args:
            family_name: Identifier for the family.
            member_groups: Group base-names to include.

        Raises:
            ValueError: If a member already belongs to a different family,
                or if *member_groups* is empty.

        """
        if not member_groups:
            raise ValueError("A family must have at least one member group.")

        with self._lock:
            # Validate that no member is already claimed by another family
            for group_name in member_groups:
                existing_family = self._group_to_family.get(group_name)
                if existing_family is not None and existing_family != family_name:
                    raise ValueError(
                        f"Group '{group_name}' already belongs to family '{existing_family}'. "
                        f"Remove it from that family first."
                    )

            # Release any previously assigned members of this family
            old_family = self._families.get(family_name)
            if old_family is not None:
                for old_member in old_family.members:
                    if old_member not in member_groups:
                        self._group_to_family.pop(old_member, None)

            family = GroupFamily(family_name=family_name, members=sorted(set(member_groups)))
            self._families[family_name] = family
            for group_name in family.members:
                self._group_to_family[group_name] = family_name

            self._persist()
            logger.debug(f"Set family '{family_name}' with {len(family.members)} members")

    def add_member(self, family_name: str, group_name: str) -> None:
        """Add a single group to an existing family.

        Args:
            family_name: The target family.
            group_name: The group to add.

        Raises:
            KeyError: If *family_name* does not exist.
            ValueError: If *group_name* already belongs to a different family.

        """
        with self._lock:
            family = self._families.get(family_name)
            if family is None:
                raise KeyError(f"Family '{family_name}' does not exist.")

            existing = self._group_to_family.get(group_name)
            if existing is not None and existing != family_name:
                raise ValueError(f"Group '{group_name}' already belongs to family '{existing}'.")

            if group_name not in family.members:
                family.members = sorted(set(family.members) | {group_name})
                self._group_to_family[group_name] = family_name
                self._persist()

    def remove_member(self, family_name: str, group_name: str) -> bool:
        """Remove a single group from a family.

        If the family becomes empty after removal, it is deleted entirely.

        Args:
            family_name: The family to modify.
            group_name: The group to remove.

        Returns:
            ``True`` if the group was found and removed.

        """
        with self._lock:
            family = self._families.get(family_name)
            if family is None or group_name not in family.members:
                return False

            family.members = [m for m in family.members if m != group_name]
            self._group_to_family.pop(group_name, None)

            if not family.members:
                del self._families[family_name]
                logger.debug(f"Deleted empty family '{family_name}'")
            else:
                self._families[family_name] = family

            self._persist()
            return True

    def delete(self, family_name: str) -> bool:
        """Delete an entire family, releasing all its members.

        Args:
            family_name: The family to delete.

        Returns:
            ``True`` if the family existed and was deleted.

        """
        with self._lock:
            family = self._families.pop(family_name, None)
            if family is None:
                return False

            for member in family.members:
                self._group_to_family.pop(member, None)

            self._persist()
            logger.debug(f"Deleted family '{family_name}' ({len(family.members)} members released)")
            return True


def detect_families(
    group_names: list[str],
    *,
    min_prefix_length: int = 3,
    min_family_size: int = 2,
) -> dict[str, list[str]]:
    """Suggest family groupings from group names using shared prefix heuristics.

    Groups that share a common hyphen-delimited prefix are candidates for the
    same family. When shorter and longer prefixes overlap, the longest prefix
    that captures at least *min_family_size* groups wins, and its members are
    excluded from shorter prefix families.

    This is a suggestion engine - results should be reviewed by an admin before
    persisting via ``GroupFamilyStore.set_family()``.

    Args:
        group_names: Distinct group base-names to analyze.
        min_prefix_length: Minimum character length for a prefix to be considered.
        min_family_size: Minimum number of groups required to form a family.

    Returns:
        Mapping of suggested family prefix -> list of member group names.

    """
    prefix_to_groups: dict[str, set[str]] = defaultdict(set)

    for group_name in group_names:
        parts = group_name.split("-")
        for depth in range(len(parts) - 1, 0, -1):
            prefix = "-".join(parts[:depth])
            if len(prefix) >= min_prefix_length:
                prefix_to_groups[prefix].add(group_name)

    # Deduplicate: longest prefix wins, claim its members
    sorted_prefixes: list[str] = sorted(prefix_to_groups.keys(), key=len, reverse=True)
    claimed: set[str] = set()
    result: dict[str, list[str]] = {}

    for prefix in sorted_prefixes:
        unclaimed = sorted(prefix_to_groups[prefix] - claimed)
        if len(unclaimed) >= min_family_size:
            result[prefix] = unclaimed
            claimed.update(unclaimed)

    return dict(sorted(result.items()))
