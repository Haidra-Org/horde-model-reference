"""File-backed persistence for text model group aliases.

Provides a canonical-name → alias mapping so that distinct parser base names
(e.g., ``c4ai-command-r``, ``c4ai-command-r-plus``, ``c4ai-command-a``) can
be collapsed into a single group during grouping operations.

This is the correct mechanism for model families whose names differ in ways
the parser cannot resolve heuristically — the alias is always an explicit
admin decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference.util import atomic_write_json


class GroupAlias(BaseModel):
    """A mapping from one canonical group name to its known aliases."""

    model_config = ConfigDict(extra="forbid")

    canonical: str
    """The target group name that all aliases resolve to."""

    aliases: list[str] = Field(default_factory=list)
    """Base names that should be treated as members of the canonical group."""


class GroupAliasStore:
    """Thread-safe file-backed storage for group alias mappings.

    Each entry maps a *canonical* group name to a list of *alias* base names.
    When the grouping layer calls :meth:`resolve`, any alias is transparently
    mapped back to its canonical name, collapsing multiple parser-produced
    groups into one.
    """

    def __init__(self, *, file_path: Path) -> None:
        """Create or load a store backed by the given JSON file.

        Args:
            file_path: Path to the JSON persistence file.

        """
        self._file_path = file_path
        self._lock = RLock()
        self._entries: dict[str, GroupAlias] = {}
        # Reverse index: alias → canonical (built from _entries on load/mutate)
        self._alias_to_canonical: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return
        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
            for canonical, data in raw.items():
                self._entries[canonical] = GroupAlias.model_validate({"canonical": canonical, **data})
            self._rebuild_reverse_index()
            logger.debug(f"Loaded {len(self._entries)} group alias entries from {self._file_path}")
        except Exception:
            logger.exception(f"Failed to load group aliases from {self._file_path}")

    def _persist(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {canonical: {"aliases": entry.aliases} for canonical, entry in self._entries.items()}
        atomic_write_json(self._file_path, payload)

    def _rebuild_reverse_index(self) -> None:
        self._alias_to_canonical = {}
        for canonical, entry in self._entries.items():
            for alias in entry.aliases:
                self._alias_to_canonical[alias] = canonical

    def resolve(self, base_name: str) -> str:
        """Return the canonical group name for *base_name*, or *base_name* itself.

        This is the hot-path call wired into the grouping layer.

        Args:
            base_name: A parser-produced base model name.

        Returns:
            The canonical group name if *base_name* is a registered alias,
            otherwise *base_name* unchanged.

        """
        with self._lock:
            return self._alias_to_canonical.get(base_name, base_name)

    def get(self, canonical: str) -> GroupAlias | None:
        """Return the alias entry for *canonical*, or ``None``."""
        with self._lock:
            entry = self._entries.get(canonical)
            return entry.model_copy(deep=True) if entry else None

    def set_aliases(self, canonical: str, aliases: list[str]) -> None:
        """Persist *aliases* for *canonical*, replacing any previous aliases.

        Args:
            canonical: The target group name.
            aliases: Base names that should resolve to *canonical*.

        Raises:
            ValueError: If *canonical* appears in *aliases* (self-reference)
                or if any alias is already claimed by a different canonical group.

        """
        with self._lock:
            if canonical in aliases:
                raise ValueError(f"Canonical name {canonical!r} cannot be its own alias")

            # Check for conflicts with other canonical entries
            for alias in aliases:
                existing_owner = self._alias_to_canonical.get(alias)
                if existing_owner is not None and existing_owner != canonical:
                    raise ValueError(f"Alias {alias!r} is already registered under canonical group {existing_owner!r}")

            self._entries[canonical] = GroupAlias(canonical=canonical, aliases=list(aliases))
            self._rebuild_reverse_index()
            self._persist()
            logger.info(f"Saved {len(aliases)} aliases for canonical group '{canonical}'")

    def add_alias(self, canonical: str, alias: str) -> None:
        """Add a single *alias* to an existing or new canonical entry.

        Args:
            canonical: The target group name.
            alias: A base name to add as an alias.

        Raises:
            ValueError: If *alias* equals *canonical* or is already registered
                under a different canonical group.

        """
        with self._lock:
            if alias == canonical:
                raise ValueError(f"Cannot alias {canonical!r} to itself")

            existing_owner = self._alias_to_canonical.get(alias)
            if existing_owner is not None and existing_owner != canonical:
                raise ValueError(f"Alias {alias!r} is already registered under canonical group {existing_owner!r}")

            entry = self._entries.get(canonical)
            if entry is None:
                entry = GroupAlias(canonical=canonical, aliases=[])
                self._entries[canonical] = entry

            if alias not in entry.aliases:
                entry.aliases.append(alias)
                self._rebuild_reverse_index()
                self._persist()
                logger.info(f"Added alias '{alias}' → canonical '{canonical}'")

    def remove_alias(self, canonical: str, alias: str) -> bool:
        """Remove a single *alias* from a canonical entry.

        Returns ``True`` if the alias was present and removed.
        """
        with self._lock:
            entry = self._entries.get(canonical)
            if entry is None or alias not in entry.aliases:
                return False

            entry.aliases.remove(alias)
            if not entry.aliases:
                del self._entries[canonical]
            self._rebuild_reverse_index()
            self._persist()
            logger.info(f"Removed alias '{alias}' from canonical '{canonical}'")
            return True

    def delete(self, canonical: str) -> bool:
        """Remove the entire alias entry for *canonical*.

        Returns ``True`` if it existed.
        """
        with self._lock:
            if canonical not in self._entries:
                return False
            del self._entries[canonical]
            self._rebuild_reverse_index()
            self._persist()
            logger.info(f"Deleted all aliases for canonical group '{canonical}'")
            return True

    def list_all(self) -> dict[str, GroupAlias]:
        """Return a deep copy of all alias entries."""
        with self._lock:
            return {k: v.model_copy(deep=True) for k, v in self._entries.items()}

    def is_alias(self, name: str) -> bool:
        """Return ``True`` if *name* is registered as an alias (not canonical)."""
        with self._lock:
            return name in self._alias_to_canonical

    def get_canonical_for(self, alias: str) -> str | None:
        """Return the canonical name that *alias* maps to, or ``None``."""
        with self._lock:
            return self._alias_to_canonical.get(alias)
