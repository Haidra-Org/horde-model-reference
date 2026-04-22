"""File-backed persistence for text model group naming schemas."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from loguru import logger

from horde_model_reference.group_aliases import GroupAliasStore
from horde_model_reference.model_reference_records import TextModelGroupNameSchema
from horde_model_reference.util import atomic_write_json


class GroupSchemaStore:
    """Thread-safe file-backed storage for per-group naming schemas.

    When an optional :class:`GroupAliasStore` is provided, :meth:`get`
    transparently resolves aliases so that schemas stored under a
    canonical group name are also returned for alias lookups.
    """

    def __init__(
        self,
        *,
        file_path: Path,
        alias_store: GroupAliasStore | None = None,
    ) -> None:
        """Create a store backed by the given JSON file.

        Args:
            file_path: Path to the JSON persistence file.
            alias_store: Optional alias store used to resolve lookups.

        """
        self._file_path = file_path
        self._lock = RLock()
        self._schemas: dict[str, TextModelGroupNameSchema] = {}
        self._alias_store = alias_store
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return
        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
            for group_name, schema_data in raw.items():
                self._schemas[group_name] = TextModelGroupNameSchema.model_validate(schema_data)
            logger.debug(f"Loaded {len(self._schemas)} group naming schemas from {self._file_path}")
        except Exception:
            logger.exception(f"Failed to load group schemas from {self._file_path}")

    def _persist(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: schema.model_dump(mode="json") for name, schema in self._schemas.items()}
        atomic_write_json(self._file_path, payload)

    def get(self, group_name: str) -> TextModelGroupNameSchema | None:
        """Return the persisted schema for *group_name*, or ``None``.

        If an alias store is configured and no direct match exists, the
        alias store is consulted to resolve *group_name* to its canonical
        form before retrying the lookup.
        """
        with self._lock:
            schema = self._schemas.get(group_name)
            if schema is None and self._alias_store is not None:
                canonical = self._alias_store.resolve(group_name)
                if canonical != group_name:
                    schema = self._schemas.get(canonical)
            return schema.model_copy() if schema else None

    def set(self, group_name: str, schema: TextModelGroupNameSchema) -> None:
        """Persist a custom naming schema for *group_name*."""
        with self._lock:
            self._schemas[group_name] = schema.model_copy()
            self._persist()
            logger.info(f"Saved custom naming schema for group '{group_name}'")

    def delete(self, group_name: str) -> bool:
        """Remove the custom schema for *group_name*. Returns ``True`` if it existed."""
        with self._lock:
            if group_name not in self._schemas:
                return False
            del self._schemas[group_name]
            self._persist()
            logger.info(f"Deleted custom naming schema for group '{group_name}'")
            return True

    def list_all(self) -> dict[str, TextModelGroupNameSchema]:
        """Return a copy of all persisted schemas."""
        with self._lock:
            return {name: schema.model_copy() for name, schema in self._schemas.items()}
