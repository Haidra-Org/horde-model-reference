"""Atomic persistence and validation for the served image baseline catalog."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from horde_model_reference.image_baseline import (
    BaselineRecordMetadata,
    ImageBaselineCatalog,
    ImageBaselineCatalogMetadata,
    ImageBaselineChangeSet,
    ImageBaselineRecord,
)
from horde_model_reference.model_consts.image import register_image_baselines_from_catalog
from horde_model_reference.util import atomic_write_json

_CATALOG_FILENAME = "catalog.json"


class BaselineConflictError(ValueError):
    """Raised when a touched baseline changed after a proposal was reviewed."""


class ImageBaselineStore:
    """Thread-safe store for one versioned image baseline catalog.

    Every successful load, apply, and replica refresh hydrates the in-process baseline registry, so
    that name recognition follows the served catalog rather than the packaged vocabulary alone.
    """

    def __init__(self, *, root_path: Path, bootstrap_path: Path | None = None, writable: bool = True) -> None:
        """Load a mutable or packaged baseline catalog.

        Args:
            root_path: Runtime directory containing ``catalog.json``.
            bootstrap_path: Optional packaged catalog used when runtime data is absent.
            writable: Whether change sets may be applied.
        """
        self._root_path = root_path
        self._bootstrap_path = bootstrap_path
        self._writable = writable
        self._lock = RLock()
        self._catalog = self._load()
        register_image_baselines_from_catalog(self._catalog)
        if writable and not (root_path / _CATALOG_FILENAME).exists():
            self._persist(self._catalog)

    @property
    def writable(self) -> bool:
        """Return whether the catalog accepts approved changes."""
        return self._writable

    def metadata(self) -> ImageBaselineCatalogMetadata:
        """Return current catalog revision metadata."""
        with self._lock:
            return self._catalog.metadata.model_copy(deep=True)

    def export(self) -> ImageBaselineCatalog:
        """Return a deep copy of the complete published catalog."""
        with self._lock:
            return self._catalog.model_copy(deep=True)

    def list(self) -> list[ImageBaselineRecord]:
        """Return baselines in stable name order."""
        with self._lock:
            return [self._catalog.baselines[name].model_copy(deep=True) for name in sorted(self._catalog.baselines)]

    def get(self, name: str) -> ImageBaselineRecord | None:
        """Return one baseline by exact name."""
        with self._lock:
            record = self._catalog.baselines.get(name)
            return record.model_copy(deep=True) if record is not None else None

    def preview_change_set(
        self,
        change_set: ImageBaselineChangeSet,
        *,
        referenced_baselines: set[str],
    ) -> ImageBaselineCatalog:
        """Validate a proposal and return its prospective catalog without persisting it."""
        with self._lock:
            candidate = self._apply_to_copy(
                change_set,
                referenced_baselines=referenced_baselines,
                editor_id=None,
            )
            return candidate.model_copy(deep=True)

    def apply_change_set(
        self,
        change_set: ImageBaselineChangeSet,
        *,
        referenced_baselines: set[str],
        editor_id: str,
    ) -> ImageBaselineCatalog:
        """Atomically validate and persist one approved baseline change set."""
        if not self._writable:
            raise PermissionError("Image baseline catalog is read-only.")
        with self._lock:
            # The catalog file and pending-queue database cannot share a transaction. If persistence
            # succeeds but marking the queue item applied fails, the queue deliberately releases its
            # reservation for retry. Treat an already-effective change set as a successful no-op so
            # that retry can finish the queue transition instead of failing its stale precondition.
            if self._change_set_is_effective(change_set):
                return self._catalog.model_copy(deep=True)
            candidate = self._apply_to_copy(
                change_set,
                referenced_baselines=referenced_baselines,
                editor_id=editor_id,
            )
            candidate.metadata = ImageBaselineCatalogMetadata(
                schema_version=candidate.metadata.schema_version,
                revision=self._catalog.metadata.revision + 1,
                updated_at=_now_timestamp(),
            )
            self._persist(candidate)
            self._catalog = candidate
            register_image_baselines_from_catalog(candidate)
            return candidate.model_copy(deep=True)

    def _change_set_is_effective(self, change_set: ImageBaselineChangeSet) -> bool:
        """Return whether every requested end state is already present in the current catalog."""
        for change in change_set.changes:
            current = self._catalog.baselines.get(change.name)
            if change.operation == "delete":
                if current is not None:
                    return False
                continue
            if change.record is None or current is None:
                return False
            # Audit metadata is assigned at apply time and is intentionally absent/stale in the
            # reviewed payload; compare the requested semantic fields using the current metadata.
            requested = change.record.model_copy(update={"metadata": current.metadata})
            if requested != current:
                return False
        return True

    def refresh_replica_export(self, export_payload: dict[str, Any]) -> None:
        """Replace the read-only replica cache from a validated PRIMARY export."""
        candidate = ImageBaselineCatalog.model_validate(export_payload)
        self._validate_catalog(candidate)
        with self._lock:
            self._root_path.mkdir(parents=True, exist_ok=True)
            self._persist(candidate)
            self._catalog = candidate
            register_image_baselines_from_catalog(candidate)

    def _apply_to_copy(
        self,
        change_set: ImageBaselineChangeSet,
        *,
        referenced_baselines: set[str],
        editor_id: str | None,
    ) -> ImageBaselineCatalog:
        candidate = self._catalog.model_copy(deep=True)
        timestamp = _now_timestamp()

        for change in change_set.changes:
            current = candidate.baselines.get(change.name)
            if current != change.expected_before:
                raise BaselineConflictError(f"Baseline '{change.name}' changed after review.")
            if change.operation == "delete":
                if current is None:
                    raise ValueError(f"Cannot delete missing baseline '{change.name}'.")
                if change.name in referenced_baselines:
                    raise ValueError(f"Baseline '{change.name}' is still named by a canonical image model.")
                del candidate.baselines[change.name]
                continue
            if change.record is None or change.record.name != change.name:
                raise ValueError(f"Cannot upsert baseline '{change.name}' without a matching record.")
            candidate.baselines[change.name] = self._with_metadata(
                change.record,
                previous=current,
                editor_id=editor_id,
                timestamp=timestamp,
            )

        self._validate_catalog(candidate)
        return candidate

    @staticmethod
    def _with_metadata(
        record: ImageBaselineRecord,
        *,
        previous: ImageBaselineRecord | None,
        editor_id: str | None,
        timestamp: int,
    ) -> ImageBaselineRecord:
        old_metadata = previous.metadata if previous is not None else BaselineRecordMetadata()
        metadata = BaselineRecordMetadata(
            revision=old_metadata.revision + (1 if previous is not None else 0),
            created_at=old_metadata.created_at or timestamp,
            created_by=old_metadata.created_by or editor_id,
            updated_at=timestamp,
            updated_by=editor_id,
        )
        return record.model_copy(update={"metadata": metadata})

    @staticmethod
    def _validate_catalog(catalog: ImageBaselineCatalog) -> None:
        alternative_owner: dict[str, str] = {}
        for name, record in catalog.baselines.items():
            if record.name != name:
                raise ValueError(f"Baseline key '{name}' does not match its name.")
            for alternative_name in record.alternative_names:
                if alternative_name != name and alternative_name in catalog.baselines:
                    raise ValueError(f"Baseline alternative name '{alternative_name}' names another baseline.")
                owner = alternative_owner.get(alternative_name)
                if owner is not None and owner != name:
                    raise ValueError(f"Baseline alternative name '{alternative_name}' is shared by two baselines.")
                alternative_owner[alternative_name] = name

    def _load(self) -> ImageBaselineCatalog:
        mutable_path = self._root_path / _CATALOG_FILENAME
        source_path = mutable_path if mutable_path.is_file() else None
        if source_path is None and self._bootstrap_path is not None:
            bootstrap_path = self._bootstrap_path / _CATALOG_FILENAME
            source_path = bootstrap_path if bootstrap_path.is_file() else None
        if source_path is None:
            return ImageBaselineCatalog()
        return ImageBaselineCatalog.model_validate_json(source_path.read_text(encoding="utf-8"))

    def _persist(self, catalog: ImageBaselineCatalog) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._root_path / _CATALOG_FILENAME, catalog.model_dump(mode="json"))


def _now_timestamp() -> int:
    """Return the current UTC Unix timestamp."""
    return int(datetime.now(tz=UTC).timestamp())
