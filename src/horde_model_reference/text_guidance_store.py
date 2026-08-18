"""Atomic persistence and validation for reusable text-model guidance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix
from horde_model_reference.text_guidance import (
    GuidanceProfileKind,
    GuidanceRecordMetadata,
    ResolvedTextGuidance,
    TextGuidanceAssignment,
    TextGuidanceCatalog,
    TextGuidanceCatalogMetadata,
    TextGuidanceChangeSet,
    TextGuidanceStatus,
    TextGuidanceSummary,
    TextPromptContract,
    TextUsageProfile,
    TextUsageRecipe,
)
from horde_model_reference.util import atomic_write_json

_CATALOG_FILENAME = "catalog.json"


class GuidanceConflictError(ValueError):
    """Raised when a touched record changed after a proposal was reviewed."""


class TextGuidanceStore:
    """Thread-safe store for one versioned text guidance catalog."""

    def __init__(self, *, root_path: Path, bootstrap_path: Path | None = None, writable: bool = True) -> None:
        """Load a mutable or packaged guidance catalog.

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
        if writable and not (root_path / _CATALOG_FILENAME).exists():
            self._persist(self._catalog)

    @property
    def writable(self) -> bool:
        """Return whether the catalog accepts approved changes."""
        return self._writable

    def metadata(self) -> TextGuidanceCatalogMetadata:
        """Return current catalog revision metadata."""
        with self._lock:
            return self._catalog.metadata.model_copy(deep=True)

    def export(self) -> TextGuidanceCatalog:
        """Return a deep copy of the complete published catalog."""
        with self._lock:
            return self._catalog.model_copy(deep=True)

    def list_profiles(self, *, include_deprecated: bool = False) -> list[TextUsageProfile]:
        """Return profiles in stable identifier order."""
        with self._lock:
            return [
                self._catalog.profiles[profile_id].model_copy(deep=True)
                for profile_id in sorted(self._catalog.profiles)
                if include_deprecated or not self._catalog.profiles[profile_id].deprecated
            ]

    def get_profile(self, profile_id: str) -> TextUsageProfile | None:
        """Return one profile by stable identifier."""
        with self._lock:
            profile = self._catalog.profiles.get(profile_id)
            return profile.model_copy(deep=True) if profile is not None else None

    def list_assignments(self) -> list[TextGuidanceAssignment]:
        """Return explicit assignments in exact-model identifier order."""
        with self._lock:
            return [
                self._catalog.assignments[name].model_copy(deep=True) for name in sorted(self._catalog.assignments)
            ]

    def get_assignment(self, model_name: str) -> TextGuidanceAssignment | None:
        """Return the assignment for an exact model identifier."""
        with self._lock:
            assignment = self._catalog.assignments.get(model_name)
            return assignment.model_copy(deep=True) if assignment is not None else None

    def summary_for_model(self, model_name: str, *, legacy_instruct_format: str | None) -> TextGuidanceSummary:
        """Return compact published/legacy/undocumented guidance state."""
        assignment = self.get_assignment(model_name)
        if assignment is not None:
            return TextGuidanceSummary(
                status=TextGuidanceStatus.PUBLISHED,
                primary_profile_id=assignment.primary_profile_id,
                supplemental_profile_ids=assignment.supplemental_profile_ids,
            )
        if legacy_instruct_format and legacy_instruct_format.strip():
            return TextGuidanceSummary(status=TextGuidanceStatus.LEGACY_LABEL)
        return TextGuidanceSummary(status=TextGuidanceStatus.UNDOCUMENTED)

    def resolve(self, model_name: str, *, legacy_instruct_format: str | None) -> ResolvedTextGuidance:
        """Resolve a model's primary contract and ordered recipes."""
        with self._lock:
            summary = self.summary_for_model(model_name, legacy_instruct_format=legacy_instruct_format)
            assignment = self._catalog.assignments.get(model_name)
            primary: TextPromptContract | None = None
            supplemental: list[TextUsageRecipe] = []
            if assignment is not None:
                primary = self._contract(assignment.primary_profile_id)
                supplemental = [self._recipe(profile_id) for profile_id in assignment.supplemental_profile_ids]
            return ResolvedTextGuidance(
                model_name=model_name,
                summary=summary,
                primary_profile=primary,
                supplemental_profiles=supplemental,
                legacy_instruct_format=legacy_instruct_format,
                catalog_metadata=self._catalog.metadata,
            )

    def _contract(self, profile_id: str) -> TextPromptContract:
        """Return the prompt contract stored under ``profile_id``.

        Catalog validation guarantees the kind for assignments made through the store; a catalog file
        edited by hand can violate it, and that must surface as an error rather than a mistyped object.
        """
        profile = self._catalog.profiles[profile_id]
        if not isinstance(profile, TextPromptContract):
            raise ValueError(f"Guidance profile '{profile_id}' is not a prompt contract.")
        return profile

    def _recipe(self, profile_id: str) -> TextUsageRecipe:
        """Return the usage recipe stored under ``profile_id``."""
        profile = self._catalog.profiles[profile_id]
        if not isinstance(profile, TextUsageRecipe):
            raise ValueError(f"Guidance profile '{profile_id}' is not a usage recipe.")
        return profile

    def preview_change_set(
        self,
        change_set: TextGuidanceChangeSet,
        *,
        canonical_model_names: set[str],
    ) -> TextGuidanceCatalog:
        """Validate a proposal and return its prospective catalog without persisting it."""
        with self._lock:
            candidate = self._apply_to_copy(
                change_set,
                canonical_model_names=canonical_model_names,
                editor_id=None,
                enforce_preconditions=True,
            )
            return candidate.model_copy(deep=True)

    def apply_change_set(
        self,
        change_set: TextGuidanceChangeSet,
        *,
        canonical_model_names: set[str],
        editor_id: str,
    ) -> TextGuidanceCatalog:
        """Atomically validate and persist one approved guidance change set."""
        if not self._writable:
            raise PermissionError("Text guidance catalog is read-only.")
        with self._lock:
            candidate = self._apply_to_copy(
                change_set,
                canonical_model_names=canonical_model_names,
                editor_id=editor_id,
                enforce_preconditions=True,
            )
            timestamp = _now_timestamp()
            candidate.metadata = TextGuidanceCatalogMetadata(
                schema_version=candidate.metadata.schema_version,
                revision=self._catalog.metadata.revision + 1,
                updated_at=timestamp,
            )
            self._persist(candidate)
            self._catalog = candidate
            return candidate.model_copy(deep=True)

    def refresh_replica_export(self, export_payload: dict[str, Any]) -> None:
        """Replace the read-only replica cache from a validated PRIMARY export."""
        candidate = TextGuidanceCatalog.model_validate(export_payload)
        with self._lock:
            self._root_path.mkdir(parents=True, exist_ok=True)
            self._persist(candidate)
            self._catalog = candidate

    def _apply_to_copy(
        self,
        change_set: TextGuidanceChangeSet,
        *,
        canonical_model_names: set[str],
        editor_id: str | None,
        enforce_preconditions: bool,
    ) -> TextGuidanceCatalog:
        candidate = self._catalog.model_copy(deep=True)
        timestamp = _now_timestamp()

        for change in change_set.profile_changes:
            current = candidate.profiles.get(change.profile_id)
            if enforce_preconditions and current != change.expected_before:
                raise GuidanceConflictError(f"Guidance profile '{change.profile_id}' changed after review.")
            if change.operation == "create":
                if current is not None or change.profile is None:
                    raise ValueError(f"Cannot create guidance profile '{change.profile_id}'.")
                candidate.profiles[change.profile_id] = self._with_metadata(
                    change.profile,
                    previous=None,
                    editor_id=editor_id,
                    timestamp=timestamp,
                )
            elif change.operation == "update":
                if current is None or change.profile is None:
                    raise ValueError(f"Cannot update missing guidance profile '{change.profile_id}'.")
                candidate.profiles[change.profile_id] = self._with_metadata(
                    change.profile,
                    previous=current,
                    editor_id=editor_id,
                    timestamp=timestamp,
                )
            else:
                if current is None:
                    raise ValueError(f"Cannot deprecate missing guidance profile '{change.profile_id}'.")
                candidate.profiles[change.profile_id] = self._with_metadata(
                    current.model_copy(update={"deprecated": True}),
                    previous=current,
                    editor_id=editor_id,
                    timestamp=timestamp,
                )

        for change in change_set.assignment_changes:
            current = candidate.assignments.get(change.model_name)
            if enforce_preconditions and current != change.expected_before:
                raise GuidanceConflictError(f"Guidance assignment for '{change.model_name}' changed after review.")
            if change.assignment is None:
                candidate.assignments.pop(change.model_name, None)
            else:
                candidate.assignments[change.model_name] = self._assignment_with_metadata(
                    change.assignment,
                    previous=current,
                    editor_id=editor_id,
                    timestamp=timestamp,
                )

        self._validate_catalog(candidate, canonical_model_names=canonical_model_names)
        return candidate

    @staticmethod
    def _with_metadata(
        profile: TextUsageProfile,
        *,
        previous: TextUsageProfile | None,
        editor_id: str | None,
        timestamp: int,
    ) -> TextUsageProfile:
        old_metadata = previous.metadata if previous is not None else GuidanceRecordMetadata()
        metadata = GuidanceRecordMetadata(
            revision=old_metadata.revision + (1 if previous is not None else 0),
            created_at=old_metadata.created_at or timestamp,
            created_by=old_metadata.created_by or editor_id,
            updated_at=timestamp,
            updated_by=editor_id,
        )
        return profile.model_copy(update={"metadata": metadata})

    @staticmethod
    def _assignment_with_metadata(
        assignment: TextGuidanceAssignment,
        *,
        previous: TextGuidanceAssignment | None,
        editor_id: str | None,
        timestamp: int,
    ) -> TextGuidanceAssignment:
        old_metadata = previous.metadata if previous is not None else GuidanceRecordMetadata()
        metadata = GuidanceRecordMetadata(
            revision=old_metadata.revision + (1 if previous is not None else 0),
            created_at=old_metadata.created_at or timestamp,
            created_by=old_metadata.created_by or editor_id,
            updated_at=timestamp,
            updated_by=editor_id,
        )
        return assignment.model_copy(update={"metadata": metadata})

    @staticmethod
    def _validate_catalog(catalog: TextGuidanceCatalog, *, canonical_model_names: set[str]) -> None:
        alias_owner: dict[str, str] = {}
        for profile_id, profile in catalog.profiles.items():
            if profile.profile_id != profile_id:
                raise ValueError(f"Profile key '{profile_id}' does not match its profile_id.")
            for alias in profile.aliases:
                normalized = alias.casefold()
                existing = alias_owner.get(normalized)
                if existing is not None and existing != profile_id:
                    raise ValueError(f"Guidance alias '{alias}' is shared by multiple profiles.")
                alias_owner[normalized] = profile_id

        for model_name, assignment in catalog.assignments.items():
            if assignment.model_name != model_name:
                raise ValueError(f"Assignment key '{model_name}' does not match its model_name.")
            if has_legacy_text_backend_prefix(model_name):
                raise ValueError(f"Guidance must target canonical text models, not '{model_name}'.")
            if model_name not in canonical_model_names:
                raise ValueError(f"Unknown canonical text model: {model_name}")
            primary = catalog.profiles.get(assignment.primary_profile_id)
            if primary is None or primary.kind is not GuidanceProfileKind.PROMPT_CONTRACT or primary.deprecated:
                raise ValueError(f"Assignment for '{model_name}' requires an active prompt contract.")
            for profile_id in assignment.supplemental_profile_ids:
                recipe = catalog.profiles.get(profile_id)
                if recipe is None or recipe.kind is not GuidanceProfileKind.USAGE_RECIPE or recipe.deprecated:
                    raise ValueError(f"Supplemental profile '{profile_id}' must be an active usage recipe.")

        assigned_profile_ids = {
            profile_id
            for assignment in catalog.assignments.values()
            for profile_id in [assignment.primary_profile_id, *assignment.supplemental_profile_ids]
        }
        deprecated_assigned = sorted(
            profile_id
            for profile_id in assigned_profile_ids
            if catalog.profiles.get(profile_id) is not None and catalog.profiles[profile_id].deprecated
        )
        if deprecated_assigned:
            raise ValueError(f"Deprecated profiles remain assigned: {', '.join(deprecated_assigned)}")

    def _load(self) -> TextGuidanceCatalog:
        mutable_path = self._root_path / _CATALOG_FILENAME
        source_path = mutable_path if mutable_path.is_file() else None
        if source_path is None and self._bootstrap_path is not None:
            bootstrap_path = self._bootstrap_path / _CATALOG_FILENAME
            source_path = bootstrap_path if bootstrap_path.is_file() else None
        if source_path is None:
            return TextGuidanceCatalog()
        return TextGuidanceCatalog.model_validate_json(source_path.read_text(encoding="utf-8"))

    def _persist(self, catalog: TextGuidanceCatalog) -> None:
        self._root_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self._root_path / _CATALOG_FILENAME, catalog.model_dump(mode="json"))


def _now_timestamp() -> int:
    """Return the current UTC Unix timestamp."""
    return int(datetime.now(tz=UTC).timestamp())
