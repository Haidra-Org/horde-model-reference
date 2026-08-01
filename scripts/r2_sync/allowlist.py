"""Load and validate the fail-closed redistribution policy used by the R2 mirror.

The legacy allowlist keyed only by model name is intentionally not sufficient for uploads: model names are not
globally unique, a model can contain files with different provenance, and a bare name does not record why
redistribution is permitted.  The active policy is therefore category-aware, supports optional per-file scope,
and requires review evidence for every approved entry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from strenum import StrEnum

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

__all__ = [
    "RedistributableAllowlist",
    "RedistributableEntry",
    "RedistributionDecision",
    "RedistributionPolicy",
]

_DEFAULT_POLICY_PATH = Path(__file__).with_name("redistribution_policy.json")


class RedistributionDecision(StrEnum):
    """Whether an artifact is approved for mirroring or deliberately blocked."""

    APPROVED = "approved"
    BLOCKED = "blocked"


class RedistributableEntry(BaseModel):
    """Represent one category/model approval, optionally narrowed to named files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: MODEL_REFERENCE_CATEGORY
    """Model-reference category containing the reviewed model."""
    name: str
    """Exact model-reference name reviewed for redistribution."""
    decision: RedistributionDecision
    """Explicit review decision; only ``approved`` entries can be mirrored."""
    files: tuple[str, ...] | None = None
    """Optional exact file-name scope; ``None`` means every declared file in this model was reviewed."""
    license_expression: str | None = None
    """Deprecated migration assertion; current model licensing is authoritative."""
    evidence: tuple[str, ...] = Field(default_factory=tuple)
    """URLs or repository-relative audit references supporting the decision."""
    reviewed_by: str | None = None
    """Maintainer or audit identity responsible for the decision."""
    reviewed_at: date | None = None
    """Date on which the redistribution terms were reviewed."""
    note: str | None = None
    """Free-text caveats, attribution requirements, or reason for blocking."""
    attribution: str | None = None
    """Deprecated migration attribution; current model licensing is authoritative."""
    licensing_fingerprint: str | None = None
    """Optional reviewed fingerprint of the effective model/file assignment."""

    @model_validator(mode="after")
    def validate_approval_evidence(self) -> RedistributableEntry:
        """Validate that an approval is auditable and cannot be created as a bare name."""
        if self.decision is RedistributionDecision.BLOCKED:
            if not self.note:
                raise ValueError("blocked redistribution entries must explain the reason in note")
            return self
        missing: list[str] = []
        if not self.reviewed_by:
            missing.append("reviewed_by")
        if self.reviewed_at is None:
            missing.append("reviewed_at")
        if missing:
            raise ValueError(f"approved redistribution entries require: {', '.join(missing)}")
        return self

    def covers(self, *, file_name: str) -> bool:
        """Return whether this decision covers *file_name*."""
        return self.files is None or file_name in self.files

    def object_metadata(self) -> dict[str, str]:
        """Return compact audit metadata suitable for R2 custom object metadata."""
        metadata = {
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }
        return {key: value for key, value in metadata.items() if value}


@dataclass(frozen=True)
class RedistributionPolicy:
    """Hold explicit redistribution decisions keyed by ``(category, model name)``."""

    entries: Mapping[tuple[MODEL_REFERENCE_CATEGORY, str], RedistributableEntry] = field(default_factory=dict)
    """Reviewed entries keyed by category and model name."""
    source_path: Path | None = None
    """Policy file from which these decisions were loaded, when applicable."""

    @classmethod
    def load(cls, path: Path | None = None) -> RedistributionPolicy:
        """Load and strictly validate the redistribution policy from JSON.

        Raises:
            FileNotFoundError: If the selected policy does not exist.
            ValueError: If the policy contains duplicate category/model identities.
            pydantic.ValidationError: If an entry is malformed or an approval lacks evidence.
        """
        source = path or _DEFAULT_POLICY_PATH
        if not source.is_file():
            raise FileNotFoundError(f"Redistribution policy not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        raw_entries = payload.get("models")
        if not isinstance(raw_entries, list):
            raise ValueError("Redistribution policy must contain a 'models' array")

        entries: dict[tuple[MODEL_REFERENCE_CATEGORY, str], RedistributableEntry] = {}
        for raw_entry in raw_entries:
            entry = RedistributableEntry.model_validate(raw_entry)
            identity = (entry.category, entry.name)
            if identity in entries:
                raise ValueError(f"Duplicate redistribution policy entry: {entry.category}/{entry.name}")
            entries[identity] = entry
        return cls(entries=entries, source_path=source)

    def decision_for(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY | str,
        model_name: str,
        file_name: str,
    ) -> RedistributableEntry | None:
        """Return the decision covering a declared file, or ``None`` when it is unreviewed."""
        normalized_category = MODEL_REFERENCE_CATEGORY(category)
        entry = self.entries.get((normalized_category, model_name))
        return entry if entry is not None and entry.covers(file_name=file_name) else None

    def allows(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY | str,
        model_name: str,
        file_name: str,
    ) -> bool:
        """Return whether a category/model/file identity is explicitly approved."""
        entry = self.decision_for(category=category, model_name=model_name, file_name=file_name)
        return entry is not None and entry.decision is RedistributionDecision.APPROVED

    def metadata_for(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY | str,
        model_name: str,
        file_name: str,
    ) -> dict[str, str]:
        """Return audit metadata for an approved artifact, otherwise an empty mapping."""
        entry = self.decision_for(category=category, model_name=model_name, file_name=file_name)
        if entry is None or entry.decision is not RedistributionDecision.APPROVED:
            return {}
        return entry.object_metadata()


# Compatibility import for callers that used the original class name.  Its semantics are now strict and
# category-aware; the alias does not preserve unsafe name-only matching.
RedistributableAllowlist = RedistributionPolicy
