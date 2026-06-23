"""The opt-in redistributable allowlist that decides which models the sync tool may mirror.

Hosting a model on our own bucket is *redistribution*, and not every upstream weight permits it. So the tool
mirrors nothing by default: a model is eligible only when a maintainer has explicitly opted it in **by name**.
This is deliberately fail-closed and deliberately per-model: there is no category-wide opt-in, because that
would silently clear every future model added to the category (whatever its licence) without anyone reviewing
it. Opting a model in is the place the licence review is recorded, so each entry may carry the licence it was
cleared under and a free-text note for the audit trail.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RedistributableAllowlist", "RedistributableEntry"]

_DEFAULT_ALLOWLIST_PATH = Path(__file__).with_name("redistributable_allowlist.json")


@dataclass(frozen=True)
class RedistributableEntry:
    """One model a maintainer has cleared for redistribution, with the provenance of that decision."""

    name: str
    """The model-reference name cleared for redistribution."""
    license: str | None = None
    """The licence the model was reviewed and cleared under (stamped onto every uploaded object)."""
    note: str | None = None
    """Optional free-text justification/provenance for the clearance (stamped onto every uploaded object)."""

    def metadata(self) -> dict[str, str]:
        """Return the non-empty provenance fields to attach to an uploaded object."""
        fields = {"license": self.license, "redistribution_note": self.note}
        return {key: value for key, value in fields.items() if value}


@dataclass(frozen=True)
class RedistributableAllowlist:
    """Which models a maintainer has cleared for redistribution on the R2 mirror, keyed by model name.

    Empty by default, so nothing is mirrored until a model is explicitly opted in.
    """

    entries: Mapping[str, RedistributableEntry] = field(default_factory=dict)
    """Cleared models keyed by name."""

    @classmethod
    def load(cls, path: Path | None = None) -> RedistributableAllowlist:
        """Load the allowlist from *path* (defaults to the bundled JSON), tolerating an absent file as empty.

        The ``models`` array accepts either a bare model name (a string) or an object with ``name`` and the
        optional ``license`` / ``note`` provenance fields, so simple entries stay terse while reviewed ones can
        record why they are cleared.
        """
        source = path or _DEFAULT_ALLOWLIST_PATH
        if not source.is_file():
            return cls()
        data = json.loads(source.read_text(encoding="utf-8"))
        entries: dict[str, RedistributableEntry] = {}
        for raw in data.get("models", []):
            if isinstance(raw, str):
                entries[raw] = RedistributableEntry(name=raw)
                continue
            name = raw["name"]
            entries[name] = RedistributableEntry(name=name, license=raw.get("license"), note=raw.get("note"))
        return cls(entries=entries)

    def allows(self, *, model_name: str) -> bool:
        """Return whether *model_name* is cleared for redistribution."""
        return model_name in self.entries

    def metadata_for(self, model_name: str) -> dict[str, str]:
        """Return the provenance metadata to stamp onto *model_name*'s objects (empty when not cleared)."""
        entry = self.entries.get(model_name)
        return entry.metadata() if entry is not None else {}
