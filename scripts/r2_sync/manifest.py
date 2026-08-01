"""Build the generated, content-addressed inventory consumed by mirror-aware clients."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from scripts.r2_sync.planner import SyncAction, SyncPlan

__all__ = [
    "MIRROR_MANIFEST_KEY",
    "MirrorManifest",
    "MirrorManifestObject",
    "MirrorManifestReference",
    "build_mirror_manifest",
    "write_mirror_manifest",
]

MIRROR_MANIFEST_KEY = "manifests/current.json"
"""Stable private-bucket key from which the gateway serves the current inventory."""


class MirrorManifestReference(BaseModel):
    """Represent one model-reference declaration that points at a mirrored object."""

    model_config = ConfigDict(frozen=True)

    category: str
    model_name: str
    file_name: str


class MirrorManifestObject(BaseModel):
    """Represent one verified R2 object and its redistribution information."""

    model_config = ConfigDict(frozen=True)

    size_bytes: int
    license_expressions: list[str] = Field(default_factory=list)
    attributions: list[str] = Field(default_factory=list)
    references: list[MirrorManifestReference] = Field(default_factory=list)


class MirrorManifest(BaseModel):
    """Represent the atomically published set of hashes clients may request from the gateway."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: datetime
    objects: dict[str, MirrorManifestObject]


@dataclass
class _ManifestAggregate:
    """Accumulate all references and notices for one deduplicated content hash."""

    size_bytes: int
    license_expressions: set[str] = field(default_factory=set)
    attributions: set[str] = field(default_factory=set)
    references: list[MirrorManifestReference] = field(default_factory=list)


def build_mirror_manifest(plan: SyncPlan, *, generated_at: datetime | None = None) -> MirrorManifest:
    """Create a deduplicated mirror inventory from successful plan items.

    Args:
        plan: Completed apply plan whose uploaded/present objects are known to exist.
        generated_at: Optional deterministic timestamp for tests.

    Returns:
        A manifest keyed by lowercase whole-file SHA-256.

    Raises:
        ValueError: If a successful item lacks the hash or size required for publication.
    """
    grouped: dict[str, _ManifestAggregate] = {}
    successful_actions = {SyncAction.UPLOAD, SyncAction.ALREADY_PRESENT}
    for item in plan.items:
        if item.action not in successful_actions:
            continue
        if item.sha256 is None or item.size_bytes is None:
            identity = f"{item.category}/{item.model_name}/{item.file_name}"
            raise ValueError(f"Successful mirror item lacks hash/size: {identity}")
        sha256 = item.sha256.lower()
        aggregate = grouped.setdefault(sha256, _ManifestAggregate(size_bytes=item.size_bytes))
        if aggregate.size_bytes != item.size_bytes:
            raise ValueError(f"Conflicting sizes for content hash {sha256}")
        if item.license_expression:
            aggregate.license_expressions.add(item.license_expression)
        if item.attribution:
            aggregate.attributions.add(item.attribution)
        aggregate.references.append(
            MirrorManifestReference(
                category=item.category,
                model_name=item.model_name,
                file_name=item.file_name,
            ),
        )

    objects = {
        sha256: MirrorManifestObject(
            size_bytes=aggregate.size_bytes,
            license_expressions=sorted(aggregate.license_expressions),
            attributions=sorted(aggregate.attributions),
            references=list(aggregate.references),
        )
        for sha256, aggregate in sorted(grouped.items())
    }
    timestamp = generated_at or datetime.now(UTC)
    return MirrorManifest(generated_at=timestamp, objects=objects)


def write_mirror_manifest(manifest: MirrorManifest, path: Path) -> None:
    """Write *manifest* as deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
