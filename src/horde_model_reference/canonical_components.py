"""The canonical set of shareable model components, and how it is derived from the reference.

A component (a VAE or a text encoder) is worth sharing across processes only when several models use the
byte-identical weights. This module holds the derived answer to "which component content-hashes are worth
materialising once and sharing?" plus, for each, where a worker can obtain the standalone file (which model
carries it, and whether as its own file or embedded in a checkpoint it must slice).

The set is derived centrally by the offline hashing pass and served as data, so every worker consumes one
authoritative answer rather than each re-deriving it (and drifting). Derivation is by frequency with a
curated override: a component shared by at least ``min_shared_models`` distinct models is promoted, an
allowlist can force a specific hash in regardless of count, and a denylist can exclude one. The identity is
the torch-free tensor-region hash from :mod:`horde_model_reference.component_hash`; the kind values match
``on_disk_layout.COMPONENT_PURPOSE_FOLDERS`` so placement of a materialised file reuses that routing.

This module stays torch-free: it is pure reference data and grouping logic over already-computed hashes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from horde_model_reference.component_hash import ComponentKind, component_kind_for_purpose
from horde_model_reference.model_reference_records import get_default_config

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping

    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = [
    "DEFAULT_MIN_SHARED_MODELS",
    "CanonicalComponent",
    "CanonicalComponentRegistry",
    "CanonicalComponentSource",
    "derive_canonical_registry",
]

DEFAULT_MIN_SHARED_MODELS = 2
"""A component used by this many distinct models is promoted: at two, sharing already saves a whole copy."""


class CanonicalComponentSource(BaseModel):
    """One model that carries a canonical component, and how to obtain the component's standalone bytes."""

    model_config = get_default_config()

    model_name: str
    """The reference model that carries this component."""
    kind: ComponentKind
    """Which component this source provides."""
    embedded: bool
    """True when the component must be sliced out of the model's primary checkpoint; False when the model
    already declares it as its own standalone component file."""
    file_name: str | None = None
    """The standalone component file's name (from its ``DownloadRecord``); ``None`` when ``embedded``."""
    file_purpose: str | None = None
    """The standalone component file's ``file_purpose``; ``None`` when ``embedded``."""


class CanonicalComponent(BaseModel):
    """A shareable component promoted to the canonical set, identified by its content hash."""

    model_config = get_default_config()

    content_hash: str
    """The torch-free tensor-region content hash identifying these weights."""
    kind: ComponentKind
    """The component kind (``vae`` or ``text_encoders``)."""
    shared_by_model_count: int
    """How many distinct models carry this exact component (the promotion signal, kept for observability)."""
    sources: list[CanonicalComponentSource] = Field(default_factory=list)
    """Every known way to obtain the component; a worker uses whichever source model it has on disk."""


class CanonicalComponentRegistry(BaseModel):
    """The full derived set of canonical components, with lookup by hash and by kind."""

    model_config = get_default_config()

    components: list[CanonicalComponent] = Field(default_factory=list)

    def by_hash(self, content_hash: str) -> CanonicalComponent | None:
        """Return the canonical component with *content_hash*, or None when it is not canonical."""
        return next((component for component in self.components if component.content_hash == content_hash), None)

    def for_kind(self, kind: ComponentKind) -> list[CanonicalComponent]:
        """Return the canonical components of a given *kind*."""
        return [component for component in self.components if component.kind == kind]


def _iter_component_candidates(
    model_name: str,
    record: GenericModelRecord,
) -> Iterator[tuple[str, ComponentKind, CanonicalComponentSource]]:
    """Yield every ``(content_hash, kind, source)`` a single record contributes.

    Split-file components come from each ``DownloadRecord`` that declares a component ``file_purpose`` and a
    ``content_hash``; embedded components come from the record's ``embedded_component_hashes`` map.
    """
    config = record.config
    for download in config.download:
        kind = component_kind_for_purpose(download.file_purpose)
        if kind is None or download.content_hash is None:
            continue
        yield (
            download.content_hash,
            kind,
            CanonicalComponentSource(
                model_name=model_name,
                kind=kind,
                embedded=False,
                file_name=download.file_name,
                file_purpose=download.file_purpose,
            ),
        )
    for purpose, content_hash in (config.embedded_component_hashes or {}).items():
        kind = component_kind_for_purpose(purpose)
        if kind is None:
            continue
        yield (
            content_hash,
            kind,
            CanonicalComponentSource(model_name=model_name, kind=kind, embedded=True),
        )


def derive_canonical_registry(
    records: Mapping[str, GenericModelRecord],
    *,
    min_shared_models: int = DEFAULT_MIN_SHARED_MODELS,
    allow: Collection[str] = (),
    deny: Collection[str] = (),
) -> CanonicalComponentRegistry:
    """Derive the canonical component set from *records* by frequency, with a curated override.

    A ``(content_hash, kind)`` group is promoted when its hash is in *allow*, or when its hash is not in
    *deny* and at least *min_shared_models* distinct models carry it. Sources are collected per group and
    deduplicated, so a worker can pick whichever carrier it already has on disk.

    Args:
        records: The reference, keyed by model name.
        min_shared_models: The frequency threshold for automatic promotion.
        allow: Content hashes to force into the set regardless of count.
        deny: Content hashes to exclude even when frequent.
    """
    allow_set = set(allow)
    deny_set = set(deny)

    models_by_group: dict[tuple[str, ComponentKind], set[str]] = defaultdict(set)
    sources_by_group: dict[tuple[str, ComponentKind], dict[tuple[str, bool, str | None], CanonicalComponentSource]] = (
        defaultdict(dict)
    )
    for model_name, record in records.items():
        for content_hash, kind, source in _iter_component_candidates(model_name, record):
            group = (content_hash, kind)
            models_by_group[group].add(model_name)
            sources_by_group[group][(source.model_name, source.embedded, source.file_name)] = source

    components: list[CanonicalComponent] = []
    for (content_hash, kind), model_names in models_by_group.items():
        if content_hash in deny_set:
            continue
        if content_hash not in allow_set and len(model_names) < min_shared_models:
            continue
        components.append(
            CanonicalComponent(
                content_hash=content_hash,
                kind=kind,
                shared_by_model_count=len(model_names),
                sources=list(sources_by_group[(content_hash, kind)].values()),
            ),
        )
    components.sort(key=lambda component: (component.kind, component.content_hash))
    return CanonicalComponentRegistry(components=components)
