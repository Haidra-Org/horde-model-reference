"""Decide, per declared file, what the R2 sync run should do: skip, upload, or note it is already mirrored.

This is the pure heart of the tool, free of argparse, boto3 and the model-reference manager so it can be tested
against an in-memory store and a hand-built byte source. It enforces the four shaping decisions: only
*hostable* (non-generation) categories, only *allowlisted* models, *content-addressed* keys, and *backfilled*
hashes for records that still carry the ``"FIXME"`` sentinel.

Bytes are acquired lazily: a record that already declares a real sha256 whose object is present is resolved with
a single ``head`` and no download at all; bytes are fetched only to compute a missing hash or to perform an
actual upload.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from strenum import StrEnum

from horde_model_reference.download_engine import UNKNOWN_SHA256_SENTINEL, sha256_of
from horde_model_reference.meta_consts import (
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    get_category_descriptor,
)
from scripts.r2_sync.object_store import object_key_for

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from horde_model_reference.model_reference_records import DownloadRecord, GenericModelRecord
    from scripts.r2_sync.allowlist import RedistributableAllowlist
    from scripts.r2_sync.object_store import ObjectStore

__all__ = [
    "ByteSource",
    "HashCorrection",
    "SyncAction",
    "SyncItem",
    "SyncPlan",
    "build_sync_plan",
    "hostable_categories",
]


class SyncAction(StrEnum):
    """What the run decided for one declared file."""

    UPLOAD = "upload"
    """The object is absent from the bucket and (in apply mode) was uploaded."""
    ALREADY_PRESENT = "already_present"
    """The content-addressed object already exists in the bucket; nothing to upload."""
    SKIPPED_NOT_ALLOWLISTED = "skipped_not_allowlisted"
    """The model is not opted in for redistribution, so it is left on its origin host."""
    MISSING_BYTES = "missing_bytes"
    """Neither a local copy nor an origin download could provide the bytes, so it could not be processed."""
    HASH_MISMATCH = "hash_mismatch"
    """The acquired bytes did not match the record's declared sha256; not uploaded (needs investigation)."""


class ByteSource(Protocol):
    """Provides the local bytes for a declared file, or None when they cannot be obtained."""

    def acquire(self, record: GenericModelRecord, download: DownloadRecord) -> Path | None:
        """Return a local path holding *download*'s bytes (local copy or freshly fetched), or None."""
        ...


@dataclass(frozen=True)
class HashCorrection:
    """A ``FIXME`` (or otherwise absent) sha256 the run computed and that should be backfilled into the record."""

    category: str
    model_name: str
    file_name: str
    old_sha256: str
    new_sha256: str


@dataclass(frozen=True)
class SyncItem:
    """The outcome for one declared file of one model."""

    category: str
    model_name: str
    file_name: str
    action: SyncAction
    sha256: str | None = None
    key: str | None = None
    detail: str | None = None


@dataclass
class SyncPlan:
    """The full result of a sync run: per-file outcomes plus the hash corrections to backfill."""

    items: list[SyncItem] = field(default_factory=list)
    corrections: list[HashCorrection] = field(default_factory=list)

    def counts(self) -> Counter[SyncAction]:
        """Return how many files fell into each :class:`SyncAction`."""
        return Counter(item.action for item in self.items)


def hostable_categories() -> list[MODEL_REFERENCE_CATEGORY]:
    """Return the non-generation categories whose files are small enough (and ours) to mirror on R2.

    Derived from the category registry rather than hard-coded: a category qualifies when it has an on-disk
    weights folder, is not managed by an external system, is not downloaded by a specialised external mechanism
    (LoRA/TI via CivitAI), and is not a generation category (those checkpoints are the expensive ones we do not
    host). Today this yields controlnet, clip, blip, esrgan, gfpgan, codeformer, safety_checker and
    miscellaneous.
    """
    hostable: list[MODEL_REFERENCE_CATEGORY] = []
    for category in MODEL_REFERENCE_CATEGORY:
        descriptor = get_category_descriptor(category)
        if descriptor.on_disk_folder_name is None:
            continue
        if descriptor.managed_elsewhere or descriptor.managed_download_elsewhere:
            continue
        if descriptor.purpose == MODEL_PURPOSE.generation:
            continue
        hostable.append(category)
    return hostable


def _object_metadata(
    category: str,
    record: GenericModelRecord,
    download: DownloadRecord,
    allowlist: RedistributableAllowlist,
) -> dict[str, str]:
    """Build the provenance metadata stored alongside an uploaded object.

    Includes where the object came from (category/model/file/source URL) and the *redistribution* provenance the
    maintainer recorded when clearing the model (its licence and any note), so the bucket carries the audit trail
    for why each hosted file is allowed to be there.
    """
    metadata = {
        "category": category,
        "model_name": record.name,
        "file_name": download.file_name,
        "source_url": download.file_url,
    }
    metadata.update(allowlist.metadata_for(record.name))
    return metadata


def _plan_file(
    category: str,
    record: GenericModelRecord,
    download: DownloadRecord,
    *,
    allowlist: RedistributableAllowlist,
    store: ObjectStore,
    byte_source: ByteSource,
    apply: bool,
) -> tuple[SyncItem, HashCorrection | None]:
    """Decide and (in *apply* mode) perform the action for one declared file."""
    base = {"category": category, "model_name": record.name, "file_name": download.file_name}
    declared = download.sha256sum
    known_sha = declared if declared and declared != UNKNOWN_SHA256_SENTINEL else None

    # A record that already declares a real hash and whose object is present needs no bytes at all.
    if known_sha is not None:
        key = object_key_for(known_sha)
        if store.head(key):
            return SyncItem(**base, action=SyncAction.ALREADY_PRESENT, sha256=known_sha, key=key), None

    path = byte_source.acquire(record, download)
    if path is None:
        return SyncItem(**base, action=SyncAction.MISSING_BYTES, sha256=known_sha), None

    actual_sha = sha256_of(path)
    correction: HashCorrection | None = None
    if known_sha is None:
        correction = HashCorrection(
            category=category,
            model_name=record.name,
            file_name=download.file_name,
            old_sha256=declared,
            new_sha256=actual_sha,
        )
    elif actual_sha.lower() != known_sha.lower():
        detail = f"declared {known_sha} but bytes hash to {actual_sha}"
        return SyncItem(**base, action=SyncAction.HASH_MISMATCH, sha256=known_sha, detail=detail), None

    key = object_key_for(actual_sha)
    if store.head(key):
        return SyncItem(**base, action=SyncAction.ALREADY_PRESENT, sha256=actual_sha, key=key), correction

    if apply:
        store.put(key, path, metadata=_object_metadata(category, record, download, allowlist))
    return SyncItem(**base, action=SyncAction.UPLOAD, sha256=actual_sha, key=key), correction


def build_sync_plan(
    references: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None],
    *,
    allowlist: RedistributableAllowlist,
    store: ObjectStore,
    byte_source: ByteSource,
    apply: bool,
) -> SyncPlan:
    """Plan (and, when *apply*, perform) the R2 mirroring of every allowlisted hostable file in *references*.

    Iterates only :func:`hostable_categories`; within each, only models the *allowlist* clears for
    redistribution are processed (others are recorded as skipped). Each declared file is routed through
    :func:`_plan_file`, accumulating both the per-file outcome and any sha256 correction to backfill.

    Args:
        references: Loaded model references keyed by category (a None value means the category failed to load).
        allowlist: The opt-in redistributable allowlist.
        store: The bucket to check/upload against.
        byte_source: Supplies file bytes from a local mirror or the origin host.
        apply: When True, actually upload; when False, only plan (no ``put`` calls).

    Returns:
        The :class:`SyncPlan` of outcomes and hash corrections.
    """
    plan = SyncPlan()
    for category in hostable_categories():
        records = references.get(category) or {}
        for model_name, record in records.items():
            if not allowlist.allows(model_name=model_name):
                for download in record.config.download:
                    plan.items.append(
                        SyncItem(
                            category=str(category),
                            model_name=model_name,
                            file_name=download.file_name,
                            action=SyncAction.SKIPPED_NOT_ALLOWLISTED,
                        ),
                    )
                continue
            for download in record.config.download:
                item, correction = _plan_file(
                    str(category),
                    record,
                    download,
                    allowlist=allowlist,
                    store=store,
                    byte_source=byte_source,
                    apply=apply,
                )
                plan.items.append(item)
                if correction is not None:
                    plan.corrections.append(correction)
    return plan
