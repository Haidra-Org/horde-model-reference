"""Decide, per declared file, what the R2 sync run should do: skip, upload, or note it is already mirrored.

This is the pure heart of the tool, free of argparse, boto3 and the model-reference manager so it can be tested
against an in-memory store and a hand-built byte source. It enforces the four shaping decisions: only
*hostable* (non-generation) categories, only explicitly *approved* files, *content-addressed* keys, and *backfilled*
hashes for records that still carry the ``"FIXME"`` sentinel.

Bytes are acquired lazily: a record that already declares a real sha256 whose object is present is resolved with
a single ``head`` and no download at all; bytes are fetched only to compute a missing hash or to perform an
actual upload.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit, urlunsplit

from strenum import StrEnum

from horde_model_reference.download_engine import UNKNOWN_SHA256_SENTINEL, sha256_of
from horde_model_reference.licensing import LicenseAssignment, PermissionStatus
from horde_model_reference.meta_consts import (
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    get_category_descriptor,
)
from scripts.r2_sync.allowlist import RedistributionDecision
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
    SKIPPED_UNREVIEWED = "skipped_unreviewed"
    """No category/model/file policy decision exists, so the file remains origin-only."""
    SKIPPED_BLOCKED = "skipped_blocked"
    """The redistribution policy explicitly blocks this file from the mirror."""
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
    size_bytes: int | None = None
    license_expression: str | None = None
    attribution: str | None = None


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
    allowlist: RedistributableAllowlist | None,
    *,
    actual_sha256: str,
    size_bytes: int,
    license_assignment: LicenseAssignment | None,
    fallback_license_expression: str | None,
    fallback_attribution: str | None,
) -> dict[str, str]:
    """Build the provenance and integrity metadata stored with an uploaded object."""
    metadata = {
        "category": category,
        "model_name": record.name,
        "file_name": download.file_name,
        # Reference URLs should be public, but never persist a query/fragment that could contain a signed token.
        "source_url": urlunsplit((*urlsplit(download.file_url)[:3], "", "")),
        "sha256": actual_sha256,
        "size_bytes": str(size_bytes),
    }
    if allowlist is not None:
        metadata.update(
            allowlist.metadata_for(
                category=category,
                model_name=record.name,
                file_name=download.file_name,
            ),
        )
    if license_assignment is not None:
        metadata["license"] = license_assignment.license_expression
        if license_assignment.attribution:
            metadata["attribution"] = license_assignment.attribution
    elif fallback_license_expression is not None:
        metadata["license"] = fallback_license_expression
        if fallback_attribution is not None:
            metadata["attribution"] = fallback_attribution
    return metadata


def _plan_file(
    category: str,
    record: GenericModelRecord,
    download: DownloadRecord,
    *,
    allowlist: RedistributableAllowlist | None,
    store: ObjectStore,
    byte_source: ByteSource,
    apply: bool,
) -> tuple[SyncItem, HashCorrection | None]:
    """Decide and (in *apply* mode) perform the action for one declared file."""
    base = {"category": category, "model_name": record.name, "file_name": download.file_name}
    declared = download.sha256sum
    known_sha = declared if declared and declared != UNKNOWN_SHA256_SENTINEL else None

    decision = (
        allowlist.decision_for(category=category, model_name=record.name, file_name=download.file_name)
        if allowlist is not None
        else None
    )
    license_assignment = record.licensing.assignment_for_file(download.file_name) if record.licensing else None
    license_expression = (
        license_assignment.license_expression
        if license_assignment is not None
        else decision.license_expression
        if decision is not None
        else None
    )
    attribution = (
        license_assignment.attribution
        if license_assignment is not None
        else decision.attribution
        if decision is not None
        else None
    )

    # A known hash with integrity metadata can be resolved without acquiring the bytes.
    if known_sha is not None:
        key = object_key_for(known_sha)
        existing = store.head(key)
        existing_hash = existing.metadata.get("sha256", "").lower() if existing is not None else ""
        declared_size_matches = existing is not None and (
            download.size_bytes is None or existing.size_bytes == download.size_bytes
        )
        if existing is not None and existing_hash == known_sha.lower() and declared_size_matches:
            return (
                SyncItem(
                    **base,
                    action=SyncAction.ALREADY_PRESENT,
                    sha256=known_sha,
                    key=key,
                    size_bytes=existing.size_bytes,
                    license_expression=license_expression,
                    attribution=attribution,
                ),
                None,
            )

    path = byte_source.acquire(record, download)
    if path is None:
        return SyncItem(**base, action=SyncAction.MISSING_BYTES, sha256=known_sha), None

    actual_sha = sha256_of(path, use_cache=False)
    size_bytes = path.stat().st_size
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
        return (
            SyncItem(
                **base,
                action=SyncAction.HASH_MISMATCH,
                sha256=known_sha,
                detail=detail,
                size_bytes=size_bytes,
                license_expression=license_expression,
                attribution=attribution,
            ),
            None,
        )

    key = object_key_for(actual_sha)
    existing = store.head(key)
    if existing is not None and existing.metadata.get("sha256", "").lower() == actual_sha.lower():
        return (
            SyncItem(
                **base,
                action=SyncAction.ALREADY_PRESENT,
                sha256=actual_sha,
                key=key,
                size_bytes=existing.size_bytes,
                license_expression=license_expression,
                attribution=attribution,
            ),
            correction,
        )

    if apply:
        store.put(
            key,
            path,
            metadata=_object_metadata(
                category,
                record,
                download,
                allowlist,
                actual_sha256=actual_sha,
                size_bytes=size_bytes,
                license_assignment=license_assignment,
                fallback_license_expression=license_expression,
                fallback_attribution=attribution,
            ),
        )
    detail = "repaired missing or inconsistent object metadata" if existing is not None else None
    return (
        SyncItem(
            **base,
            action=SyncAction.UPLOAD,
            sha256=actual_sha,
            key=key,
            detail=detail,
            size_bytes=size_bytes,
            license_expression=license_expression,
            attribution=attribution,
        ),
        correction,
    )


def build_sync_plan(
    references: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None],
    *,
    allowlist: RedistributableAllowlist | None,
    store: ObjectStore,
    byte_source: ByteSource,
    apply: bool,
) -> SyncPlan:
    """Plan (and, when *apply*, perform) mirroring of policy-approved hostable files in *references*.

    Iterates only :func:`hostable_categories`; within each, only files the redistribution policy clears for
    redistribution are processed (others are recorded as skipped). Each declared file is routed through
    :func:`_plan_file`, accumulating both the per-file outcome and any sha256 correction to backfill.

    Args:
        references: Loaded model references keyed by category (a None value means the category failed to load).
        allowlist: The strict category/model/file redistribution policy (legacy parameter name).
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
            for download in record.config.download:
                decision = (
                    allowlist.decision_for(
                        category=category,
                        model_name=model_name,
                        file_name=download.file_name,
                    )
                    if allowlist is not None
                    else None
                )
                if decision is None:
                    plan.items.append(
                        SyncItem(
                            category=str(category),
                            model_name=model_name,
                            file_name=download.file_name,
                            action=SyncAction.SKIPPED_UNREVIEWED,
                        ),
                    )
                    continue
                if decision.decision is RedistributionDecision.BLOCKED:
                    plan.items.append(
                        SyncItem(
                            category=str(category),
                            model_name=model_name,
                            file_name=download.file_name,
                            action=SyncAction.SKIPPED_BLOCKED,
                            detail=decision.note,
                        ),
                    )
                    continue
                effective_assignment = (
                    record.licensing.assignment_for_file(download.file_name) if record.licensing is not None else None
                )
                if effective_assignment is not None and effective_assignment.redistribution in {
                    PermissionStatus.PROHIBITED,
                    PermissionStatus.UNKNOWN,
                }:
                    plan.items.append(
                        SyncItem(
                            category=str(category),
                            model_name=model_name,
                            file_name=download.file_name,
                            action=SyncAction.SKIPPED_BLOCKED,
                            detail=f"licensing redistribution status is {effective_assignment.redistribution.value}",
                            license_expression=effective_assignment.license_expression,
                            attribution=effective_assignment.attribution,
                        ),
                    )
                    continue
                if (
                    effective_assignment is not None
                    and decision.license_expression is not None
                    and decision.license_expression != effective_assignment.license_expression
                ):
                    plan.items.append(
                        SyncItem(
                            category=str(category),
                            model_name=model_name,
                            file_name=download.file_name,
                            action=SyncAction.SKIPPED_UNREVIEWED,
                            detail="legacy policy expression differs from canonical model licensing",
                            license_expression=effective_assignment.license_expression,
                            attribution=effective_assignment.attribution,
                        ),
                    )
                    continue
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
