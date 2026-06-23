"""Tests for the annotator hosting pass of the R2 sync tool (the catalog -> R2 mirror side).

Exercised against the in-memory object store and a hand-built byte source, mirroring the category planner tests:
repo-level allowlisting, content-addressed upload/skip, hash computation/backfill, and mismatch reporting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.r2_sync.annotators as annotators_module
from horde_model_reference.annotator_catalog import AnnotatorFile
from horde_model_reference.download_engine import DownloadOutcome
from scripts.r2_sync.allowlist import RedistributableAllowlist, RedistributableEntry
from scripts.r2_sync.annotators import (
    ANNOTATOR_CATEGORY,
    AnnotatorByteSource,
    build_annotator_plan,
)
from scripts.r2_sync.object_store import InMemoryObjectStore, object_key_for
from scripts.r2_sync.planner import SyncAction


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _allow_repo(repo: str) -> RedistributableAllowlist:
    return RedistributableAllowlist(entries={repo: RedistributableEntry(name=repo)})


class _DictByteSource:
    """An annotator byte source backed by a fixed mapping of filename -> path (None entries are unavailable)."""

    def __init__(self, by_filename: dict[str, Path | None]) -> None:
        self._by_filename = by_filename
        self.acquired: list[str] = []

    def acquire(self, entry: AnnotatorFile) -> Path | None:
        self.acquired.append(entry.filename)
        return self._by_filename.get(entry.filename)


def test_annotators_are_repo_allowlisted(tmp_path: Path) -> None:
    """Verify an annotator whose repo is not cleared is skipped and its bytes are never acquired."""
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="ControlNetHED.pth")
    byte_source = _DictByteSource({"ControlNetHED.pth": tmp_path / "x"})

    plan = build_annotator_plan(
        allowlist=RedistributableAllowlist(),  # empty
        store=InMemoryObjectStore(),
        byte_source=byte_source,  # type: ignore[arg-type]
        apply=True,
        catalog=[entry],
    )

    assert [item.action for item in plan.items] == [SyncAction.SKIPPED_NOT_ALLOWLISTED]
    assert plan.items[0].category == ANNOTATOR_CATEGORY
    assert byte_source.acquired == []


def test_unhashed_annotator_is_uploaded_and_hash_backfilled(tmp_path: Path) -> None:
    """Verify a catalog file (sha256 None) gets its hash computed, uploaded by-hash, and queued for backfill."""
    payload = b"controlnet hed weights"
    sha = _sha256(payload)
    path = tmp_path / "ControlNetHED.pth"
    path.write_bytes(payload)
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="ControlNetHED.pth")
    store = InMemoryObjectStore()

    plan = build_annotator_plan(
        allowlist=_allow_repo("lllyasviel/Annotators"),
        store=store,
        byte_source=_DictByteSource({"ControlNetHED.pth": path}),  # type: ignore[arg-type]
        apply=True,
        catalog=[entry],
    )

    assert [item.action for item in plan.items] == [SyncAction.UPLOAD]
    assert store.head(object_key_for(sha)) is True
    assert store.objects[object_key_for(sha)]["source_url"] == entry.origin_url
    assert len(plan.corrections) == 1
    assert plan.corrections[0].new_sha256 == sha
    assert plan.corrections[0].file_name == "ControlNetHED.pth"


def test_annotator_upload_stamps_redistribution_provenance(tmp_path: Path) -> None:
    """Verify repo-level licence review metadata is attached to annotator uploads."""
    payload = b"reviewed annotator"
    sha = _sha256(payload)
    path = tmp_path / "ControlNetHED.pth"
    path.write_bytes(payload)
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="ControlNetHED.pth")
    allowlist = RedistributableAllowlist(
        entries={
            "lllyasviel/Annotators": RedistributableEntry(
                name="lllyasviel/Annotators",
                license="Apache-2.0",
                note="repo reviewed for redistribution",
            ),
        },
    )
    store = InMemoryObjectStore()

    build_annotator_plan(
        allowlist=allowlist,
        store=store,
        byte_source=_DictByteSource({"ControlNetHED.pth": path}),  # type: ignore[arg-type]
        apply=True,
        catalog=[entry],
    )

    metadata = store.objects[object_key_for(sha)]
    assert metadata["license"] == "Apache-2.0"
    assert metadata["redistribution_note"] == "repo reviewed for redistribution"


def test_present_object_is_skipped_idempotently(tmp_path: Path) -> None:
    """Verify a file whose content-addressed object already exists is reported present, not re-uploaded."""
    payload = b"already mirrored annotator"
    sha = _sha256(payload)
    path = tmp_path / "res101.pth"
    path.write_bytes(payload)
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="res101.pth")

    plan = build_annotator_plan(
        allowlist=_allow_repo("lllyasviel/Annotators"),
        store=InMemoryObjectStore(present={object_key_for(sha)}),
        byte_source=_DictByteSource({"res101.pth": path}),  # type: ignore[arg-type]
        apply=True,
        catalog=[entry],
    )

    assert [item.action for item in plan.items] == [SyncAction.ALREADY_PRESENT]


def test_byte_source_prefers_local_checkpoint(tmp_path: Path) -> None:
    """Verify the real byte source returns the on-disk checkpoint under the ckpts dir without fetching origin."""
    payload = b"local annotator copy"
    ckpts = tmp_path / "annotators"
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="mlsd_large_512_fp32.pth")
    local = ckpts.joinpath(*entry.relative_path.split("/"))
    local.parent.mkdir(parents=True)
    local.write_bytes(payload)
    source = AnnotatorByteSource(ckpts_dir=ckpts, cache_dir=tmp_path / "cache")

    resolved = source.acquire(entry)

    assert resolved == local
    assert source._origin_fetches == 0


def test_annotator_byte_source_replaces_bad_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify an annotator cache file that fails a known sha256 is removed and refetched."""
    good = b"correct annotator"
    entry_unknown = AnnotatorFile(repo="lllyasviel/Annotators", filename="ControlNetHED.pth")
    entry_known = AnnotatorFile(
        repo="lllyasviel/Annotators",
        filename="ControlNetHED.pth",
        sha256=_sha256(good),
    )
    source = AnnotatorByteSource(ckpts_dir=None, cache_dir=tmp_path / "cache")

    def write_bad(_url: str, destination: Path, **_kwargs: object) -> DownloadOutcome:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad cache")
        return DownloadOutcome(success=True, final_path=destination, bytes_written=9, sha256=_sha256(b"bad cache"))

    monkeypatch.setattr(annotators_module, "download_file", write_bad)
    poisoned = source.acquire(entry_unknown)
    assert poisoned is not None
    assert poisoned.read_bytes() == b"bad cache"

    def write_good(_url: str, destination: Path, **_kwargs: object) -> DownloadOutcome:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(good)
        return DownloadOutcome(success=True, final_path=destination, bytes_written=len(good), sha256=_sha256(good))

    monkeypatch.setattr(annotators_module, "download_file", write_good)
    resolved = source.acquire(entry_known)

    assert resolved == poisoned
    assert resolved.read_bytes() == good
    assert source._origin_fetches == 2


def test_byte_source_reports_missing_without_local_or_cache() -> None:
    """Verify a missing local checkpoint with no cache dir reports unavailable rather than downloading."""
    entry = AnnotatorFile(repo="lllyasviel/Annotators", filename="facenet.pth")
    source = AnnotatorByteSource(ckpts_dir=Path("/nonexistent"), cache_dir=None)

    assert source.acquire(entry) is None
