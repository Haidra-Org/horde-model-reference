"""Tests for the devops R2 sync tool: allowlist gating, content-addressed planning, and hash backfill.

The planner is exercised against an in-memory object store and a dict-backed byte source so the four shaping
behaviours (hostable-only, allowlisted-only, idempotent uploads, FIXME backfill) are verified without a live
bucket or network.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import scripts.r2_sync.byte_source as byte_source_module
from horde_model_reference.download_engine import DownloadOutcome
from horde_model_reference.licensing import (
    LicenseAssignment,
    ModelLicensing,
    PermissionStatus,
    unknown_model_licensing,
)
from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from scripts.r2_sync.allowlist import RedistributableAllowlist, RedistributableEntry, RedistributionDecision
from scripts.r2_sync.byte_source import LocalThenOriginByteSource
from scripts.r2_sync.manifest import build_mirror_manifest
from scripts.r2_sync.object_store import InMemoryObjectStore, object_key_for
from scripts.r2_sync.planner import (
    SyncAction,
    SyncItem,
    SyncPlan,
    build_sync_plan,
    hostable_categories,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _allow(*names: str) -> RedistributableAllowlist:
    """Build a per-model allowlist clearing each of *names* for redistribution."""
    entries = {
        (category, name): RedistributableEntry(
            category=category,
            name=name,
            decision=RedistributionDecision.APPROVED,
            license_expression="MIT",
            evidence=("test evidence",),
            reviewed_by="test",
            reviewed_at=date(2026, 1, 1),
        )
        for category in hostable_categories()
        for name in names
    }
    return RedistributableAllowlist(entries=entries)


def _record(
    name: str,
    category: MODEL_REFERENCE_CATEGORY,
    downloads: list[DownloadRecord],
    *,
    licensing: ModelLicensing | None = None,
) -> GenericModelRecord:
    descriptor_purpose = (
        MODEL_PURPOSE.miscellaneous
        if category == MODEL_REFERENCE_CATEGORY.miscellaneous
        else MODEL_PURPOSE.post_processing
    )
    return GenericModelRecord(
        name=name,
        record_type=category,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=descriptor_purpose),
        config=GenericModelRecordConfig(download=downloads),
        licensing=licensing,
    )


class _DictByteSource:
    """A byte source backed by a fixed mapping of file_name -> on-disk path (None entries are 'unavailable')."""

    def __init__(self, by_file_name: dict[str, Path | None]) -> None:
        self._by_file_name = by_file_name
        self.acquired: list[str] = []

    def acquire(self, record: GenericModelRecord, download: DownloadRecord) -> Path | None:
        self.acquired.append(download.file_name)
        return self._by_file_name.get(download.file_name)


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    target = tmp_path / name
    target.write_bytes(data)
    return target


def test_hostable_categories_excludes_generation_and_managed_elsewhere() -> None:
    """Verify the hostable set is the non-generation auxiliary categories, not checkpoints or LoRA/TI."""
    hostable = {str(c) for c in hostable_categories()}

    assert "controlnet" in hostable
    assert "controlnet_annotator" in hostable
    assert "esrgan" in hostable
    assert "safety_checker" in hostable
    assert "miscellaneous" in hostable
    assert "image_generation" not in hostable
    assert "text_generation" not in hostable
    assert "lora" not in hostable
    assert "ti" not in hostable


def test_non_allowlisted_models_are_skipped(tmp_path: Path) -> None:
    """Verify a model not opted in for redistribution is skipped and never has its bytes acquired."""
    payload = b"esrgan weights"
    record = _record(
        "RealESRGAN_x4plus",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x4.pth", file_url="https://origin/x4.pth", sha256sum=_sha256(payload))],
    )
    byte_source = _DictByteSource({"x4.pth": _write(tmp_path, "x4.pth", payload)})

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.esrgan: {"RealESRGAN_x4plus": record}},
        allowlist=RedistributableAllowlist(),  # empty: nothing opted in
        store=InMemoryObjectStore(),
        byte_source=byte_source,
        apply=True,
    )

    assert [item.action for item in plan.items] == [SyncAction.SKIPPED_UNREVIEWED]
    assert byte_source.acquired == []


def test_explicitly_blocked_model_is_documented_and_never_acquired(tmp_path: Path) -> None:
    """A blocked policy decision records its reason without touching model bytes."""
    payload = b"restricted weights"
    record = _record(
        "restricted",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="r.bin", file_url="https://origin/r", sha256sum=_sha256(payload))],
    )
    category = MODEL_REFERENCE_CATEGORY.esrgan
    policy = RedistributableAllowlist(
        entries={
            (category, record.name): RedistributableEntry(
                category=category,
                name=record.name,
                decision=RedistributionDecision.BLOCKED,
                note="redistribution is not permitted",
            ),
        },
    )
    byte_source = _DictByteSource({"r.bin": _write(tmp_path, "r.bin", payload)})

    plan = build_sync_plan(
        {category: {record.name: record}},
        allowlist=policy,
        store=InMemoryObjectStore(),
        byte_source=byte_source,
        apply=True,
    )

    assert plan.items[0].action == SyncAction.SKIPPED_BLOCKED
    assert plan.items[0].detail == "redistribution is not permitted"
    assert byte_source.acquired == []


def test_policy_approval_cannot_override_unknown_canonical_redistribution(tmp_path: Path) -> None:
    """Verify stale operational approval cannot turn an unaudited model into a hosted artifact."""
    payload = b"unaudited weights"
    record = _record(
        "unaudited",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="unknown.bin", file_url="https://origin/unknown", sha256sum=_sha256(payload))],
        licensing=unknown_model_licensing(),
    )
    byte_source = _DictByteSource({"unknown.bin": _write(tmp_path, "unknown.bin", payload)})

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.esrgan: {record.name: record}},
        allowlist=_allow(record.name),
        store=InMemoryObjectStore(),
        byte_source=byte_source,
        apply=True,
    )

    assert plan.items[0].action == SyncAction.SKIPPED_BLOCKED
    assert plan.items[0].license_expression == "NOASSERTION"
    assert "unknown" in str(plan.items[0].detail)
    assert byte_source.acquired == []


def test_file_override_blocks_only_the_restricted_artifact(tmp_path: Path) -> None:
    """Verify file-level licensing takes precedence without suppressing a separately cleared file."""
    allowed_payload = b"allowed weights"
    restricted_payload = b"restricted weights"
    licensing = ModelLicensing(
        license_expression="MIT",
        license_ids=("MIT",),
        commercial_use=PermissionStatus.ALLOWED,
        redistribution=PermissionStatus.ALLOWED_WITH_CONDITIONS,
        files={
            "restricted.bin": LicenseAssignment(
                license_expression="LicenseRef-Restricted",
                license_ids=("LicenseRef-Restricted",),
                commercial_use=PermissionStatus.PROHIBITED,
                redistribution=PermissionStatus.PROHIBITED,
            ),
        },
    )
    record = _record(
        "mixed-files",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [
            DownloadRecord(
                file_name="allowed.bin",
                file_url="https://origin/allowed",
                sha256sum=_sha256(allowed_payload),
            ),
            DownloadRecord(
                file_name="restricted.bin",
                file_url="https://origin/restricted",
                sha256sum=_sha256(restricted_payload),
            ),
        ],
        licensing=licensing,
    )
    byte_source = _DictByteSource(
        {
            "allowed.bin": _write(tmp_path, "allowed.bin", allowed_payload),
            "restricted.bin": _write(tmp_path, "restricted.bin", restricted_payload),
        },
    )

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.esrgan: {record.name: record}},
        allowlist=_allow(record.name),
        store=InMemoryObjectStore(),
        byte_source=byte_source,
        apply=True,
    )

    actions_by_file = {item.file_name: item.action for item in plan.items}
    assert actions_by_file == {
        "allowed.bin": SyncAction.UPLOAD,
        "restricted.bin": SyncAction.SKIPPED_BLOCKED,
    }
    assert byte_source.acquired == ["allowed.bin"]


def test_allowlisted_missing_object_is_uploaded_and_present_object_is_skipped(tmp_path: Path) -> None:
    """Verify an absent content-addressed object is uploaded, and a second run finds it already present."""
    payload = b"upscaler weights blob"
    sha = _sha256(payload)
    record = _record(
        "RealESRGAN_x4plus",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x4.pth", file_url="https://origin/x4.pth", sha256sum=sha)],
    )
    references = {MODEL_REFERENCE_CATEGORY.esrgan: {"RealESRGAN_x4plus": record}}
    allowlist = _allow("RealESRGAN_x4plus")
    store = InMemoryObjectStore()
    byte_source = _DictByteSource({"x4.pth": _write(tmp_path, "x4.pth", payload)})

    first = build_sync_plan(references, allowlist=allowlist, store=store, byte_source=byte_source, apply=True)
    assert [item.action for item in first.items] == [SyncAction.UPLOAD]
    assert store.head(object_key_for(sha)) is not None
    assert store.objects[object_key_for(sha)]["model_name"] == "RealESRGAN_x4plus"

    second = build_sync_plan(references, allowlist=allowlist, store=store, byte_source=byte_source, apply=True)
    assert [item.action for item in second.items] == [SyncAction.ALREADY_PRESENT]


def test_present_record_with_known_hash_needs_no_bytes() -> None:
    """Verify an already-present object with a real declared hash is resolved with a head and no byte acquire."""
    sha = _sha256(b"already mirrored")
    record = _record(
        "model",
        MODEL_REFERENCE_CATEGORY.controlnet,
        [DownloadRecord(file_name="c.safetensors", file_url="https://origin/c", sha256sum=sha)],
    )
    byte_source = _DictByteSource({})  # would return None if ever consulted

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.controlnet: {"model": record}},
        allowlist=_allow("model"),
        store=InMemoryObjectStore(present={object_key_for(sha)}),
        byte_source=byte_source,
        apply=True,
    )

    assert [item.action for item in plan.items] == [SyncAction.ALREADY_PRESENT]
    assert byte_source.acquired == []


def test_fixme_hash_is_computed_and_backfilled(tmp_path: Path) -> None:
    """Verify a 'FIXME' record gets its real sha256 computed, uploaded under that hash, and queued for backfill."""
    payload = b"blip captioner"
    sha = _sha256(payload)
    record = _record(
        "blip_large",
        MODEL_REFERENCE_CATEGORY.blip,
        [DownloadRecord(file_name="blip.pth", file_url="https://origin/blip.pth", sha256sum="FIXME")],
    )
    store = InMemoryObjectStore()
    byte_source = _DictByteSource({"blip.pth": _write(tmp_path, "blip.pth", payload)})

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.blip: {"blip_large": record}},
        allowlist=_allow("blip_large"),
        store=store,
        byte_source=byte_source,
        apply=True,
    )

    assert [item.action for item in plan.items] == [SyncAction.UPLOAD]
    assert store.head(object_key_for(sha)) is not None
    assert len(plan.corrections) == 1
    correction = plan.corrections[0]
    assert correction.old_sha256 == "FIXME"
    assert correction.new_sha256 == sha
    assert correction.model_name == "blip_large"


def test_hash_mismatch_is_reported_not_uploaded(tmp_path: Path) -> None:
    """Verify bytes that do not match a record's declared hash are flagged and never uploaded."""
    record = _record(
        "model",
        MODEL_REFERENCE_CATEGORY.gfpgan,
        [DownloadRecord(file_name="g.pth", file_url="https://origin/g", sha256sum=_sha256(b"expected"))],
    )
    store = InMemoryObjectStore()
    byte_source = _DictByteSource({"g.pth": _write(tmp_path, "g.pth", b"actually different bytes")})

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.gfpgan: {"model": record}},
        allowlist=_allow("model"),
        store=store,
        byte_source=byte_source,
        apply=True,
    )

    assert [item.action for item in plan.items] == [SyncAction.HASH_MISMATCH]
    assert store.objects == {}


def test_dry_run_does_not_upload(tmp_path: Path) -> None:
    """Verify a dry-run plans an upload without writing to the store."""
    payload = b"codeformer"
    record = _record(
        "codeformer",
        MODEL_REFERENCE_CATEGORY.codeformer,
        [DownloadRecord(file_name="cf.pth", file_url="https://origin/cf", sha256sum=_sha256(payload))],
    )
    store = InMemoryObjectStore()

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.codeformer: {"codeformer": record}},
        allowlist=_allow("codeformer"),
        store=store,
        byte_source=_DictByteSource({"cf.pth": _write(tmp_path, "cf.pth", payload)}),
        apply=False,
    )

    assert [item.action for item in plan.items] == [SyncAction.UPLOAD]
    assert store.objects == {}  # nothing actually uploaded


def test_byte_source_prefers_local_copy(tmp_path: Path) -> None:
    """Verify the real byte source returns the on-disk file under the weights root without fetching origin."""
    payload = b"local mirror copy"
    weights_root = tmp_path / "models"
    (weights_root / "esrgan").mkdir(parents=True)
    local = weights_root / "esrgan" / "x4.pth"
    local.write_bytes(payload)
    record = _record(
        "RealESRGAN_x4plus",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x4.pth", file_url="https://origin/should-not-be-used", sha256sum=_sha256(payload))],
    )
    source = LocalThenOriginByteSource(weights_root=weights_root, cache_dir=tmp_path / "cache")

    resolved = source.acquire(record, record.config.download[0])

    assert resolved == local
    assert source._origin_fetches == 0


def test_byte_source_cache_path_cannot_escape_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify reference-controlled file names cannot route origin cache writes outside the cache dir."""
    payload = b"downloaded bytes"
    cache_dir = tmp_path / "cache"

    def fake_download(_url: str, destination: Path, **_kwargs: object) -> DownloadOutcome:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return DownloadOutcome(
            success=True,
            final_path=destination,
            bytes_written=len(payload),
            sha256=_sha256(payload),
        )

    monkeypatch.setattr(byte_source_module, "download_file", fake_download)
    record = _record(
        "unsafe/model",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="../escape.pth", file_url="https://origin/escape", sha256sum=_sha256(payload))],
    )
    source = LocalThenOriginByteSource(weights_root=tmp_path / "missing", cache_dir=cache_dir)

    resolved = source.acquire(record, record.config.download[0])

    assert resolved is not None
    assert resolved.is_relative_to(cache_dir)
    assert (tmp_path / "escape.pth").exists() is False
    assert resolved.read_bytes() == payload


def test_byte_source_replaces_bad_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a cache file that fails the declared sha256 is removed and fetched again."""
    good = b"correct bytes"
    cache_dir = tmp_path / "cache"
    first_record = _record(
        "model",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x.pth", file_url="https://origin/x", sha256sum="FIXME")],
    )
    source = LocalThenOriginByteSource(weights_root=tmp_path / "missing", cache_dir=cache_dir)

    def write_bad(_url: str, destination: Path, **_kwargs: object) -> DownloadOutcome:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bad cache")
        return DownloadOutcome(success=True, final_path=destination, bytes_written=9, sha256=_sha256(b"bad cache"))

    monkeypatch.setattr(byte_source_module, "download_file", write_bad)
    poisoned = source.acquire(first_record, first_record.config.download[0])
    assert poisoned is not None
    assert poisoned.read_bytes() == b"bad cache"

    def write_good(_url: str, destination: Path, **_kwargs: object) -> DownloadOutcome:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(good)
        return DownloadOutcome(success=True, final_path=destination, bytes_written=len(good), sha256=_sha256(good))

    second_record = _record(
        "model",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x.pth", file_url="https://origin/x", sha256sum=_sha256(good))],
    )
    monkeypatch.setattr(byte_source_module, "download_file", write_good)

    resolved = source.acquire(second_record, second_record.config.download[0])

    assert resolved == poisoned
    assert resolved is not None
    assert resolved.read_bytes() == good
    assert source._origin_fetches == 2


def test_allowlist_is_per_model_not_per_category(tmp_path: Path) -> None:
    """Verify clearing one model does not clear a category-mate: opt-in is strictly by name."""
    cleared_payload = b"cleared upscaler"
    other_payload = b"unreviewed upscaler"
    cleared = _record(
        "Cleared",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="cleared.pth", file_url="https://o/c", sha256sum=_sha256(cleared_payload))],
    )
    other = _record(
        "Unreviewed",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="other.pth", file_url="https://origin/other", sha256sum=_sha256(other_payload))],
    )
    byte_source = _DictByteSource(
        {
            "cleared.pth": _write(tmp_path, "cleared.pth", cleared_payload),
            "other.pth": _write(tmp_path, "other.pth", other_payload),
        },
    )

    plan = build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.esrgan: {"Cleared": cleared, "Unreviewed": other}},
        allowlist=_allow("Cleared"),
        store=InMemoryObjectStore(),
        byte_source=byte_source,
        apply=True,
    )

    by_model = {item.model_name: item.action for item in plan.items}
    assert by_model["Cleared"] == SyncAction.UPLOAD
    assert by_model["Unreviewed"] == SyncAction.SKIPPED_UNREVIEWED
    assert "other.pth" not in byte_source.acquired  # the unreviewed model's bytes are never touched


def test_redistribution_provenance_is_stamped_on_uploaded_object(tmp_path: Path) -> None:
    """Verify the licence/note recorded when clearing a model are attached to its uploaded object's metadata."""
    payload = b"a redistributable upscaler"
    sha = _sha256(payload)
    record = _record(
        "RealESRGAN_x4plus",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="x4.pth", file_url="https://origin/x4.pth", sha256sum=sha)],
    )
    allowlist = RedistributableAllowlist(
        entries={
            (MODEL_REFERENCE_CATEGORY.esrgan, "RealESRGAN_x4plus"): RedistributableEntry(
                category=MODEL_REFERENCE_CATEGORY.esrgan,
                name="RealESRGAN_x4plus",
                decision=RedistributionDecision.APPROVED,
                license_expression="BSD-3-Clause",
                evidence=("upstream LICENSE",),
                reviewed_by="test",
                reviewed_at=date(2026, 1, 1),
                note="upstream README permits redistribution",
            ),
        },
    )
    store = InMemoryObjectStore()

    build_sync_plan(
        {MODEL_REFERENCE_CATEGORY.esrgan: {"RealESRGAN_x4plus": record}},
        allowlist=allowlist,
        store=store,
        byte_source=_DictByteSource({"x4.pth": _write(tmp_path, "x4.pth", payload)}),
        apply=True,
    )

    metadata = store.objects[object_key_for(sha)]
    assert metadata["license"] == "BSD-3-Clause"
    assert metadata["source_url"] == "https://origin/x4.pth"


def test_policy_load_rejects_bare_names_and_requires_review_evidence(tmp_path: Path) -> None:
    """Verify legacy bare-name allowlisting cannot authorize an upload."""
    path = tmp_path / "allow.json"
    path.write_text(
        '{"models": ["Terse", {"name": "Reviewed", "license": "MIT", "note": "ok"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        RedistributableAllowlist.load(path)


def test_byte_source_returns_none_without_local_or_cache() -> None:
    """Verify a missing local file with no cache dir reports unavailable rather than attempting a download."""
    record = _record(
        "model",
        MODEL_REFERENCE_CATEGORY.esrgan,
        [DownloadRecord(file_name="missing.pth", file_url="https://origin/missing", sha256sum="FIXME")],
    )
    source = LocalThenOriginByteSource(weights_root=Path("/nonexistent-root"), cache_dir=None)

    assert source.acquire(record, record.config.download[0]) is None


def test_policy_identity_includes_category() -> None:
    """An approval for a same-named model in one category must not authorize another category."""
    policy = _allow("same-name")
    esrgan_entry = policy.entries[(MODEL_REFERENCE_CATEGORY.esrgan, "same-name")]
    narrowed = RedistributableAllowlist(
        entries={(MODEL_REFERENCE_CATEGORY.esrgan, "same-name"): esrgan_entry},
    )

    assert narrowed.allows(
        category=MODEL_REFERENCE_CATEGORY.esrgan,
        model_name="same-name",
        file_name="weights.bin",
    )
    assert not narrowed.allows(
        category=MODEL_REFERENCE_CATEGORY.gfpgan,
        model_name="same-name",
        file_name="weights.bin",
    )


def test_manifest_deduplicates_hash_and_preserves_references() -> None:
    """The published inventory stores one object per hash while retaining every model reference."""
    sha256 = _sha256(b"shared")
    plan = SyncPlan(
        items=[
            SyncItem(
                category="esrgan",
                model_name=model_name,
                file_name=f"{model_name}.bin",
                action=SyncAction.ALREADY_PRESENT,
                sha256=sha256,
                size_bytes=6,
                license_expression="MIT",
            )
            for model_name in ("one", "two")
        ],
    )

    manifest = build_mirror_manifest(plan, generated_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert list(manifest.objects) == [sha256]
    assert {reference.model_name for reference in manifest.objects[sha256].references} == {"one", "two"}
