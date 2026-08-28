"""Persistence, concurrency-control, and registry hydration for served image baselines."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_model_reference import ModelReferenceManager, PrefetchStrategy, ReplicateMode
from horde_model_reference.backends.github_backend import GitHubBackend
from horde_model_reference.backends.http_backend import HTTPBackend
from horde_model_reference.image_baseline import (
    BaselineCapabilities,
    HordeBaselinePolicy,
    ImageBaselineChange,
    ImageBaselineChangeSet,
    ImageBaselineRecord,
)
from horde_model_reference.image_baseline_store import BaselineConflictError, ImageBaselineStore
from horde_model_reference.meta_consts import get_baseline_native_resolution, is_known_image_baseline

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "src" / "horde_model_reference" / "data" / "baselines"


def _record(name: str, *, native_resolution: int = 1024) -> ImageBaselineRecord:
    """Build a baseline record with the capabilities of a modern flow-matching family."""
    return ImageBaselineRecord(
        name=name,
        display_name=name.replace("_", " ").title(),
        native_resolution=native_resolution,
        capabilities=BaselineCapabilities(
            controlnet=False,
            transparent=False,
            qr_code=False,
            remix=False,
            flow_matching=True,
        ),
        horde_policy=HordeBaselinePolicy(kudos=8, batching=5, ttl=3, resolution_floor=1024),
    )


def test_packaged_bootstrap_supplies_the_published_baseline_vocabulary(tmp_path: Path) -> None:
    """A store with no runtime data serves the packaged catalog and its horde policy."""
    store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH)

    names = {record.name for record in store.list()}
    assert {"stable_diffusion_1", "stable_diffusion_xl", "flux_1", "krea2_turbo", "anima"} <= names
    stable_diffusion_xl = store.get("stable_diffusion_xl")
    assert stable_diffusion_xl is not None
    assert stable_diffusion_xl.native_resolution == 1024
    assert stable_diffusion_xl.horde_policy.kudos_qr_code == 4
    assert store.get("no_such_baseline") is None
    assert (tmp_path / "catalog.json").is_file()


def test_upsert_bumps_the_revision_and_teaches_the_registry_the_new_baseline(tmp_path: Path) -> None:
    """A published baseline becomes a known name without any packaged vocabulary change."""
    store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH)
    assert not is_known_image_baseline("test_published_baseline")

    catalog = store.apply_change_set(
        ImageBaselineChangeSet(
            title="Publish a new family",
            changes=[
                ImageBaselineChange(
                    operation="upsert", name="test_published_baseline", record=_record("test_published_baseline")
                ),
            ],
        ),
        referenced_baselines=set(),
        editor_id="maintainer",
    )

    assert catalog.metadata.revision == 2
    assert store.metadata().revision == 2
    assert is_known_image_baseline("test_published_baseline")
    assert get_baseline_native_resolution("test_published_baseline") == 1024


def test_a_stale_precondition_is_refused_as_a_conflict(tmp_path: Path) -> None:
    """A proposal reviewed against a different value never overwrites the current one."""
    store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH)

    with pytest.raises(BaselineConflictError, match="changed after review"):
        store.apply_change_set(
            ImageBaselineChangeSet(
                title="Retune an already-edited baseline",
                changes=[
                    ImageBaselineChange(
                        operation="upsert",
                        name="flux_1",
                        record=_record("flux_1"),
                        expected_before=_record("flux_1", native_resolution=512),
                    ),
                ],
            ),
            referenced_baselines=set(),
            editor_id="maintainer",
        )

    assert store.metadata().revision == 1


def test_replaying_an_already_persisted_change_set_is_a_revision_preserving_noop(tmp_path: Path) -> None:
    """A queue retry can finish after catalog persistence succeeded but queue marking failed."""
    store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH)
    change_set = ImageBaselineChangeSet(
        title="Publish a retry-safe family",
        changes=[
            ImageBaselineChange(
                operation="upsert",
                name="test_retry_baseline",
                record=_record("test_retry_baseline"),
            ),
        ],
    )

    first = store.apply_change_set(change_set, referenced_baselines=set(), editor_id="maintainer")
    replay = store.apply_change_set(change_set, referenced_baselines=set(), editor_id="maintainer")

    assert first.metadata.revision == 2
    assert replay == first
    assert store.metadata().revision == 2


def test_a_baseline_named_by_a_canonical_model_cannot_be_deleted(tmp_path: Path) -> None:
    """Removing a baseline models still name would strand those records."""
    store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH)
    current = store.get("stable_cascade")

    with pytest.raises(ValueError, match="still named by a canonical image model"):
        store.apply_change_set(
            ImageBaselineChangeSet(
                title="Retire Stable Cascade",
                changes=[ImageBaselineChange(operation="delete", name="stable_cascade", expected_before=current)],
            ),
            referenced_baselines={"stable_cascade"},
            editor_id="maintainer",
        )

    assert store.get("stable_cascade") == current


def test_replica_refresh_hydrates_from_export_and_rejects_a_malformed_one(tmp_path: Path) -> None:
    """A read-only replica adopts a coherent PRIMARY export and keeps its last one otherwise."""
    replica_store = ImageBaselineStore(root_path=tmp_path, bootstrap_path=_BOOTSTRAP_PATH, writable=False)
    published = _record("test_replicated_baseline")

    replica_store.refresh_replica_export(
        {
            "metadata": {"schema_version": 1, "revision": 7, "updated_at": 123},
            "baselines": {"test_replicated_baseline": published.model_dump(mode="json")},
        },
    )

    assert replica_store.metadata().revision == 7
    assert replica_store.get("test_replicated_baseline") == published
    assert is_known_image_baseline("test_replicated_baseline")
    with pytest.raises(PermissionError):
        replica_store.apply_change_set(
            ImageBaselineChangeSet(
                title="Edit a replica",
                changes=[ImageBaselineChange(operation="delete", name="test_replicated_baseline")],
            ),
            referenced_baselines=set(),
            editor_id="replica",
        )

    with pytest.raises(ValueError, match="does not match its name"):
        replica_store.refresh_replica_export(
            {
                "metadata": {"schema_version": 1, "revision": 99},
                "baselines": {"mislabelled": published.model_dump(mode="json")},
            },
        )

    assert replica_store.metadata().revision == 7
    assert replica_store.get("test_replicated_baseline") == published


def test_replica_manager_hydrates_once_and_refreshes_on_demand(
    tmp_path: Path,
    restore_manager_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-lived replica picks up a baseline published after start-up only when asked to."""
    github_backend = GitHubBackend(base_path=tmp_path, replicate_mode=ReplicateMode.REPLICA)
    backend = HTTPBackend(primary_api_url="https://primary", github_backend=github_backend)
    fetch_count = 0

    def fake_fetch(_self: HTTPBackend) -> dict[str, object]:
        nonlocal fetch_count
        fetch_count += 1
        return {
            "metadata": {"schema_version": 1, "revision": fetch_count},
            "baselines": {
                f"test_hydrated_baseline_{fetch_count}": _record(
                    f"test_hydrated_baseline_{fetch_count}",
                ).model_dump(mode="json"),
            },
        }

    monkeypatch.setattr(HTTPBackend, "fetch_image_baseline_export", fake_fetch)
    manager = ModelReferenceManager(
        backend=backend,
        prefetch_strategy=PrefetchStrategy.LAZY,
        replicate_mode=ReplicateMode.REPLICA,
    )

    assert manager.image_baseline_store.get("test_hydrated_baseline_1") is not None
    assert manager.image_baseline_store.metadata().revision == 1
    assert fetch_count == 1

    assert manager.refresh_image_baselines() is True
    assert manager.image_baseline_store.get("test_hydrated_baseline_2") is not None
    assert fetch_count == 2
