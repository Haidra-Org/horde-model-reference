"""Tests for the resumable, disk-first machinery of the component-hashing pass.

These cover the new progress file, the progress-to-record re-application, the stable task key, and the
local/absent read policy. The full disk pre-flight (path resolution against a weights root) is exercised by
running the script against a real reference; here the pure, network-free pieces are pinned.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import requests

from horde_model_reference import ModelClassification
from horde_model_reference.component_hash import ComponentKind, hash_standalone_component_file
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from scripts.hash_components import (
    ComponentHashTask,
    HashProgress,
    PlannedHash,
    _apply_progress_to_records,
    _hash_one,
    _task_key,
)

_IMAGE = MODEL_REFERENCE_CATEGORY.image_generation


def _download(file_name: str, purpose: str | None = None) -> DownloadRecord:
    return DownloadRecord(file_name=file_name, file_url=f"https://example.invalid/{file_name}", file_purpose=purpose)


def _record(name: str, downloads: list[DownloadRecord]) -> GenericModelRecord:
    return GenericModelRecord(
        record_type="image_generation",
        name=name,
        model_classification=ModelClassification(domain="image", purpose="generation"),
        config=GenericModelRecordConfig(download=downloads),
    )


def _write_standalone_vae(path: Path) -> None:
    tensors = [
        ("decoder.conv_in.weight", "F16", [2, 2], bytes(range(1, 9))),
        ("encoder.conv_out.weight", "F32", [1, 2], bytes(range(16, 24))),
    ]
    header: dict[str, object] = {}
    buffer = bytearray()
    for tensor_name, dtype, shape, data in tensors:
        begin = len(buffer)
        buffer += data
        header[tensor_name] = {"dtype": dtype, "shape": shape, "data_offsets": [begin, len(buffer)]}
    header_json = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_json)) + header_json + bytes(buffer))


def test_hash_progress_round_trips_with_timestamp(tmp_path: Path) -> None:
    """A recorded result persists to disk and reloads with its content hash, status, and a timestamp."""
    progress_path = tmp_path / "hash_progress.json"
    progress = HashProgress.load(progress_path)
    assert progress.results == {}

    progress.record(
        "key-1",
        category="image_generation",
        name="Mono",
        kind="vae",
        embedded=True,
        content_hash="a" * 64,
        status="hashed",
        reason=None,
        source="local",
    )
    assert progress_path.exists()

    reloaded = HashProgress.load(progress_path)
    entry = reloaded.get("key-1")
    assert entry is not None
    assert entry["content_hash"] == "a" * 64
    assert entry["status"] == "hashed"
    assert entry["at"]


def test_corrupt_progress_file_starts_fresh(tmp_path: Path) -> None:
    """An unreadable progress file is treated as empty rather than crashing the pass."""
    progress_path = tmp_path / "hash_progress.json"
    progress_path.write_text("{ not json", encoding="utf-8")
    assert HashProgress.load(progress_path).results == {}


def test_task_key_is_stable_and_distinct() -> None:
    """The task key is deterministic and distinguishes kind and embedded/standalone form."""
    embedded_vae = _task_key(_IMAGE, "M", ComponentKind.VAE, embedded=True)
    standalone_vae = _task_key(_IMAGE, "M", ComponentKind.VAE, embedded=False)
    embedded_te = _task_key(_IMAGE, "M", ComponentKind.TEXT_ENCODERS, embedded=True)
    assert len({embedded_vae, standalone_vae, embedded_te}) == 3
    assert embedded_vae == _task_key(_IMAGE, "M", ComponentKind.VAE, embedded=True)


def test_apply_progress_populates_records_and_reports_touched() -> None:
    """Prior hashes are re-applied to the embedded map and the standalone download, and names are reported."""
    monolithic = _record("Mono", [_download("mono.safetensors")])
    split_file = _record("Split", [_download("ae.safetensors", "vae")])
    records = {_IMAGE: {"Mono": monolithic, "Split": split_file}}

    progress = HashProgress(path=Path("unused"))
    progress.results = {
        "k-embedded": {
            "category": "image_generation",
            "name": "Mono",
            "kind": "vae",
            "embedded": True,
            "content_hash": "a" * 64,
            "status": "hashed",
        },
        "k-standalone": {
            "category": "image_generation",
            "name": "Split",
            "kind": "vae",
            "embedded": False,
            "content_hash": "b" * 64,
            "status": "hashed",
        },
        "k-failed": {
            "category": "image_generation",
            "name": "Mono",
            "kind": "text_encoders",
            "embedded": True,
            "content_hash": None,
            "status": "failed",
        },
    }

    touched = _apply_progress_to_records(records, progress)

    assert monolithic.config.embedded_component_hashes == {"vae": "a" * 64}
    assert split_file.config.download[0].content_hash == "b" * 64
    assert touched[_IMAGE] == {"Mono", "Split"}


def _planned(task: ComponentHashTask, *, local_path: Path | None, on_disk: bool) -> PlannedHash:
    return PlannedHash(
        category=_IMAGE,
        name="M",
        record=_record("M", [_download("ae.safetensors", "vae")]),
        task=task,
        key="k",
        local_path=local_path,
        on_disk=on_disk,
    )


def test_hash_one_reads_a_present_local_file(tmp_path: Path) -> None:
    """With an on-disk file and local source, the component is hashed from disk (no network)."""
    vae_file = tmp_path / "ae.safetensors"
    _write_standalone_vae(vae_file)
    task = ComponentHashTask(0, "https://example.invalid/ae.safetensors", ComponentKind.VAE, embedded=False)

    content_hash, status, reason, source = _hash_one(
        _planned(task, local_path=vae_file, on_disk=True),
        source="local",
        session=requests.Session(),
    )

    assert status == "hashed"
    assert source == "local"
    assert reason is None
    assert content_hash == hash_standalone_component_file(vae_file)


def test_hash_one_local_source_skips_absent_file() -> None:
    """A component not on disk under the local source is reported absent, not fetched."""
    task = ComponentHashTask(0, "https://example.invalid/ae.safetensors", ComponentKind.VAE, embedded=False)

    content_hash, status, _reason, source = _hash_one(
        _planned(task, local_path=None, on_disk=False),
        source="local",
        session=requests.Session(),
    )

    assert content_hash is None
    assert status == "absent"
    assert source == "local"
