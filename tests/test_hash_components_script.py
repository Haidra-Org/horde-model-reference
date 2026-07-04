"""Tests for the offline hashing pass's pure planning logic (no network I/O)."""

from __future__ import annotations

from horde_model_reference import ModelClassification
from horde_model_reference.component_hash import ComponentKind
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from scripts.hash_components import plan_component_hash_tasks


def _download(file_name: str, purpose: str | None = None, content_hash: str | None = None) -> DownloadRecord:
    return DownloadRecord(
        file_name=file_name,
        file_url=f"https://example.invalid/{file_name}",
        file_purpose=purpose,
        content_hash=content_hash,
    )


def _record(downloads: list[DownloadRecord], embedded: dict[str, str] | None = None) -> GenericModelRecord:
    return GenericModelRecord(
        record_type="image_generation",
        name="unused",
        model_classification=ModelClassification(domain="image", purpose="generation"),
        config=GenericModelRecordConfig(download=downloads, embedded_component_hashes=embedded),
    )


def test_split_file_plans_standalone_not_embedded() -> None:
    """A split-file model hashes its standalone VAE and text encoder, and skips embedded extraction."""
    record = _record(
        [
            _download("unet.safetensors", "unet"),
            _download("ae.safetensors", "vae"),
            _download("te.safetensors", "text_encoders"),
        ],
    )
    tasks = plan_component_hash_tasks(record, skip_existing=False)
    assert not any(task.embedded for task in tasks)
    assert {(task.kind, task.download_index) for task in tasks} == {
        (ComponentKind.VAE, 1),
        (ComponentKind.TEXT_ENCODERS, 2),
    }


def test_monolithic_plans_embedded_vae() -> None:
    """A monolithic checkpoint (no separate VAE) is probed for an embedded VAE."""
    record = _record([_download("sdxl_finetune.safetensors")])
    tasks = plan_component_hash_tasks(record, skip_existing=False)
    assert len(tasks) == 1
    assert tasks[0].embedded is True
    assert tasks[0].kind is ComponentKind.VAE


def test_non_safetensors_is_skipped() -> None:
    """A pickle checkpoint yields no tasks (it cannot be hashed torch-free)."""
    record = _record([_download("legacy.ckpt")])
    assert plan_component_hash_tasks(record, skip_existing=False) == []


def test_skip_existing_standalone_hash() -> None:
    """skip_existing leaves an already-hashed standalone component alone."""
    record = _record([_download("ae.safetensors", "vae", content_hash="d" * 64)])
    assert plan_component_hash_tasks(record, skip_existing=True) == []
    assert len(plan_component_hash_tasks(record, skip_existing=False)) == 1


def test_skip_existing_embedded_vae() -> None:
    """skip_existing leaves an already-recorded embedded VAE alone."""
    record = _record([_download("sdxl_finetune.safetensors")], embedded={"vae": "e" * 64})
    assert plan_component_hash_tasks(record, skip_existing=True) == []
