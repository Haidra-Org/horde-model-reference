"""Tests for the symlink-safe on-disk layout migration tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE
from horde_model_reference.cli.migrate_layout import (
    apply_layout_migration,
    plan_layout_migration,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

_UNET = DownloadRecord(file_name="unet.safetensors", file_url="https://example/unet")


def _vae(file_name: str = "ae.safetensors") -> DownloadRecord:
    """Return a VAE component download record (routed to the sibling ``vae`` folder)."""
    return DownloadRecord(file_name=file_name, file_url="https://example/ae", file_purpose="vae")


def _image_record(name: str, downloads: list[DownloadRecord]) -> ImageGenerationModelRecord:
    """Build a minimal image-generation record carrying the given download entries."""
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=False,
        config=GenericModelRecordConfig(download=downloads),
    )


def _make_root(tmp_path: Path) -> Path:
    """Create a weights root with an empty ``compvis`` category folder and return it."""
    (tmp_path / "compvis").mkdir()
    return tmp_path


def test_noop_when_component_already_in_sibling(tmp_path: Path) -> None:
    """A component already in its sibling folder needs no migration."""
    root = _make_root(tmp_path)
    (root / "compvis" / "unet.safetensors").write_bytes(b"u")
    (root / "vae").mkdir()
    (root / "vae" / "ae.safetensors").write_bytes(b"v")

    plan = plan_layout_migration({"M": _image_record("M", [_UNET, _vae()])}, root)
    assert plan.is_noop is True
    assert plan.skipped_symlinks == []


def test_non_component_file_never_moves(tmp_path: Path) -> None:
    """A plain checkpoint (no routed purpose) is already canonical and is never moved."""
    root = _make_root(tmp_path)
    (root / "compvis" / "unet.safetensors").write_bytes(b"u")

    plan = plan_layout_migration({"M": _image_record("M", [_UNET])}, root)
    assert plan.is_noop is True


def test_flat_component_is_planned_and_moved(tmp_path: Path) -> None:
    """A component sitting flat in the category folder is planned and moved to its sibling folder."""
    root = _make_root(tmp_path)
    flat_vae = root / "compvis" / "ae.safetensors"
    flat_vae.write_bytes(b"v")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert len(plan.moves) == 1
    move = plan.moves[0]
    assert move.source == flat_vae
    assert move.destination == root / "vae" / "ae.safetensors"

    applied = apply_layout_migration(plan)
    assert len(applied) == 1
    assert not flat_vae.exists()
    assert (root / "vae" / "ae.safetensors").read_bytes() == b"v"


def test_sidecars_travel_with_the_file(tmp_path: Path) -> None:
    """A moved component takes its ``.sha256`` sidecar along to the sibling folder."""
    root = _make_root(tmp_path)
    (root / "compvis" / "ae.safetensors").write_bytes(b"v")
    (root / "compvis" / "ae.sha256").write_text("deadbeef *ae.safetensors")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert plan.moves[0].sidecars == (
        (root / "compvis" / "ae.sha256", root / "vae" / "ae.sha256"),
    )

    apply_layout_migration(plan)
    assert not (root / "compvis" / "ae.sha256").exists()
    assert (root / "vae" / "ae.sha256").read_text() == "deadbeef *ae.safetensors"


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running the migration after applying it is a no-op."""
    root = _make_root(tmp_path)
    (root / "compvis" / "ae.safetensors").write_bytes(b"v")
    reference = {"M": _image_record("M", [_vae()])}

    apply_layout_migration(plan_layout_migration(reference, root))
    second_plan = plan_layout_migration(reference, root)
    assert second_plan.is_noop is True


def test_never_clobbers_existing_canonical(tmp_path: Path) -> None:
    """When the canonical file already exists, the flat copy is left and nothing is overwritten."""
    root = _make_root(tmp_path)
    (root / "compvis" / "ae.safetensors").write_bytes(b"flat")
    (root / "vae").mkdir()
    (root / "vae" / "ae.safetensors").write_bytes(b"canonical")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert plan.is_noop is True
    # The canonical file is left untouched; nothing is overwritten.
    assert (root / "vae" / "ae.safetensors").read_bytes() == b"canonical"


def test_dry_run_does_not_move(tmp_path: Path) -> None:
    """Planning alone moves nothing; the flat file stays until the plan is applied."""
    root = _make_root(tmp_path)
    flat_vae = root / "compvis" / "ae.safetensors"
    flat_vae.write_bytes(b"v")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert plan.moves  # a move is planned
    # ...but planning alone moves nothing.
    assert flat_vae.exists()
    assert not (root / "vae" / "ae.safetensors").exists()


def test_symlinked_source_is_skipped(tmp_path: Path) -> None:
    """A symlinked source file is reported and never moved."""
    root = _make_root(tmp_path)
    real = tmp_path / "elsewhere" / "ae.safetensors"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"v")
    flat_vae = root / "compvis" / "ae.safetensors"
    try:
        flat_vae.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert plan.moves == []
    assert flat_vae in plan.skipped_symlinks


def test_symlinked_ancestor_is_skipped(tmp_path: Path) -> None:
    """A file under a symlinked category folder is reported and never moved."""
    root = _make_root(tmp_path)
    real_dir = tmp_path / "real_compvis"
    real_dir.mkdir()
    (real_dir / "ae.safetensors").write_bytes(b"v")

    # Replace the category folder with a symlink to the real directory.
    (root / "compvis").rmdir()
    try:
        (root / "compvis").symlink_to(real_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    plan = plan_layout_migration({"M": _image_record("M", [_vae()])}, root)
    assert plan.moves == []
    assert (root / "compvis" / "ae.safetensors") in plan.skipped_symlinks
