"""Tests for the torch-free on-disk model layout module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
    get_weights_marker_folders,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from horde_model_reference.on_disk_layout import (
    category_folder,
    component_relative_path,
    file_paths_for,
    free_bytes_for,
    is_present,
    resolve_weights_root,
)


def _record(category: MODEL_REFERENCE_CATEGORY, downloads: list[DownloadRecord]) -> GenericModelRecord:
    return GenericModelRecord(
        name="m",
        record_type=category,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.miscellaneous),
        config=GenericModelRecordConfig(download=downloads),
    )


def _download(file_name: str, *, file_purpose: str | None = None) -> DownloadRecord:
    return DownloadRecord(file_name=file_name, file_url=f"https://example.com/{file_name}", file_purpose=file_purpose)


def test_category_folder_known_values() -> None:
    """Verify category_folder returns today's folder names, with image_generation mapping to compvis."""
    assert category_folder(MODEL_REFERENCE_CATEGORY.image_generation) == "compvis"
    assert category_folder(MODEL_REFERENCE_CATEGORY.controlnet) == "controlnet"
    assert category_folder(MODEL_REFERENCE_CATEGORY.miscellaneous) == "miscellaneous"
    assert category_folder(MODEL_REFERENCE_CATEGORY.clip) == "clip"


def test_category_folder_none_for_categories_without_a_folder() -> None:
    """Verify text/video/audio generation have no weights folder in this ecosystem."""
    assert category_folder(MODEL_REFERENCE_CATEGORY.text_generation) is None
    assert category_folder(MODEL_REFERENCE_CATEGORY.video_generation) is None
    assert category_folder(MODEL_REFERENCE_CATEGORY.audio_generation) is None


def test_category_folder_none_for_unknown_category() -> None:
    """Verify category_folder yields None for an unregistered category string."""
    assert category_folder("not_a_real_category") is None


def test_weights_markers_are_compvis_and_clip() -> None:
    """Verify the BFS markers derived from the registry are exactly compvis and clip."""
    assert set(get_weights_marker_folders()) == {"compvis", "clip"}


def test_component_relative_path_routes_components_to_sibling_folders() -> None:
    """Verify vae/text_encoder purposes redirect to sibling folders and others stay in place."""
    assert component_relative_path("model.safetensors", None) == Path("model.safetensors")
    assert component_relative_path("vae.safetensors", "vae") == Path("..") / "vae" / "vae.safetensors"
    assert (
        component_relative_path("te.safetensors", "text_encoders") == Path("..") / "text_encoders" / "te.safetensors"
    )
    assert component_relative_path("te.safetensors", "text_encoder") == Path("..") / "text_encoders" / "te.safetensors"
    assert component_relative_path("x.safetensors", "unet") == Path("x.safetensors")


def test_resolve_weights_root_finds_marker_tree_breadth_first(tmp_path: Path) -> None:
    """Verify resolve_weights_root descends to the directory holding both marker folders."""
    nested = tmp_path / "sub"
    (nested / "compvis").mkdir(parents=True)
    (nested / "clip").mkdir()
    assert resolve_weights_root(tmp_path) == nested


def test_resolve_weights_root_marker_at_base(tmp_path: Path) -> None:
    """Verify resolve_weights_root returns the base when it directly contains the markers."""
    (tmp_path / "compvis").mkdir()
    (tmp_path / "clip").mkdir()
    assert resolve_weights_root(tmp_path) == tmp_path


def test_resolve_weights_root_falls_back_to_base_when_no_markers(tmp_path: Path) -> None:
    """Verify resolve_weights_root returns the base directory when no marker tree exists."""
    assert resolve_weights_root(tmp_path) == tmp_path


def test_file_paths_for_resolves_primary_and_component_paths(tmp_path: Path) -> None:
    """Verify declared files resolve under the category folder, with components routed to siblings."""
    record = _record(
        MODEL_REFERENCE_CATEGORY.miscellaneous,
        [_download("model.safetensors"), _download("vae.safetensors", file_purpose="vae")],
    )
    paths = file_paths_for(record, tmp_path)
    assert paths == [
        tmp_path / "miscellaneous" / "model.safetensors",
        tmp_path / "vae" / "vae.safetensors",
    ]


def test_is_present_true_when_all_files_exist(tmp_path: Path) -> None:
    """Verify is_present is True only when every declared file exists on disk."""
    record = _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin"), _download("b.bin")])
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"a")
    assert is_present(record, tmp_path) is False
    (folder / "b.bin").write_bytes(b"b")
    assert is_present(record, tmp_path) is True


def test_is_present_false_for_record_without_resolvable_folder(tmp_path: Path) -> None:
    """Verify is_present is False when the record's category has no on-disk folder."""
    record = _record(MODEL_REFERENCE_CATEGORY.text_generation, [_download("a.bin")])
    assert is_present(record, tmp_path) is False


def test_is_present_searches_extra_roots(tmp_path: Path) -> None:
    """Verify a file living only in an extra root still counts as present."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (extra / "miscellaneous").mkdir(parents=True)
    (extra / "miscellaneous" / "a.bin").write_bytes(b"a")
    record = _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin")])
    assert is_present(record, primary) is False
    assert is_present(record, primary, extra_roots=[extra]) is True


def test_file_paths_for_prefers_existing_copy_in_extra_root(tmp_path: Path) -> None:
    """Verify file_paths_for returns the extra-root copy when the primary root lacks it."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (extra / "miscellaneous").mkdir(parents=True)
    existing = extra / "miscellaneous" / "a.bin"
    existing.write_bytes(b"a")
    record = _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin")])
    assert file_paths_for(record, primary, extra_roots=[extra]) == [existing]
    # With no existing copy anywhere, the primary-root target is returned.
    record_missing = _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("missing.bin")])
    assert file_paths_for(record_missing, primary, extra_roots=[extra]) == [
        primary / "miscellaneous" / "missing.bin",
    ]


def test_is_present_follows_symlinks(tmp_path: Path) -> None:
    """Verify a symlinked model file counts as present, while a broken symlink does not."""
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    real_target = tmp_path / "elsewhere" / "real.bin"
    real_target.parent.mkdir()
    real_target.write_bytes(b"weights")
    link = folder / "a.bin"
    try:
        link.symlink_to(real_target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    record = _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin")])
    assert is_present(record, tmp_path) is True

    real_target.unlink()
    assert is_present(record, tmp_path) is False


def test_free_bytes_for_returns_positive_value(tmp_path: Path) -> None:
    """Verify free_bytes_for reports a non-negative figure for an existing directory."""
    free = free_bytes_for(tmp_path)
    assert free is not None
    assert free >= 0


def test_resolve_weights_root_uses_env_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_weights_root reads AIWORKER_CACHE_HOME when no explicit root is given."""
    (tmp_path / "compvis").mkdir()
    (tmp_path / "clip").mkdir()
    monkeypatch.setenv("AIWORKER_CACHE_HOME", os.fspath(tmp_path))
    assert resolve_weights_root() == tmp_path
