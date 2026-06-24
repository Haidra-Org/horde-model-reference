"""Tests for the torch-free on-disk model layout module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from horde_model_reference.annotator_catalog import annotators_for_control_types
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
    PresenceSummary,
    annotator_file_path,
    annotator_root,
    annotators_present,
    annotators_present_for_control_types,
    category_folder,
    component_relative_path,
    file_paths_for,
    free_bytes_for,
    is_present,
    presence_summary,
    resolve_weights_root,
)


def _record(
    category: MODEL_REFERENCE_CATEGORY,
    downloads: list[DownloadRecord],
    *,
    name: str = "m",
) -> GenericModelRecord:
    return GenericModelRecord(
        name=name,
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


def test_presence_summary_partitions_present_missing_and_unknown(tmp_path: Path) -> None:
    """presence_summary buckets each requested name by on-disk presence and reference membership."""
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    (folder / "here.bin").write_bytes(b"x")
    reference = {
        "Present": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("here.bin")], name="Present"),
        "Missing": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("absent.bin")], name="Missing"),
    }

    summary = presence_summary(reference, ["Present", "Missing", "NotInReference"], tmp_path)

    assert summary == PresenceSummary(present=("Present",), missing=("Missing",), unknown=("NotInReference",))
    assert summary.num_present == 1
    assert summary.num_requested == 3


def test_presence_summary_preserves_input_order_within_buckets(tmp_path: Path) -> None:
    """Each bucket keeps the order in which the requested names were supplied."""
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    for name in ("a.bin", "b.bin"):
        (folder / name).write_bytes(b"x")
    reference = {
        "A": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin")], name="A"),
        "B": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("b.bin")], name="B"),
        "Gone": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("gone.bin")], name="Gone"),
    }

    summary = presence_summary(reference, ["B", "A", "Gone"], tmp_path)

    assert summary.present == ("B", "A")
    assert summary.missing == ("Gone",)


def test_presence_summary_honours_extra_roots(tmp_path: Path) -> None:
    """A model present only in an extra root is reported present, mirroring is_present."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    (extra / "miscellaneous").mkdir(parents=True)
    (extra / "miscellaneous" / "a.bin").write_bytes(b"x")
    reference = {"A": _record(MODEL_REFERENCE_CATEGORY.miscellaneous, [_download("a.bin")], name="A")}

    assert presence_summary(reference, ["A"], primary).missing == ("A",)
    assert presence_summary(reference, ["A"], primary, extra_roots=[extra]).present == ("A",)


def test_presence_summary_empty_request_is_empty(tmp_path: Path) -> None:
    """No requested names yields empty buckets and a zero count."""
    summary = presence_summary({}, [], tmp_path)
    assert summary == PresenceSummary(present=(), missing=(), unknown=())
    assert summary.num_requested == 0


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


def _place_annotator(root: Path, relative_path: str) -> Path:
    """Write a stub annotator file at hordelib's expected on-disk location under *root*, returning its path."""
    target = annotator_root(root) / Path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"weights")
    return target


def test_annotator_root_mirrors_hordelib_ckpts_dir(tmp_path: Path) -> None:
    """The annotator dir is ``<weights_root>/controlnet/annotators`` (hordelib's AUX_ANNOTATOR_CKPTS_PATH fallback)."""
    assert annotator_root(tmp_path) == tmp_path / "controlnet" / "annotators"


def test_annotator_file_path_matches_custom_hf_download_layout(tmp_path: Path) -> None:
    """A file resolves to ``<root>/controlnet/annotators/<repo>/<subfolder>/<filename>`` (what the node expects)."""
    hed = next(entry for entry in annotators_for_control_types(["hed"]))
    assert annotator_file_path(hed, tmp_path) == (
        tmp_path / "controlnet" / "annotators" / "lllyasviel" / "Annotators" / "ControlNetHED.pth"
    )


def test_annotator_file_path_prefers_an_existing_copy_in_an_extra_root(tmp_path: Path) -> None:
    """When a copy lives only in an extra root, that copy is returned; absent everywhere, the primary target is."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    hed = next(entry for entry in annotators_for_control_types(["hed"]))
    existing = _place_annotator(extra, hed.relative_path)
    assert annotator_file_path(hed, primary, extra_roots=[extra]) == existing
    # With no copy anywhere, resolution targets the primary root (deterministic download destination).
    assert annotator_file_path(hed, primary) == annotator_root(primary) / Path(hed.relative_path)


def test_annotators_present_is_true_only_when_every_file_exists(tmp_path: Path) -> None:
    """Presence needs every file: a multi-file detector (LeReS depth) is absent until both files are placed."""
    depth_files = annotators_for_control_types(["depth"])
    assert annotators_present(depth_files, tmp_path) is False
    _place_annotator(tmp_path, depth_files[0].relative_path)
    assert annotators_present(depth_files, tmp_path) is False  # one of two present is not enough
    _place_annotator(tmp_path, depth_files[1].relative_path)
    assert annotators_present(depth_files, tmp_path) is True


def test_annotators_present_empty_is_vacuously_true(tmp_path: Path) -> None:
    """Requiring no files is satisfied even on empty disk: there is nothing that could be missing."""
    assert annotators_present([], tmp_path) is True


def test_annotators_present_searches_extra_roots(tmp_path: Path) -> None:
    """A file living only in an extra root still counts as present, mirroring is_present's multi-root search."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    hed = annotators_for_control_types(["hed"])
    _place_annotator(extra, hed[0].relative_path)
    assert annotators_present(hed, primary) is False
    assert annotators_present(hed, primary, extra_roots=[extra]) is True


def test_weightless_control_types_are_present_on_empty_disk(tmp_path: Path) -> None:
    """canny/scribble need no annotator files, so they are 'present' even with nothing on disk (never block)."""
    assert annotators_present_for_control_types(["canny", "scribble"], tmp_path) is True


def test_control_type_presence_requires_its_files_on_disk(tmp_path: Path) -> None:
    """A weighted control type is absent until its files land; placing them flips it present."""
    assert annotators_present_for_control_types(["mlsd"], tmp_path) is False
    mlsd = annotators_for_control_types(["mlsd"])
    _place_annotator(tmp_path, mlsd[0].relative_path)
    assert annotators_present_for_control_types(["mlsd"], tmp_path) is True
    # The "hough" alias resolves to the same file, so it is now present too without a second download.
    assert annotators_present_for_control_types(["hough"], tmp_path) is True


def test_mixed_selection_is_absent_until_the_weighted_member_is_present(tmp_path: Path) -> None:
    """A weightless + weighted mix is blocked by the weighted member alone (the weightless one never contributes)."""
    assert annotators_present_for_control_types(["canny", "hed"], tmp_path) is False
    hed = annotators_for_control_types(["hed"])
    _place_annotator(tmp_path, hed[0].relative_path)
    assert annotators_present_for_control_types(["canny", "hed"], tmp_path) is True
