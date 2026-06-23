"""Tests for the controlnet-annotator catalog: the exact on-disk paths and origin URLs must match the node.

These lock the derivation that lets a pre-placed file be found by ``comfyui_controlnet_aux``'s
``custom_hf_download`` (so it skips its own download). If the path or URL drifts, the node would re-download and
the whole unification silently no-ops, so the assertions are deliberately exact.
"""

from __future__ import annotations

from horde_model_reference.annotator_catalog import (
    ANNOTATOR_FILES,
    ANNOTATOR_HF_REPO,
    AnnotatorFile,
)

# The verified set of horde-exposed annotator files that load via custom_hf_download, read from the pinned
# comfyui_controlnet_aux detectors. canny/scribble need no weights; MiDaS (normal) loads via transformers.
_EXPECTED_FILENAMES = {
    "ControlNetHED.pth",
    "res101.pth",
    "latest_net_G.pth",
    "body_pose_model.pth",
    "hand_pose_model.pth",
    "facenet.pth",
    "upernet_global_small.pth",
    "mlsd_large_512_fp32.pth",
}


def test_catalog_lists_exactly_the_known_annotator_files() -> None:
    """Verify the catalog is exactly the known custom_hf_download file set (no more, no fewer, no duplicates)."""
    filenames = [entry.filename for entry in ANNOTATOR_FILES]
    assert len(filenames) == len(set(filenames))  # no duplicates
    assert set(filenames) == _EXPECTED_FILENAMES


def test_every_file_is_from_the_annotators_repo_with_no_subfolder() -> None:
    """Verify all horde-exposed annotators resolve to lllyasviel/Annotators at the repo root (subfolder '')."""
    for entry in ANNOTATOR_FILES:
        assert entry.repo == ANNOTATOR_HF_REPO == "lllyasviel/Annotators"
        assert entry.subfolder == ""


def test_relative_path_matches_custom_hf_download_layout() -> None:
    """Verify the on-disk relative path is ``<repo>/<filename>`` (what custom_hf_download writes/looks for)."""
    hed = next(entry for entry in ANNOTATOR_FILES if entry.filename == "ControlNetHED.pth")
    assert hed.relative_path == "lllyasviel/Annotators/ControlNetHED.pth"


def test_origin_url_is_the_huggingface_resolve_url() -> None:
    """Verify the origin URL is the HuggingFace ``resolve/main`` URL for the repo + file."""
    mlsd = next(entry for entry in ANNOTATOR_FILES if entry.filename == "mlsd_large_512_fp32.pth")
    assert mlsd.origin_url == "https://huggingface.co/lllyasviel/Annotators/resolve/main/mlsd_large_512_fp32.pth"


def test_subfolder_is_threaded_into_both_path_and_url() -> None:
    """Verify a subfolder (matching custom_hf_download's argument) appears in both the path and the URL."""
    entry = AnnotatorFile(repo="some/Repo", filename="w.pth", subfolder="annotator/ckpts")
    assert entry.relative_path == "some/Repo/annotator/ckpts/w.pth"
    assert entry.origin_url == "https://huggingface.co/some/Repo/resolve/main/annotator/ckpts/w.pth"


def test_hashes_start_unknown_until_backfilled() -> None:
    """Verify every file's sha256 starts None: the gated mirror only serves a file once its hash is backfilled."""
    assert all(entry.sha256 is None for entry in ANNOTATOR_FILES)


def test_control_types_cover_every_weighted_preprocessor() -> None:
    """Verify the catalog's control types cover all horde control types that actually load annotator weights."""
    covered = {control_type for entry in ANNOTATOR_FILES for control_type in entry.control_types}
    assert {"hed", "fakescribbles", "depth", "openpose", "seg", "mlsd", "hough"} <= covered
