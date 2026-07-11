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
    annotators_for_control_types,
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
    "sk_model.pth",
    "sk_model2.pth",
    "netG.pth",
    "erika.pth",
    "table5_pidinet.pth",
    "7_model.pth",
    "scannet.pt",
    "depth_anything_v2_vitl.pth",
}


def test_catalog_lists_exactly_the_known_annotator_files() -> None:
    """Verify the catalog is exactly the known custom_hf_download file set (no more, no fewer, no duplicates)."""
    filenames = [entry.filename for entry in ANNOTATOR_FILES]
    assert len(filenames) == len(set(filenames))  # no duplicates
    assert set(filenames) == _EXPECTED_FILENAMES


def test_classic_files_use_the_annotators_repo_root() -> None:
    """Verify classic annotator files retain the lllyasviel/Annotators root layout."""
    classic_entries = [entry for entry in ANNOTATOR_FILES if entry.repo == ANNOTATOR_HF_REPO]
    assert classic_entries
    assert all(entry.subfolder == "" for entry in classic_entries)


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


def test_every_file_has_a_backfilled_sha256() -> None:
    """Verify every file carries a real (64-hex lowercase) sha256, so the gated mirror can serve it.

    The hashes were computed from the pinned ``lllyasviel/Annotators`` files and backfilled into the catalog;
    a file with no hash could not be content-addressed on the mirror.
    """
    for entry in ANNOTATOR_FILES:
        assert entry.sha256 is not None, entry.filename
        assert len(entry.sha256) == 64 and entry.sha256 == entry.sha256.lower(), entry.filename
        int(entry.sha256, 16)  # valid hex


def test_control_types_cover_every_weighted_preprocessor() -> None:
    """Verify the catalog's control types cover all horde control types that actually load annotator weights."""
    covered = {control_type for entry in ANNOTATOR_FILES for control_type in entry.control_types}
    assert {
        "hed",
        "fakescribbles",
        "depth",
        "openpose",
        "seg",
        "mlsd",
        "hough",
        "lineart",
        "lineart_anime",
        "lineart_anime_denoise",
        "pidinet",
        "scribble_pidinet",
        "teed",
        "normal_bae",
        "depth_anything_v2",
    } <= covered


def test_selecting_depth_returns_both_leres_files() -> None:
    """A weighted control type that needs several files returns all of them (LeReS depth needs two)."""
    files = annotators_for_control_types(["depth"])
    assert {entry.filename for entry in files} == {"res101.pth", "latest_net_G.pth"}


def test_selecting_openpose_returns_its_three_estimators() -> None:
    """OpenPose pulls its body, hand and face estimators together."""
    files = annotators_for_control_types(["openpose"])
    assert {entry.filename for entry in files} == {"body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"}


def test_weightless_control_types_need_no_files() -> None:
    """canny/scribble load no weights, so they must select nothing (not be mistaken for 'undeterminable')."""
    assert annotators_for_control_types(["canny"]) == ()
    assert annotators_for_control_types(["canny", "scribble"]) == ()


def test_empty_selection_is_empty() -> None:
    """No requested control types selects no files."""
    assert annotators_for_control_types([]) == ()


def test_unknown_control_type_contributes_nothing() -> None:
    """A control type the catalog does not know about pulls no file (it is not horde-exposed)."""
    assert annotators_for_control_types(["banana"]) == ()


def test_mixed_selection_keeps_only_the_weighted_known_types() -> None:
    """A mix of weighted, weightless and unknown types yields exactly the weighted-known files."""
    files = annotators_for_control_types(["hed", "canny", "banana"])
    assert [entry.filename for entry in files] == ["ControlNetHED.pth"]


def test_alias_control_types_dedupe_to_one_file() -> None:
    """The mlsd and hough aliases share one detector, so requesting both yields a single (un-duplicated) file."""
    files = annotators_for_control_types(["mlsd", "hough"])
    assert [entry.filename for entry in files] == ["mlsd_large_512_fp32.pth"]


def test_selection_order_follows_the_catalog_not_the_request() -> None:
    """Output order is the stable catalog order regardless of how the request was ordered."""
    forward = annotators_for_control_types(["hed", "depth"])
    reversed_request = annotators_for_control_types(["depth", "hed"])
    assert forward == reversed_request
    assert [entry.filename for entry in forward] == ["ControlNetHED.pth", "res101.pth", "latest_net_G.pth"]
