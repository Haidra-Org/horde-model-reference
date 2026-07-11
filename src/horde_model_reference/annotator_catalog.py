"""The catalog of ControlNet *annotator* checkpoint files the horde fetches, as torch-free reference data.

ControlNet preprocessing in the worker is done by the ``comfyui_controlnet_aux`` package (pinned in hordelib).
Its detectors download their weights from the HuggingFace hub on first use via that package's ``custom_hf_download``
helper, into ``<AUX_ANNOTATOR_CKPTS_PATH>/<repo>/<subfolder>/<filename>``. Those downloads are otherwise opaque
to the rest of the ecosystem: the model reference does not list them, so the gated R2 mirror and the worker's
download accounting never saw them.

This module makes that set *known*: a single, verified list of the annotator files the horde-exposed
preprocessors actually pull, each with the exact repo / subfolder / filename the node expects on disk (so a
pre-placed file is found and the node skips its own HuggingFace download) and the origin URL to fetch it from.
It deliberately carries no torch/transformers/hordelib dependency so the orchestrator, the devops upload tool,
and hordelib can all share it.

Scope: files reachable through ``custom_hf_download`` for the Horde-exposed preprocessors. Detectors that load
complete repositories through HuggingFace ``transformers`` are intentionally excluded because they use the hub
cache rather than the annotator checkpoint layout. Weightless OpenCV/numpy detectors likewise need no records.

The sha256 of each file starts as None ("not yet known"): the devops upload tool computes the real hashes and
backfills them here, after which the gated content-addressed mirror can serve the file. Until then the file is
fetched straight from its HuggingFace origin, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "ANNOTATOR_FILES",
    "ANNOTATOR_HF_REPO",
    "AnnotatorFile",
    "annotators_for_control_types",
]

ANNOTATOR_HF_REPO = "lllyasviel/Annotators"
"""The HuggingFace repo hosting the classic ControlNet annotator checkpoints."""

_HF_RESOLVE_BASE = "https://huggingface.co"
"""Base of the HuggingFace ``resolve`` URL a file is downloaded from (``/<repo>/resolve/main/<path>``)."""


@dataclass(frozen=True)
class AnnotatorFile:
    """One annotator checkpoint file, located exactly the way ``custom_hf_download`` locates it.

    The on-disk path (relative to the annotator checkpoints directory) and the origin URL are both derived from
    ``repo`` / ``subfolder`` / ``filename`` so they cannot drift from each other.
    """

    repo: str
    """The HuggingFace repo id the file lives in (e.g. ``lllyasviel/Annotators``)."""
    filename: str
    """The file name within the repo (and ``subfolder``)."""
    subfolder: str = ""
    """The subfolder within the repo, if any (matches ``custom_hf_download``'s ``subfolder`` argument)."""
    sha256: str | None = None
    """The file's sha256, or None until the upload tool computes and backfills it (enables the gated mirror)."""
    control_types: tuple[str, ...] = ()
    """The horde control type(s) this file serves (informational; e.g. ``("depth",)``)."""
    preprocessors: tuple[str, ...] = ()
    """The ``comfyui_controlnet_aux`` node class(es) that load this file (used to gate lean-install prefetch)."""

    @property
    def relative_path(self) -> str:
        """Return the file's path relative to the annotator checkpoints dir: ``<repo>/<subfolder>/<filename>``.

        This mirrors ``custom_hf_download`` exactly, so a file written here is found by the detector and its own
        HuggingFace download is skipped. Always uses forward slashes (a POSIX-style relative path).
        """
        parts = [self.repo, *(part for part in self.subfolder.split("/") if part), self.filename]
        return "/".join(parts)

    @property
    def origin_url(self) -> str:
        """Return the HuggingFace ``resolve`` URL the file is downloaded from when not mirrored."""
        in_repo = "/".join(part for part in (self.subfolder, self.filename) if part)
        return f"{_HF_RESOLVE_BASE}/{self.repo}/resolve/main/{in_repo}"


# The verified set, read directly from the pinned comfyui_controlnet_aux detectors' ``from_pretrained`` defaults.
# This catalog is the worker/HIU-side authority for which annotator weights exist and where they live on disk;
# hordelib's preprocessor map is one downstream consumer of it, not the source of truth. Revisit this list on an
# annotator pin bump, when a preprocessor's checkpoint default can change.
ANNOTATOR_FILES: tuple[AnnotatorFile, ...] = (
    # HED edge detector: serves "hed" and (re-used by) the fake-scribble preprocessor.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="ControlNetHED.pth",
        sha256="5ca93762ffd68a29fee1af9d495bf6aab80ae86f08905fb35472a083a4c7a8fa",
        control_types=("hed", "fakescribbles"),
        preprocessors=("HEDPreprocessor", "FakeScribblePreprocessor"),
    ),
    # LeReS depth: two files (the depth model and its pix2pix refiner), both loaded by from_pretrained.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="res101.pth",
        sha256="1d696b2ef3e8336b057d0c15bc82d2fecef821bfebe5ef9d7671a5ec5dde520b",
        control_types=("depth",),
        preprocessors=("LeReS-DepthMapPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="latest_net_G.pth",
        sha256="50ec735d74ed6499562d898f41b49343e521808b8dae589aa3c2f5c9ac9f7462",
        control_types=("depth",),
        preprocessors=("LeReS-DepthMapPreprocessor",),
    ),
    # OpenPose: body, hand and face pose estimators, all from lllyasviel/Annotators on the default path.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="body_pose_model.pth",
        sha256="25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="hand_pose_model.pth",
        sha256="b76b00d1750901abd07b9f9d8c98cc3385b8fe834a26d4b4f0aad439e75fc600",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="facenet.pth",
        sha256="8beb52e548624ffcc4aed12af7aee7dcbfaeea420c75609fee999fe7add79d43",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    # UniFormer semantic segmentation: serves "seg" (the SemSegPreprocessor alias).
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="upernet_global_small.pth",
        sha256="bebfa1264c10381e389d8065056baaadbdadee8ddc6e36770d1ec339dc84d970",
        control_types=("seg",),
        preprocessors=("SemSegPreprocessor",),
    ),
    # M-LSD straight-line detector: serves "mlsd" and the "hough" backward-compatibility alias.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="mlsd_large_512_fp32.pth",
        sha256="5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082",
        control_types=("mlsd", "hough"),
        preprocessors=("M-LSDPreprocessor",),
    ),
    # Realistic lineart loads both fine and coarse-mode weights; the coarse mode is a detector parameter,
    # not a separate control type.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="sk_model.pth",
        sha256="c686ced2a666b4850b4bb6ccf0748031c3eda9f822de73a34b8979970d90f0c6",
        control_types=("lineart",),
        preprocessors=("LineArtPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="sk_model2.pth",
        sha256="30a534781061f34e83bb9406b4335da4ff2616c95d22a585c1245aa8363e74e0",
        control_types=("lineart",),
        preprocessors=("LineArtPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="netG.pth",
        sha256="ccabdcc3f5cf3c07cf65d58776acb21df7dfda825cdc70c9766a93fd62bfc488",
        control_types=("lineart_anime",),
        preprocessors=("AnimeLineArtPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="erika.pth",
        sha256="badbd6baf013cefbd98993307b02cc14a26c770d067416e4fdecc8720b88feeb",
        control_types=("lineart_anime_denoise",),
        preprocessors=("Manga2Anime_LineArt_Preprocessor",),
    ),
    # PiDiNet serves both soft-edge and scribble renderings from the same checkpoint.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="table5_pidinet.pth",
        sha256="80860ac267258b5f27486e0ef152a211d0b08120f62aeb185a050acc30da486c",
        control_types=("pidinet", "scribble_pidinet"),
        preprocessors=("PiDiNetPreprocessor", "Scribble_PiDiNet_Preprocessor"),
    ),
    AnnotatorFile(
        repo="bdsqlsz/qinglong_controlnet-lllite",
        subfolder="Annotators",
        filename="7_model.pth",
        sha256="b9037964149c55156c6adbffdfbd7e8ca7d2ef2a4d90573520efa7f3a1aacf06",
        control_types=("teed",),
        preprocessors=("TEEDPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="scannet.pt",
        sha256="03dbf1600c51ee3d45c29f77b77bf1a3b7a24c3452dba62a4ae658f37330c209",
        control_types=("normal_bae",),
        preprocessors=("BAE-NormalMapPreprocessor",),
    ),
    AnnotatorFile(
        repo="depth-anything/Depth-Anything-V2-Large",
        filename="depth_anything_v2_vitl.pth",
        sha256="a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
        control_types=("depth_anything_v2",),
        preprocessors=("DepthAnythingV2Preprocessor",),
    ),
)


def annotators_for_control_types(control_types: Iterable[str]) -> tuple[AnnotatorFile, ...]:
    """Return the annotator files needed to run the given horde control types (deduplicated, stable order).

    A control type whose preprocessor loads no weights (``canny``, ``scribble``) contributes nothing, so a
    selection of only those yields an empty tuple: there is genuinely nothing to fetch for them, which a
    caller must not mistake for "undeterminable". A control type unknown to the catalog likewise contributes
    nothing (its annotator, if any, is not horde-exposed). Files are returned in :data:`ANNOTATOR_FILES`
    order and each appears once even when several requested control types share it (e.g. ``mlsd``/``hough``).
    """
    wanted = set(control_types)
    return tuple(entry for entry in ANNOTATOR_FILES if wanted.intersection(entry.control_types))
