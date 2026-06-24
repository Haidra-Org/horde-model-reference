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

Scope: only the files reachable through ``custom_hf_download`` for the preprocessors in
``CONTROLNET_IMAGE_PREPROCESSOR_MAP``. The ``normal`` map (MiDaS) is intentionally excluded: it loads via
HuggingFace ``transformers`` from ``Intel/dpt-hybrid-midas`` into the transformers cache, a different mechanism
hordelib already handles separately. ``canny`` and ``scribble`` need no weights at all.

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
"""The HuggingFace repo every horde-exposed ``custom_hf_download`` annotator file is served from."""

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
# Every horde-exposed preprocessor that loads weights through ``custom_hf_download`` resolves to lllyasviel/Annotators
# with an empty subfolder. Keep this list in step with CONTROLNET_IMAGE_PREPROCESSOR_MAP on an annotator pin bump.
ANNOTATOR_FILES: tuple[AnnotatorFile, ...] = (
    # HED edge detector: serves "hed" and (re-used by) the fake-scribble preprocessor.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="ControlNetHED.pth",
        control_types=("hed", "fakescribbles"),
        preprocessors=("HEDPreprocessor", "FakeScribblePreprocessor"),
    ),
    # LeReS depth: two files (the depth model and its pix2pix refiner), both loaded by from_pretrained.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="res101.pth",
        control_types=("depth",),
        preprocessors=("LeReS-DepthMapPreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="latest_net_G.pth",
        control_types=("depth",),
        preprocessors=("LeReS-DepthMapPreprocessor",),
    ),
    # OpenPose: body, hand and face pose estimators, all from lllyasviel/Annotators on the default path.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="body_pose_model.pth",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="hand_pose_model.pth",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="facenet.pth",
        control_types=("openpose",),
        preprocessors=("OpenposePreprocessor",),
    ),
    # UniFormer semantic segmentation: serves "seg" (the SemSegPreprocessor alias).
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="upernet_global_small.pth",
        control_types=("seg",),
        preprocessors=("SemSegPreprocessor",),
    ),
    # M-LSD straight-line detector: serves "mlsd" and the "hough" backward-compatibility alias.
    AnnotatorFile(
        repo=ANNOTATOR_HF_REPO,
        filename="mlsd_large_512_fp32.pth",
        control_types=("mlsd", "hough"),
        preprocessors=("M-LSDPreprocessor",),
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
