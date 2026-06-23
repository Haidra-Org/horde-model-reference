"""Mirror the controlnet-annotator catalog to R2, alongside the model-reference categories.

The annotators (the ``comfyui_controlnet_aux`` checkpoints) are not model-reference records: they live in the
torch-free :mod:`horde_model_reference.annotator_catalog`, located on disk by repo/subfolder/filename rather than
in a category folder. So they get their own small planning pass here, reusing the same content-addressed
:class:`~scripts.r2_sync.object_store.ObjectStore`, the same opt-in allowlist, and the same
:class:`~scripts.r2_sync.planner.SyncItem` / :class:`~scripts.r2_sync.planner.SyncPlan` reporting as the category
sync, so the tool's output and exit semantics are uniform.

Because the catalog ships with each file's ``sha256`` unset (None) until it is backfilled, an annotator's content
address is computed here from the bytes and emitted as a :class:`~scripts.r2_sync.planner.HashCorrection` for a
maintainer to paste back into the catalog (mirroring the FIXME-hash backfill for records).

Allowlisting is by **repo** (e.g. ``lllyasviel/Annotators``): one HuggingFace repo is one upstream-licence
review unit, and every horde-exposed annotator currently comes from that single repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

from horde_model_reference.annotator_catalog import ANNOTATOR_FILES
from horde_model_reference.download_engine import download_file, sha256_of
from scripts.r2_sync.object_store import object_key_for
from scripts.r2_sync.planner import HashCorrection, SyncAction, SyncItem, SyncPlan

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from horde_model_reference.annotator_catalog import AnnotatorFile
    from scripts.r2_sync.allowlist import RedistributableAllowlist
    from scripts.r2_sync.object_store import ObjectStore

__all__ = ["ANNOTATOR_CATEGORY", "AnnotatorByteSource", "build_annotator_plan"]

ANNOTATOR_CATEGORY = "controlnet_annotator"
"""The reporting label used for annotator files (they are not a real model-reference category)."""

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_cache_name(*, source_id: str, display_name: str) -> str:
    """Return a cache filename that cannot be influenced by path separators in catalog data."""
    safe_display = _UNSAFE_NAME.sub("_", display_name).strip("._-") or "file"
    digest = sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{safe_display}"


@dataclass
class AnnotatorByteSource:
    """Supply an annotator file's bytes from the local checkpoint dir, else by fetching its HuggingFace origin."""

    ckpts_dir: Path | None
    """The local annotator checkpoints directory (``<repo>/<subfolder>/<filename>`` under it), if available."""
    cache_dir: Path | None
    """Where origin downloads are cached. When None, files not present locally are reported missing."""

    _origin_fetches: int = field(default=0, init=False)
    """How many origin downloads were performed (for reporting/tests)."""

    def acquire(self, entry: AnnotatorFile) -> Path | None:
        """Return a local path holding *entry*'s bytes, fetching from HuggingFace into the cache if needed."""
        if self.ckpts_dir is not None:
            local = self.ckpts_dir.joinpath(*entry.relative_path.split("/"))
            if local.is_file():
                return local
        if self.cache_dir is None:
            return None

        target = self.cache_dir / _safe_cache_name(source_id=entry.relative_path, display_name=entry.filename)
        if target.is_file():
            if entry.sha256 is not None and sha256_of(target).lower() != entry.sha256.lower():
                target.unlink()
            else:
                return target
        self._origin_fetches += 1
        outcome = download_file(entry.origin_url, target, expected_sha256=entry.sha256)
        if entry.sha256 is not None and target.is_file() and not outcome.success:
            target.unlink(missing_ok=True)
            Path(f"{target}.part").unlink(missing_ok=True)
        # Unknown-hash downloads are returned for backfill. Known-hash corrupt downloads are removed above so a
        # future run cannot trust poisoned cache bytes.
        return target if target.is_file() else None


def _plan_annotator_file(
    entry: AnnotatorFile,
    *,
    store: ObjectStore,
    byte_source: AnnotatorByteSource,
    allowlist: RedistributableAllowlist,
    apply: bool,
) -> tuple[SyncItem, HashCorrection | None]:
    """Decide and (in *apply* mode) perform the upload of one annotator file."""
    base = {"category": ANNOTATOR_CATEGORY, "model_name": entry.repo, "file_name": entry.filename}

    if entry.sha256 is not None:
        key = object_key_for(entry.sha256)
        if store.head(key):
            return SyncItem(**base, action=SyncAction.ALREADY_PRESENT, sha256=entry.sha256, key=key), None

    path = byte_source.acquire(entry)
    if path is None:
        return SyncItem(**base, action=SyncAction.MISSING_BYTES, sha256=entry.sha256), None

    actual = sha256_of(path)
    correction: HashCorrection | None = None
    if entry.sha256 is None:
        correction = HashCorrection(
            category=ANNOTATOR_CATEGORY,
            model_name=entry.repo,
            file_name=entry.filename,
            old_sha256="",
            new_sha256=actual,
        )
    elif actual.lower() != entry.sha256.lower():
        detail = f"declared {entry.sha256} but bytes hash to {actual}"
        return SyncItem(**base, action=SyncAction.HASH_MISMATCH, sha256=entry.sha256, detail=detail), None

    key = object_key_for(actual)
    if store.head(key):
        return SyncItem(**base, action=SyncAction.ALREADY_PRESENT, sha256=actual, key=key), correction

    if apply:
        metadata = {
            "category": ANNOTATOR_CATEGORY,
            "repo": entry.repo,
            "file_name": entry.filename,
            "source_url": entry.origin_url,
        }
        metadata.update(allowlist.metadata_for(entry.repo))
        store.put(key, path, metadata=metadata)
    return SyncItem(**base, action=SyncAction.UPLOAD, sha256=actual, key=key), correction


def build_annotator_plan(
    *,
    allowlist: RedistributableAllowlist,
    store: ObjectStore,
    byte_source: AnnotatorByteSource,
    apply: bool,
    catalog: Iterable[AnnotatorFile] = ANNOTATOR_FILES,
) -> SyncPlan:
    """Plan (and, when *apply*, perform) the R2 mirroring of every allowlisted annotator file.

    A file's *repo* must be cleared in the *allowlist* (one repo is one licence-review unit); others are recorded
    as skipped. Returns a :class:`SyncPlan` that composes with the category plan's items and corrections.
    """
    plan = SyncPlan()
    for entry in catalog:
        if not allowlist.allows(model_name=entry.repo):
            plan.items.append(
                SyncItem(
                    category=ANNOTATOR_CATEGORY,
                    model_name=entry.repo,
                    file_name=entry.filename,
                    action=SyncAction.SKIPPED_NOT_ALLOWLISTED,
                ),
            )
            continue
        item, correction = _plan_annotator_file(
            entry,
            store=store,
            byte_source=byte_source,
            allowlist=allowlist,
            apply=apply,
        )
        plan.items.append(item)
        if correction is not None:
            plan.corrections.append(correction)
    return plan
