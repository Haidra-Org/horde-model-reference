"""CLI entry point for mirroring hostable model files onto the gated Cloudflare R2 bucket.

Run a dry-run (the default) to see what would be uploaded, then ``--apply`` to perform it::

    uv run python -m scripts.r2_sync.sync --dry-run
    uv run python -m scripts.r2_sync.sync --apply

The run is idempotent (content-addressed; already-present objects are skipped), opt-in (only models in the
redistributable allowlist are touched), and emits a backfill report of the real sha256 it computed for any
record still carrying the ``"FIXME"`` sentinel. A CI workflow consumes that report to open a PR correcting the
reference (the maintainer-facing half of the "backfill + open a PR" decision).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference import HordeModelReferenceSettings
from horde_model_reference.on_disk_layout import resolve_weights_root
from scripts.r2_sync.allowlist import RedistributableAllowlist
from scripts.r2_sync.byte_source import LocalThenOriginByteSource
from scripts.r2_sync.object_store import InMemoryObjectStore, R2ObjectStore
from scripts.r2_sync.planner import SyncAction, SyncPlan, build_sync_plan

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from horde_model_reference.model_reference_records import GenericModelRecord
    from scripts.r2_sync.object_store import ObjectStore

_FAILURE_ACTIONS = frozenset({SyncAction.MISSING_BYTES, SyncAction.HASH_MISMATCH})


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command-line arguments for the sync tool."""
    parser = argparse.ArgumentParser(description="Verify and upload hostable model files to the gated R2 bucket.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Perform uploads (default is a dry-run plan only).")
    mode.add_argument("--dry-run", action="store_true", help="Plan only; never upload (the default).")
    parser.add_argument("--allowlist", type=Path, default=None, help="Path to the redistributable allowlist JSON.")
    parser.add_argument(
        "--no-allowlist",
        action="store_true",
        help="Disable allowlist filtering; treat every model/annotator as allowed.",
    )
    parser.add_argument("--weights-root", type=Path, default=None, help="Local model-weights root to mirror from.")
    parser.add_argument(
        "--extra-root",
        type=Path,
        action="append",
        default=[],
        help="Additional local weights root to search (repeatable).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Where to cache origin downloads when a file is not present locally. This enables fetching bytes "
            "from origin hosts to hash them, INCLUDING during a dry-run (needed to content-address "
            "still-FIXME records). Omit to operate on local files only and never download."
        ),
    )
    parser.add_argument("--backfill-report", type=Path, default=None, help="Write computed sha256 corrections here.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every file's outcome, not just totals.")
    return parser.parse_args(argv)


def _build_store(settings: HordeModelReferenceSettings, *, apply: bool) -> ObjectStore:
    """Return the bucket to operate against: the real R2 store when configured, else an empty in-memory one.

    A real store is needed even for a dry-run so "already present" is answered truthfully. When credentials are
    absent a dry-run falls back to an empty in-memory store (over-reporting uploads, with a warning); an apply
    without credentials is a hard error.
    """
    if settings.r2.upload_bucket and settings.r2.upload_endpoint_url:
        return R2ObjectStore(settings.r2)
    if apply:
        raise SystemExit("Cannot --apply without R2 credentials (HORDE_MODEL_REFERENCE_R2__UPLOAD_*).")
    logger.warning(
        "No R2 credentials configured (HORDE_MODEL_REFERENCE_R2__UPLOAD_*). This dry-run treats EVERY object as "
        "absent, so it OVER-REPORTS uploads and cannot tell you what is already mirrored. With --cache-dir set it "
        "will also fetch bytes from origin hosts to hash them. Set read-capable R2 credentials for a truthful "
        "dry-run.",
    )
    return InMemoryObjectStore()


def _load_references() -> Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None]:
    """Load all model references via the manager singleton, overlaying the authoritative annotator set.

    The ``controlnet_annotator`` category is canonical on the PRIMARY API only, and the verified list of
    annotator files lives in :mod:`horde_model_reference.annotator_records` (derived from the pinned package).
    Overlaying it here lets the mirror tool process annotators uniformly through the generic planner without
    depending on whether the configured source has been seeded yet, and surfaces their ``FIXME`` hashes for
    backfill exactly like any other category.
    """
    from horde_model_reference.annotator_records import annotator_records
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    manager = ModelReferenceManager()
    references = dict(manager.get_all_model_references_or_none())
    references[MODEL_REFERENCE_CATEGORY.controlnet_annotator] = annotator_records()
    return references


def _write_backfill_report(plan: SyncPlan, path: Path) -> None:
    """Write the computed sha256 corrections to *path* as JSON for a CI step to turn into a reference PR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "category": correction.category,
            "model_name": correction.model_name,
            "file_name": correction.file_name,
            "old_sha256": correction.old_sha256,
            "new_sha256": correction.new_sha256,
        }
        for correction in plan.corrections
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote {} sha256 correction(s) to backfill report: {}", len(payload), path)


def _report(plan: SyncPlan, *, verbose: bool, apply: bool) -> None:
    """Log the plan's per-action totals (and each file when *verbose*)."""
    verb = "Uploaded" if apply else "Would upload"
    counts = plan.counts()
    logger.info("R2 sync {}: {}", "apply" if apply else "dry-run", dict(counts))
    logger.info(
        "{} {} file(s); {} already present; {} skipped; {} correction(s) to backfill",
        verb,
        counts.get(SyncAction.UPLOAD, 0),
        counts.get(SyncAction.ALREADY_PRESENT, 0),
        counts.get(SyncAction.SKIPPED_NOT_ALLOWLISTED, 0),
        len(plan.corrections),
    )
    if verbose:
        for item in plan.items:
            logger.info(
                "{} :: {}/{} -> {} [{}]{}",
                item.action,
                item.category,
                item.model_name,
                item.file_name,
                item.key or "-",
                f" ({item.detail})" if item.detail else "",
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sync tool. Returns a non-zero exit code when any file could not be processed."""
    args = _parse_args(argv)
    settings = HordeModelReferenceSettings()
    allowlist: RedistributableAllowlist | None = (
        None if args.no_allowlist else RedistributableAllowlist.load(args.allowlist)
    )
    store = _build_store(settings, apply=args.apply)
    weights_root = args.weights_root or resolve_weights_root()
    byte_source = LocalThenOriginByteSource(
        weights_root=weights_root,
        extra_roots=tuple(args.extra_root),
        cache_dir=args.cache_dir,
    )

    plan = build_sync_plan(
        _load_references(),
        allowlist=allowlist,
        store=store,
        byte_source=byte_source,
        apply=args.apply,
    )

    _report(plan, verbose=args.verbose, apply=args.apply)
    if args.backfill_report and plan.corrections:
        _write_backfill_report(plan, args.backfill_report)

    failures = sum(plan.counts().get(action, 0) for action in _FAILURE_ACTIONS)
    if failures:
        logger.warning("{} file(s) could not be processed (missing bytes or hash mismatch).", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
