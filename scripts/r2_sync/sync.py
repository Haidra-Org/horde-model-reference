"""Operator CLI for reviewing, reconciling, and publishing the gated R2 mirror."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference import HordeModelReferenceSettings
from horde_model_reference.on_disk_layout import resolve_weights_root
from scripts.r2_sync.allowlist import RedistributionPolicy
from scripts.r2_sync.byte_source import LocalThenOriginByteSource
from scripts.r2_sync.manifest import MIRROR_MANIFEST_KEY, build_mirror_manifest, write_mirror_manifest
from scripts.r2_sync.object_store import InMemoryObjectStore, R2ObjectStore
from scripts.r2_sync.planner import SyncAction, SyncPlan, build_sync_plan, hostable_categories

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
    from horde_model_reference.model_reference_records import GenericModelRecord
    from scripts.r2_sync.object_store import ObjectStore

_FAILURE_ACTIONS = frozenset({SyncAction.MISSING_BYTES, SyncAction.HASH_MISMATCH})
_DEFAULT_BUILD_DIR = Path("build/r2_sync")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse a fail-closed command line; there is deliberately no policy bypass."""
    parser = argparse.ArgumentParser(description="Review and sync approved hostable models to gated R2 storage.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("candidates", "plan", "reconcile", "apply"),
        default="plan",
        help="candidates inventories policy work; plan/reconcile inspect R2; apply uploads and publishes inventory.",
    )
    parser.add_argument("--policy", type=Path, default=None, help="Strict redistribution policy JSON.")
    parser.add_argument("--category", action="append", default=[], help="Exact category to process (repeatable).")
    parser.add_argument("--model", action="append", default=[], help="Exact model name to process (repeatable).")
    parser.add_argument("--weights-root", type=Path, default=None, help="Local model-weights root to search.")
    parser.add_argument("--extra-root", type=Path, action="append", default=[], help="Extra local root (repeatable).")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional origin-download cache. Without it, plan/apply use only already-local bytes.",
    )
    parser.add_argument(
        "--candidate-report",
        type=Path,
        default=_DEFAULT_BUILD_DIR / "candidates.json",
        help="Machine-readable policy/reference reconciliation report.",
    )
    parser.add_argument(
        "--backfill-report",
        type=Path,
        default=_DEFAULT_BUILD_DIR / "sha256_backfill.json",
        help="Computed replacements for unknown sha256 declarations.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=_DEFAULT_BUILD_DIR / "mirror_manifest.json",
        help="Local copy of the generated client inventory.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every file outcome.")
    return parser.parse_args(argv)


def _build_store(settings: HordeModelReferenceSettings, *, apply: bool) -> ObjectStore:
    """Build the real R2 store, or an explicitly approximate in-memory store for read-only planning."""
    if settings.r2.upload_bucket and settings.r2.upload_endpoint_url:
        return R2ObjectStore(settings.r2)
    if apply:
        raise SystemExit("Cannot apply without R2 credentials (HORDE_MODEL_REFERENCE_R2__UPLOAD_*).")
    logger.warning("R2 credentials are absent; this plan treats every object as absent.")
    return InMemoryObjectStore()


def _load_references() -> Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None]:
    """Load canonical references and refuse to hide category load failures with empty mappings."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    references = dict(ModelReferenceManager().get_all_model_references_or_none())
    failed = [str(category) for category in hostable_categories() if references.get(category) is None]
    if failed:
        raise RuntimeError(f"Hostable reference categories failed to load: {', '.join(failed)}")
    return references


def _filter_references(
    references: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None],
    *,
    categories: Sequence[str],
    models: Sequence[str],
) -> dict[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None]:
    """Apply exact optional operator filters while retaining every category key."""
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

    selected_categories = {MODEL_REFERENCE_CATEGORY(value) for value in categories}
    selected_models = set(models)
    filtered: dict[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None] = {}
    for category, records in references.items():
        if selected_categories and category not in selected_categories:
            filtered[category] = {}
        elif records is None or not selected_models:
            filtered[category] = records
        else:
            filtered[category] = {name: record for name, record in records.items() if name in selected_models}
    return filtered


def _candidate_payload(
    references: Mapping[MODEL_REFERENCE_CATEGORY, Mapping[str, GenericModelRecord] | None],
    policy: RedistributionPolicy,
) -> dict[str, object]:
    """Return all declared files plus stale policy targets so adding candidates is a reviewable edit."""
    candidates: list[dict[str, object]] = []
    seen: set[tuple[MODEL_REFERENCE_CATEGORY, str]] = set()
    for category in hostable_categories():
        for model_name, record in (references.get(category) or {}).items():
            seen.add((category, model_name))
            for download in record.config.download:
                decision = policy.decision_for(
                    category=category,
                    model_name=model_name,
                    file_name=download.file_name,
                )
                candidates.append(
                    {
                        "category": str(category),
                        "model_name": model_name,
                        "file_name": download.file_name,
                        "source_url": download.file_url,
                        "sha256": download.sha256sum,
                        "size_bytes": download.size_bytes,
                        "status": decision.decision if decision is not None else "unreviewed",
                        "policy_note": decision.note if decision is not None else None,
                    },
                )
    unmatched = [
        {
            "category": str(category),
            "model_name": name,
            "decision": entry.decision,
            "files": entry.files,
        }
        for (category, name), entry in policy.entries.items()
        if (category, name) not in seen
    ]
    summary: dict[str, int] = {}
    for item in candidates:
        status = str(item["status"])
        summary[status] = summary.get(status, 0) + 1
    return {"schema_version": 1, "summary": summary, "candidates": candidates, "unmatched_policy": unmatched}


def _write_json(payload: object, path: Path) -> None:
    """Write deterministic JSON and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_backfill_report(plan: SyncPlan, path: Path) -> None:
    """Write computed sha256 corrections, including an empty report."""
    payload = [
        {
            "category": item.category,
            "model_name": item.model_name,
            "file_name": item.file_name,
            "old_sha256": item.old_sha256,
            "new_sha256": item.new_sha256,
        }
        for item in plan.corrections
    ]
    _write_json(payload, path)


def _report(plan: SyncPlan, *, verbose: bool, command: str) -> None:
    """Log action totals and optionally each file outcome."""
    counts = plan.counts()
    logger.info("R2 sync {}: {}", command, {str(action): count for action, count in counts.items()})
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


def _publish_manifest(store: ObjectStore, plan: SyncPlan, output: Path) -> None:
    """Write and upload the inventory only after the entire apply plan succeeds."""
    manifest = build_mirror_manifest(plan)
    write_mirror_manifest(manifest, output)
    with tempfile.TemporaryDirectory(prefix="hmr-r2-manifest-") as temporary_dir:
        upload_path = Path(temporary_dir) / "current.json"
        write_mirror_manifest(manifest, upload_path)
        store.put(
            MIRROR_MANIFEST_KEY,
            upload_path,
            metadata={"schema_version": str(manifest.schema_version)},
            content_type="application/json",
        )
    logger.info("Published {} mirrored object(s) to {}", len(manifest.objects), MIRROR_MANIFEST_KEY)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected operator command, returning nonzero for incomplete file processing."""
    args = _parse_args(argv)
    if args.command == "apply" and (args.category or args.model):
        raise SystemExit(
            "Filtered apply is unsafe because publication must describe the complete mirrored set; "
            "use filters with candidates/plan, then run an unfiltered apply.",
        )
    policy = RedistributionPolicy.load(args.policy)
    all_references = _load_references()
    references = _filter_references(
        all_references,
        categories=args.category,
        models=args.model,
    )
    candidate_payload = _candidate_payload(all_references, policy)
    _write_json(candidate_payload, args.candidate_report)
    logger.info("Wrote candidate reconciliation report to {}", args.candidate_report)
    if args.command == "candidates":
        logger.info("Candidate summary: {}", candidate_payload["summary"])
        return 0

    apply = args.command == "apply"
    store = _build_store(HordeModelReferenceSettings(), apply=apply)
    byte_source = LocalThenOriginByteSource(
        weights_root=args.weights_root or resolve_weights_root(),
        extra_roots=tuple(args.extra_root),
        cache_dir=args.cache_dir,
    )
    plan = build_sync_plan(references, allowlist=policy, store=store, byte_source=byte_source, apply=apply)
    _report(plan, verbose=args.verbose, command=args.command)
    _write_backfill_report(plan, args.backfill_report)

    counts = plan.counts()
    failures = sum(counts.get(action, 0) for action in _FAILURE_ACTIONS)
    if failures:
        logger.error("{} approved file(s) could not be verified; no manifest was published.", failures)
        return 1
    if apply:
        _publish_manifest(store, plan, args.manifest_output)
    elif args.command == "reconcile":
        write_mirror_manifest(build_mirror_manifest(plan), args.manifest_output)
        logger.info("Wrote reconciled local manifest to {}", args.manifest_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
