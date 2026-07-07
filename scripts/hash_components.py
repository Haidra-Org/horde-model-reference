"""Offline pass: populate reference records with component content hashes and derive the canonical set.

Computes the torch-free tensor-region content hash of each model's VAE and text-encoder components and records
it on the reference, then optionally writes the populated per-category records locally and/or submits them to a
live PRIMARY service. It always writes a ``canonical_components.json`` registry, and reports what could not be
hashed.

The pass is built for an operator running it incrementally on a machine that already holds many models:

* **Local by default.** ``--source local`` (the default) hashes only components whose files are already on
  disk under ``$AIWORKER_CACHE_HOME`` (the horde cache convention), reading them directly (no download). Use
  ``--source auto`` to fall back to HTTP range requests for anything not on disk, or ``--source remote`` to
  hash everything over the network (never downloading a whole checkpoint).
* **Resumable.** A progress file (``hash_progress.json`` in the output dir) records every result with a
  timestamp as it happens, so a run interrupted midway resumes where it left off instead of restarting. Prior
  results are re-applied to the records on the next run, and previously-failed components are not retried
  unless ``--retry-failed`` is given.
* **Pre-flight.** Before hashing, it reports how many models are on disk, how many components are already
  hashed (in the canonical record or in the progress file), and how many remain, so the operator knows the
  scope up front.
* **Visible progress.** A tqdm bar plus stage banners show what is happening at the console.

What gets hashed, per model: every declared standalone component file (a ``DownloadRecord`` whose
``file_purpose`` is ``vae`` or ``text_encoders``) is hashed whole; a monolithic checkpoint has its embedded
VAE and text encoders extracted and hashed. The embedded VAE hash equals the same VAE as a standalone file,
while the embedded text-encoder hash is a monolithic-consistent identity (matched only against other
monolithic checkpoints, since ComfyUI renames those keys at load). Anything unhashable (a ``.ckpt`` pickle, a
host that ignores range requests, a checkpoint with no such tensors) is recorded as failed and left without a
hash, so it falls back to a normal load downstream.

Submitting is opt-in and cautious: ``--submit`` previews (dry-run) unless ``--no-dry-run`` is also passed. Run
from the checkout so the library is importable, e.g.::

    uv run --no-sync scripts/hash_components.py                     # local, on-disk models, resumable
    uv run --no-sync scripts/hash_components.py --source auto        # + fetch anything not on disk
    uv run --no-sync scripts/hash_components.py --submit --no-dry-run --apikey "$AI_HORDE_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import requests
from loguru import logger

from horde_model_reference.canonical_components import DEFAULT_MIN_SHARED_MODELS, derive_canonical_registry
from horde_model_reference.component_hash import (
    ComponentKind,
    NoComponentTensorsError,
    RangeNotSupportedError,
    UnsupportedContainerError,
    component_kind_for_purpose,
    hash_embedded_component_file,
    hash_embedded_component_url,
    hash_standalone_component_file,
    hash_standalone_component_url,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.on_disk_layout import file_paths_for, is_present, resolve_weights_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is a declared dev dependency; degrade gracefully if absent
    _tqdm = None

_SAFETENSORS_SUFFIX = ".safetensors"
_EMBEDDABLE_KINDS = (ComponentKind.VAE, ComponentKind.TEXT_ENCODERS)
_DEFAULT_PRIMARY_URL = "https://models.aihorde.net"
_APIKEY_ENV_VARS = ("AI_HORDE_API_KEY", "HORDE_API_KEY")
_SUBMIT_OK_STATUSES = frozenset({200, 201, 202})
_PROGRESS_VERSION = 1


@dataclass(frozen=True)
class ComponentHashTask:
    """One planned hashing action for a record: which download to fetch, and how to hash it."""

    download_index: int
    url: str
    kind: ComponentKind
    embedded: bool
    """True to extract an embedded VAE from a monolithic checkpoint; False to hash a standalone file whole."""


def _looks_like_safetensors(file_name: str, file_url: str) -> bool:
    """Return whether a download is a safetensors container (the only form this pass can hash)."""
    if file_name.lower().endswith(_SAFETENSORS_SUFFIX):
        return True
    return file_url.lower().split("?", 1)[0].endswith(_SAFETENSORS_SUFFIX)


def plan_component_hash_tasks(record: GenericModelRecord, *, skip_existing: bool) -> list[ComponentHashTask]:
    """Plan which components of *record* to hash, without doing any I/O.

    Standalone component files are hashed whole. A model with no standalone file for a component kind is
    treated as monolithic for that kind, and its primary safetensors files are probed for the embedded VAE
    and text encoders. When *skip_existing* is set, components that already carry a hash are left alone.
    """
    downloads = record.config.download
    standalone_kinds = {component_kind_for_purpose(download.file_purpose) for download in downloads}
    embedded_hashes = record.config.embedded_component_hashes or {}

    tasks: list[ComponentHashTask] = []
    for index, download in enumerate(downloads):
        if not _looks_like_safetensors(download.file_name, download.file_url):
            continue
        kind = component_kind_for_purpose(download.file_purpose)
        if kind is not None:
            if skip_existing and download.content_hash is not None:
                continue
            tasks.append(ComponentHashTask(index, download.file_url, kind, embedded=False))
            continue
        # Primary checkpoint: probe for each embeddable component that has no standalone file of its own.
        for embed_kind in _EMBEDDABLE_KINDS:
            if embed_kind in standalone_kinds:
                continue
            if skip_existing and embedded_hashes.get(embed_kind.value) is not None:
                continue
            tasks.append(ComponentHashTask(index, download.file_url, embed_kind, embedded=True))
    return tasks


def _hash_task(task: ComponentHashTask, *, session: requests.Session) -> str | None:
    """Execute a hashing task over the network, returning the hash or None when it cannot be hashed."""
    try:
        if task.embedded:
            return hash_embedded_component_url(task.url, task.kind, session=session)
        return hash_standalone_component_url(task.url, session=session)
    except NoComponentTensorsError:
        return None
    except UnsupportedContainerError as skip_reason:
        logger.debug("Not hashable ({}): {}", skip_reason, task.url)
        return None
    except (RangeNotSupportedError, requests.RequestException) as fetch_error:
        logger.warning("Could not fetch component ({}): {}", fetch_error, task.url)
        return None


def _record_hash(record: GenericModelRecord, task: ComponentHashTask, content_hash: str) -> None:
    """Store *content_hash* on the record, on the download for a standalone file or in the embedded map."""
    if task.embedded:
        embedded = dict(record.config.embedded_component_hashes or {})
        embedded[task.kind.value] = content_hash
        record.config.embedded_component_hashes = embedded
    else:
        record.config.download[task.download_index].content_hash = content_hash


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (used to timestamp progress entries)."""
    return datetime.now(UTC).isoformat()


@dataclass
class HashProgress:
    """A resumable, timestamped record of every hashing result, persisted between runs.

    Keyed by a per-component key (category, model, kind, form). Each entry keeps the outcome, the source it
    came from, and when it happened, so a re-run can re-apply prior hashes, skip previously-failed components,
    and report progress. Written to disk after every result so an interrupted run loses nothing.
    """

    path: Path
    started_at: str = field(default_factory=_now_iso)
    results: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> HashProgress:
        """Load an existing progress file, or return a fresh one when it does not exist or is unreadable."""
        if not path.exists():
            return cls(path=path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as read_error:
            logger.warning("Could not read progress file {} ({}); starting fresh", path, read_error)
            return cls(path=path)
        return cls(
            path=path,
            started_at=str(payload.get("started_at", _now_iso())),
            results=dict(payload.get("results", {})),
        )

    def get(self, key: str) -> dict[str, object] | None:
        """Return the recorded result for *key*, or None when it has not been hashed yet."""
        return self.results.get(key)

    def record(
        self,
        key: str,
        *,
        category: str,
        name: str,
        kind: str,
        embedded: bool,
        content_hash: str | None,
        status: str,
        reason: str | None,
        source: str,
    ) -> None:
        """Record one component's outcome and persist the whole file (so a crash preserves it)."""
        self.results[key] = {
            "category": category,
            "name": name,
            "kind": kind,
            "embedded": embedded,
            "content_hash": content_hash,
            "status": status,
            "reason": reason,
            "source": source,
            "at": _now_iso(),
        }
        self.save()

    def save(self) -> None:
        """Atomically write the progress file (temp then replace) so it is never left half-written."""
        payload = {
            "version": _PROGRESS_VERSION,
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            "results": self.results,
        }
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def _task_key(category: MODEL_REFERENCE_CATEGORY, name: str, kind: ComponentKind, embedded: bool) -> str:
    """Return a stable, collision-safe progress key for one component (a null byte can't occur in a name)."""
    return "\x00".join((category.value, name, kind.value, "embedded" if embedded else "file"))


@dataclass
class PlannedHash:
    """One component the current run intends to hash, with its record and resolved on-disk path."""

    category: MODEL_REFERENCE_CATEGORY
    name: str
    record: GenericModelRecord
    task: ComponentHashTask
    key: str
    local_path: Path | None
    on_disk: bool


@dataclass
class PreflightSummary:
    """Counts describing the scope of a run, shown to the operator before any hashing begins."""

    models_total: int = 0
    models_on_disk: int = 0
    already_hashed: int = 0
    todo_local: int = 0
    todo_remote_only: int = 0
    previously_failed: int = 0


def _apply_progress_to_records(
    records_by_category: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]],
    progress: HashProgress,
) -> dict[MODEL_REFERENCE_CATEGORY, set[str]]:
    """Re-apply every prior successful hash from the progress file onto the loaded records.

    Returns the set of model names touched per category, so a later ``--submit`` pushes exactly the models the
    accumulated progress changed rather than re-submitting the whole reference.
    """
    touched: dict[MODEL_REFERENCE_CATEGORY, set[str]] = {category: set() for category in records_by_category}
    by_value = {category.value: category for category in records_by_category}
    for entry in progress.results.values():
        if entry.get("status") != "hashed" or entry.get("content_hash") is None:
            continue
        category = by_value.get(str(entry.get("category")))
        if category is None:
            continue
        record = records_by_category[category].get(str(entry.get("name")))
        if record is None:
            continue
        kind = ComponentKind(str(entry["kind"]))
        _apply_hash_by_kind(
            record,
            kind,
            embedded=bool(entry.get("embedded")),
            content_hash=str(entry["content_hash"]),
        )
        touched[category].add(str(entry["name"]))
    return touched


def _apply_hash_by_kind(record: GenericModelRecord, kind: ComponentKind, *, embedded: bool, content_hash: str) -> None:
    """Apply *content_hash* to *record* for one component kind (used when re-applying progress with no task)."""
    if embedded:
        embedded_map = dict(record.config.embedded_component_hashes or {})
        embedded_map[kind.value] = content_hash
        record.config.embedded_component_hashes = embedded_map
        return
    for download in record.config.download:
        if component_kind_for_purpose(download.file_purpose) is kind:
            download.content_hash = content_hash
            return


def _build_plan(
    records_by_category: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]],
    root: Path,
    extra_roots: Sequence[Path],
    progress: HashProgress,
    *,
    retry_failed: bool,
    limit: int,
) -> tuple[list[PlannedHash], PreflightSummary]:
    """Decide which components still need hashing and summarise the scope, doing no network or hash I/O."""
    planned: list[PlannedHash] = []
    summary = PreflightSummary()
    for category, records in records_by_category.items():
        for processed, (name, record) in enumerate(records.items()):
            if limit and processed >= limit:
                break
            summary.models_total += 1
            if is_present(record, root, extra_roots=extra_roots):
                summary.models_on_disk += 1
            all_tasks = plan_component_hash_tasks(record, skip_existing=False)
            remaining = plan_component_hash_tasks(record, skip_existing=True)
            summary.already_hashed += len(all_tasks) - len(remaining)
            paths = file_paths_for(record, root, extra_roots=extra_roots)
            for task in remaining:
                key = _task_key(category, name, task.kind, task.embedded)
                prior = progress.get(key)
                if prior is not None and prior.get("status") == "failed" and not retry_failed:
                    summary.previously_failed += 1
                    continue
                local_path = paths[task.download_index] if task.download_index < len(paths) else None
                on_disk = local_path is not None and local_path.exists()
                if on_disk:
                    summary.todo_local += 1
                else:
                    summary.todo_remote_only += 1
                planned.append(PlannedHash(category, name, record, task, key, local_path, on_disk))
    return planned, summary


def _hash_one(
    planned: PlannedHash,
    *,
    source: str,
    session: requests.Session,
) -> tuple[str | None, str, str | None, str]:
    """Hash a single planned component, returning ``(content_hash, status, reason, source_used)``.

    Honours the source policy: ``local`` reads an on-disk file (and reports ``absent`` when it is missing);
    ``remote`` always fetches over HTTP ranges; ``auto`` reads locally when present and fetches otherwise.
    """
    task = planned.task
    read_local = planned.on_disk and source in ("local", "auto")
    if not read_local and source == "local":
        return None, "absent", "not on disk", "local"

    if read_local and planned.local_path is not None:
        try:
            if task.embedded:
                return hash_embedded_component_file(planned.local_path, task.kind), "hashed", None, "local"
            return hash_standalone_component_file(planned.local_path), "hashed", None, "local"
        except NoComponentTensorsError:
            return None, "failed", "no component tensors", "local"
        except UnsupportedContainerError as skip_reason:
            return None, "failed", str(skip_reason), "local"

    content_hash = _hash_task(task, session=session)
    if content_hash is None:
        return None, "failed", "unhashable or unreachable", "remote"
    return content_hash, "hashed", None, "remote"


def _hash_planned(
    planned: Sequence[PlannedHash],
    progress: HashProgress,
    touched: dict[MODEL_REFERENCE_CATEGORY, set[str]],
    *,
    source: str,
    session: requests.Session,
) -> int:
    """Hash every planned component, recording each result and showing a progress bar. Returns hashes written."""
    hashed = 0
    with _progress_bar(len(planned), "Hashing components") as bar:
        for item in planned:
            bar.set_postfix_str(f"{item.name[:28]} {item.task.kind.value}")
            content_hash, status, reason, source_used = _hash_one(item, source=source, session=session)
            if content_hash is not None:
                _record_hash(item.record, item.task, content_hash)
                touched.setdefault(item.category, set()).add(item.name)
                hashed += 1
            progress.record(
                item.key,
                category=item.category.value,
                name=item.name,
                kind=item.task.kind.value,
                embedded=item.task.embedded,
                content_hash=content_hash,
                status=status,
                reason=reason,
                source=source_used,
            )
            bar.update(1)
    return hashed


def _resolve_apikey(cli_value: str | None) -> str:
    """Return the API key from the CLI or a known environment variable, exiting if none is set."""
    apikey = cli_value or next((os.environ[var] for var in _APIKEY_ENV_VARS if os.environ.get(var)), None)
    if not apikey:
        sys.exit(f"No API key provided. Pass --apikey or set one of {', '.join(_APIKEY_ENV_VARS)}.")
    return apikey


def _report_submission(response: requests.Response, name: str) -> None:
    """Log the outcome of a single submission."""
    if response.status_code in _SUBMIT_OK_STATUSES:
        logger.info("Submitted {}: HTTP {}", name, response.status_code)
    else:
        logger.error("Submit failed for {}: HTTP {} {}", name, response.status_code, response.text[:200])


def _submit_record(
    record: GenericModelRecord,
    name: str,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    api_version: str,
    base_url: str,
    apikey: str,
    session: requests.Session,
    dry_run: bool,
    update: bool,
    timeout: float,
) -> None:
    """Submit one populated record to a PRIMARY service via the chosen API version, honouring dry-run."""
    headers = {"apikey": apikey, "Content-Type": "application/json"}
    root = base_url.rstrip("/")
    if api_version == "v2":
        # The v2 update route is /{category}/model/{name} (with a "/model/" segment) and the name must be
        # URL-encoded (many model names contain spaces). A v2 write is queued for approval (pending), which is
        # also how a change first surfaces as a beta model.
        url = f"{root}/api/model_references/v2/{category.value}/model/{quote(name, safe='')}"
        payload = record.model_dump(mode="json")
        method_name = "PUT"
        sender = session.put
    else:
        from horde_model_reference.legacy.classes.legacy_converters import image_generation_record_to_legacy_dict

        url = f"{root}/api/model_references/v1/{category.value}"
        payload = image_generation_record_to_legacy_dict(record)
        method_name = "PUT" if update else "POST"
        sender = session.put if update else session.post

    if dry_run:
        logger.info("[dry-run] {} {} for {}", method_name, url, name)
        return
    _report_submission(sender(url, json=payload, headers=headers, timeout=timeout), name)


def _load_curated(
    curated_json: Path | None,
    allow: Sequence[str],
    deny: Sequence[str],
) -> tuple[set[str], set[str]]:
    """Merge inline ``--allow``/``--deny`` hashes with an optional curated JSON of the same shape."""
    allow_set = set(allow)
    deny_set = set(deny)
    if curated_json is not None:
        payload = json.loads(curated_json.read_text(encoding="utf-8"))
        allow_set.update(payload.get("allow", []))
        deny_set.update(payload.get("deny", []))
    return allow_set, deny_set


class _NullBar:
    """A no-op stand-in for a tqdm bar, used when tqdm is not installed so the pass still runs."""

    def update(self, _count: int = 1) -> None:
        """Ignore progress updates."""

    def set_postfix_str(self, _text: str) -> None:
        """Ignore the per-item label."""

    def __enter__(self) -> _NullBar:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _progress_bar(total: int, desc: str) -> object:
    """Return a tqdm progress bar, or a no-op bar when tqdm is unavailable."""
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit="component")
    return _NullBar()


def _stage(message: str) -> None:
    """Log a clear stage banner so the operator can follow which phase the pass is in."""
    logger.info("=== {} ===", message)


def _print_preflight(summary: PreflightSummary, *, source: str) -> None:
    """Log the pre-flight scope so the operator sees the plan before any hashing starts."""
    logger.info("Pre-flight ({} source):", source)
    logger.info("  models in scope:        {}", summary.models_total)
    logger.info("  models fully on disk:   {}", summary.models_on_disk)
    logger.info("  components already hashed (record or progress): {}", summary.already_hashed)
    logger.info("  components to hash from disk:  {}", summary.todo_local)
    label = "components on disk-miss (remote)" if source != "local" else "components skipped (not on disk)"
    logger.info("  {}: {}", label, summary.todo_remote_only)
    logger.info("  components previously failed (skipped): {}", summary.previously_failed)


def run(args: argparse.Namespace) -> int:
    """Fetch the reference, hash components (resumable, disk-first), emit/submit, and derive the registry."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_file) if args.progress_file else output_dir / "hash_progress.json"
    allow_set, deny_set = _load_curated(args.curated_json, args.allow, args.deny)
    apikey = _resolve_apikey(args.apikey) if args.submit else ""
    session = requests.Session()

    _stage("Loading model reference")
    manager = ModelReferenceManager()
    all_references = manager.get_all_model_references(overwrite_existing=args.overwrite_existing)
    records_by_category = {category: all_references.get(category, {}) for category in args.categories}
    for category, records in records_by_category.items():
        if not records:
            logger.warning("No records for category {}", category)

    progress = HashProgress.load(progress_path)
    logger.info("Progress file {}: {} prior results", progress_path, len(progress.results))

    _stage("Resolving weights root")
    root = resolve_weights_root(args.cache_home)
    extra_roots = [Path(path) for path in (args.extra_roots or [])]
    logger.info("Weights root: {}", root)

    _stage("Applying prior progress to records")
    touched = _apply_progress_to_records(records_by_category, progress)
    logger.info("Re-applied prior hashes to {} models", sum(len(names) for names in touched.values()))

    _stage("Pre-flight")
    if args.overwrite_existing:
        logger.info("--overwrite-existing: prior record hashes are being re-derived from source")
    planned, summary = _build_plan(
        records_by_category,
        root,
        extra_roots,
        progress,
        retry_failed=args.retry_failed,
        limit=args.limit,
    )
    _print_preflight(summary, source=args.source)

    _stage(f"Hashing {len(planned)} components")
    written = _hash_planned(planned, progress, touched, source=args.source, session=session)
    logger.info("Wrote {} new component hashes this run", written)

    _stage("Emitting records and submitting")
    combined: dict[str, GenericModelRecord] = {}
    for category, records in records_by_category.items():
        if not records:
            continue
        if args.emit_json:
            serialized = ModelReferenceManager.model_reference_to_json_dict(records)
            (output_dir / f"{category.value}.records.json").write_text(
                json.dumps(serialized, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if args.submit:
            for name in sorted(touched.get(category, set())):
                _submit_record(
                    records[name],
                    name,
                    category,
                    api_version=args.api_version,
                    base_url=args.primary_url,
                    apikey=apikey,
                    session=session,
                    dry_run=not args.no_dry_run,
                    update=args.update,
                    timeout=args.timeout,
                )
        combined.update(records)

    _stage("Deriving canonical set")
    registry = derive_canonical_registry(
        combined,
        min_shared_models=args.min_shared_models,
        allow=allow_set,
        deny=deny_set,
    )
    registry_path = output_dir / "canonical_components.json"
    registry_path.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Derived {} canonical components -> {}", len(registry.components), registry_path)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate reference records with component content hashes.")
    parser.add_argument("--output-dir", default="component_hashes", help="Directory for emitted JSON + progress.")
    parser.add_argument("--progress-file", type=Path, default=None, help="Resume/progress file path.")
    parser.add_argument(
        "--source",
        choices=["local", "auto", "remote"],
        default="local",
        help="Where to read components: local (on-disk only, default), auto (disk then HTTP), remote (HTTP).",
    )
    parser.add_argument("--cache-home", default=None, help="Weights base dir (default: $AIWORKER_CACHE_HOME).")
    parser.add_argument("--extra-roots", nargs="*", default=[], help="Additional weights roots to search on disk.")
    parser.add_argument("--retry-failed", action="store_true", help="Re-attempt components recorded as failed.")
    parser.add_argument(
        "--categories",
        nargs="+",
        type=MODEL_REFERENCE_CATEGORY,
        default=[MODEL_REFERENCE_CATEGORY.image_generation],
        help="Model reference categories to hash (default: image_generation).",
    )
    parser.add_argument("--min-shared-models", type=int, default=DEFAULT_MIN_SHARED_MODELS)
    parser.add_argument("--allow", nargs="*", default=[], help="Content hashes to force into the canonical set.")
    parser.add_argument("--deny", nargs="*", default=[], help="Content hashes to exclude from the canonical set.")
    parser.add_argument("--curated-json", type=Path, default=None, help='JSON: {"allow": [...], "deny": [...]}.')
    parser.add_argument("--limit", type=int, default=0, help="Hash only the first N models per category (0 = all).")
    parser.add_argument("--overwrite-existing", action="store_true", help="Force the backend to refetch reference.")
    parser.add_argument("--no-emit-json", action="store_false", dest="emit_json", help="Do not write records JSON.")
    parser.add_argument("--submit", action="store_true", help="Submit populated records to a PRIMARY service.")
    parser.add_argument("--api-version", choices=["v1", "v2"], default="v2", help="Submit API version (default v2).")
    parser.add_argument("--primary-url", default=_DEFAULT_PRIMARY_URL, help="PRIMARY service base URL for --submit.")
    parser.add_argument("--apikey", default=None, help=f"API key for --submit (or set {'/'.join(_APIKEY_ENV_VARS)}).")
    parser.add_argument("--no-dry-run", action="store_true", help="With --submit, perform real writes.")
    parser.add_argument("--update", action="store_true", help="For v1 submit, PUT (update) instead of POST (create).")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the offline component-hashing pass."""
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
