"""Offline pass: populate reference records with component content hashes and derive the canonical set.

Fetches the live reference, computes the torch-free tensor-region content hash of each model's VAE and
text-encoder components over HTTP range requests (never downloading a whole checkpoint), then optionally
writes the populated per-category records locally and/or submits them to a live PRIMARY service. It always
writes a ``canonical_components.json`` registry, and reports what could not be hashed.

What gets hashed, per model:

* every declared standalone component file (a ``DownloadRecord`` whose ``file_purpose`` is ``vae`` or
  ``text_encoders``) is hashed whole and recorded on that download's ``content_hash``; this covers the
  split-file families (Flux/Qwen/Z-Image), whose large encoders live here.
* a monolithic checkpoint (a model with no separate VAE file) has its **embedded VAE** extracted and
  recorded in the config's ``embedded_component_hashes``. Embedded text encoders are intentionally not
  hashed: ComfyUI renames and reshuffles those keys at load, so a torch-free file hash cannot reproduce the
  standalone form.

Anything that cannot be hashed (a ``.ckpt`` pickle, a host that ignores range requests, a network error, a
checkpoint with no VAE) is skipped and left without a hash, so it simply falls back to a normal load
downstream. The canonical set is then derived by frequency with a curated allow/deny override.

Submitting is opt-in and cautious: ``--submit`` writes the populated records to a PRIMARY service, but only
previews (dry-run) unless ``--no-dry-run`` is also passed. ``--api-version v2`` PUTs the v2 record (which
carries the new hash fields natively) and needs a ``canonical_format='v2'`` deployment; ``--api-version v1``
converts each record to the legacy shape and POST/PUTs it to the legacy image-generation endpoint (matching
the ``submit_*`` scripts). Run from the checkout so the library is importable, e.g.::

    uv run --no-sync scripts/hash_components.py --output-dir ./component_hashes --limit 20
    uv run --no-sync scripts/hash_components.py --submit --api-version v2 --apikey "$AI_HORDE_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from loguru import logger

from horde_model_reference.canonical_components import DEFAULT_MIN_SHARED_MODELS, derive_canonical_registry
from horde_model_reference.component_hash import (
    ComponentKind,
    NoComponentTensorsError,
    RangeNotSupportedError,
    UnsupportedContainerError,
    component_kind_for_purpose,
    hash_embedded_component_url,
    hash_standalone_component_url,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_manager import ModelReferenceManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from horde_model_reference.model_reference_records import GenericModelRecord

_SAFETENSORS_SUFFIX = ".safetensors"
_EMBEDDABLE_KINDS = (ComponentKind.VAE, ComponentKind.TEXT_ENCODERS)
_DEFAULT_PRIMARY_URL = "https://models.aihorde.net"
_APIKEY_ENV_VARS = ("AI_HORDE_API_KEY", "HORDE_API_KEY")
_SUBMIT_OK_STATUSES = frozenset({200, 201, 202})


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
    """Execute a hashing task, returning the hash or None when the component cannot be hashed."""
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


def hash_category_records(
    records: dict[str, GenericModelRecord],
    *,
    session: requests.Session,
    skip_existing: bool,
    limit: int,
) -> list[str]:
    """Hash the components of every record in a category in place; return the names that changed."""
    changed: list[str] = []
    for processed, (name, record) in enumerate(records.items()):
        if limit and processed >= limit:
            break
        wrote_any = False
        for task in plan_component_hash_tasks(record, skip_existing=skip_existing):
            content_hash = _hash_task(task, session=session)
            if content_hash is None:
                continue
            _record_hash(record, task, content_hash)
            wrote_any = True
            form = "embedded" if task.embedded else "file"
            logger.info("Hashed {} {} for {}: {}", form, task.kind, name, content_hash)
        if wrote_any:
            changed.append(name)
    return changed


def _resolve_apikey(cli_value: str | None) -> str:
    """Return the API key from the CLI or a known environment variable, exiting if none is set."""
    import os

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
        url = f"{root}/api/model_references/v2/{category.value}/{name}"
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


def run(args: argparse.Namespace) -> int:
    """Fetch the reference, hash components, emit/submit populated records, and write the canonical registry."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_set, deny_set = _load_curated(args.curated_json, args.allow, args.deny)
    apikey = _resolve_apikey(args.apikey) if args.submit else ""

    manager = ModelReferenceManager()
    all_references = manager.get_all_model_references(overwrite_existing=args.overwrite_existing)

    session = requests.Session()
    combined: dict[str, GenericModelRecord] = {}
    for category in args.categories:
        records = all_references.get(category, {})
        if not records:
            logger.warning("No records for category {}", category)
            continue
        changed = hash_category_records(records, session=session, skip_existing=args.skip_existing, limit=args.limit)
        logger.info("Category {}: {} of {} models changed", category, len(changed), len(records))
        if args.emit_json:
            serialized = ModelReferenceManager.model_reference_to_json_dict(records)
            (output_dir / f"{category.value}.records.json").write_text(
                json.dumps(serialized, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if args.submit:
            for name in changed:
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
    parser.add_argument("--output-dir", default="component_hashes", help="Directory for the emitted JSON files.")
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
    parser.add_argument("--skip-existing", action="store_true", help="Do not re-hash records with a hash already.")
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
