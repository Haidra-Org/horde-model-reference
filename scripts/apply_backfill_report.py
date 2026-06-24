"""Apply a backfill report's sha256 corrections to the canonical horde-model-reference deployment.

Reads a ``backfill_report.json`` produced by the R2-sync tool and pushes every sha256 correction the same
way, regardless of category (including ``controlnet_annotator``, which is now a first-class category rather
than a bespoke source-edited catalog): the current record is fetched from the live API on
``models.aihorde.net``, the matching ``config.files`` entries have their ``sha256sum`` patched, and the
record is PUT back.

Usage::

    uv run python scripts/apply_backfill_report.py --apikey "$AI_HORDE_API_KEY" --dry-run
    uv run python scripts/apply_backfill_report.py --apikey "$AI_HORDE_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://models.aihorde.net"
DEFAULT_REPORT_PATH = Path("backfill_report.json")

# ── helpers ──────────────────────────────────────────────────────────────


def _resolve_apikey(cli_value: str | None) -> str:
    apikey = cli_value or os.environ.get("AI_HORDE_API_KEY") or os.environ.get("HORDE_API_KEY")
    if not apikey:
        sys.exit("No API key provided. Pass --apikey or set AI_HORDE_API_KEY / HORDE_API_KEY.")
    return apikey


def _load_report(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        sys.exit(f"Backfill report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _group_by_category(report: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group report entries by category, preserving order within each group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in report:
        groups.setdefault(entry["category"], []).append(entry)
    return groups


# ── model-reference API corrections ──────────────────────────────────────


def _fetch_legacy_record(client: httpx.Client, base_url: str, category: str) -> dict[str, Any]:
    """GET the full legacy JSON dict for *category* from the v1 API."""
    url = f"{base_url.rstrip('/')}/api/model_references/v1/{category}"
    resp = client.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _patch_config_files(record: dict[str, Any], corrections: list[dict[str, Any]]) -> int:
    """Update ``sha256sum`` in ``config.files`` entries matching each correction's ``file_name``.

    Returns the number of entries patched.
    """
    config: dict[str, Any] = record.get("config", {})
    files: list[dict[str, Any]] = config.get("files", [])
    patched = 0

    for corr in corrections:
        target = corr["file_name"]
        new_sha = corr["new_sha256"]
        for fentry in files:
            if fentry.get("path") == target:
                old = fentry.get("sha256sum")
                if old != new_sha:
                    fentry["sha256sum"] = new_sha
                    patched += 1
                    print(f"    {target}: {old} -> {new_sha}")
                else:
                    print(f"    {target}: already correct ({new_sha})")
                break
        else:
            print(f"    WARNING: file '{target}' not found in config.files; skipping.", file=sys.stderr)

    return patched


def _put_legacy_record(client: httpx.Client, base_url: str, category: str, record: dict[str, Any]) -> httpx.Response:
    """PUT the updated legacy record back to the v1 API."""
    url = f"{base_url.rstrip('/')}/api/model_references/v1/{category}"
    return client.put(url, json=record, timeout=30)


def _apply_model_reference_corrections(
    client: httpx.Client,
    base_url: str,
    category: str,
    entries: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> bool:
    """Fetch, patch, and PUT a model-reference category.  Return True on success."""
    print(f"\nCategory: {category} ({len(entries)} correction(s))")

    # The backfill report groups by model_name, but for categories like safety_checker
    # where the model name equals the key, we can just patch the single record.
    # Group by model_name for generality.
    by_model: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_model.setdefault(e["model_name"], []).append(e)

    try:
        legacy_json = _fetch_legacy_record(client, base_url, category)
    except httpx.HTTPError as exc:
        print(f"  FAILED to fetch current record: {exc}", file=sys.stderr)
        return False

    any_patched = False
    for model_name, corrections in by_model.items():
        if model_name not in legacy_json:
            print(
                f"  WARNING: model '{model_name}' not found in {category} response; skipping.",
                file=sys.stderr,
            )
            continue

        record = legacy_json[model_name]
        patched = _patch_config_files(record, corrections)
        if patched == 0:
            print(f"  No changes needed for '{model_name}'.")
            continue

        any_patched = True
        if dry_run:
            print(f"  [DRY-RUN] Would PUT updated record for '{model_name}'.")
            continue

        try:
            resp = _put_legacy_record(client, base_url, category, record)
        except httpx.HTTPError as exc:
            print(f"  FAILED to PUT updated record for '{model_name}': {exc}", file=sys.stderr)
            return False

        if resp.status_code in (200, 202):
            verb = "queued for approval" if resp.status_code == 202 else "updated"
            print(f"  OK ({resp.status_code}): '{model_name}' {verb}.")
        else:
            print(f"  FAILED ({resp.status_code}): {resp.text}", file=sys.stderr)
            return False

    if not any_patched:
        print("  All entries already up-to-date.")
    return True


# ── v2-only category corrections (no legacy representation) ───────────────


def _is_v2_only_category(category: str) -> bool:
    """Return whether *category* has no legacy representation (e.g. ``controlnet_annotator``).

    Such a category has no v1 endpoint, so its corrections must go through the v2 per-model API.
    """
    try:
        from horde_model_reference.meta_consts import get_category_descriptor

        return get_category_descriptor(category).has_legacy_format is False
    except Exception:
        return False


def _patch_config_download(record: dict[str, Any], corrections: list[dict[str, Any]]) -> int:
    """Update ``sha256sum`` in v2 ``config.download`` entries matching each correction's ``file_name``.

    Returns the number of entries patched.
    """
    downloads: list[dict[str, Any]] = record.get("config", {}).get("download", [])
    patched = 0
    for corr in corrections:
        target = corr["file_name"]
        new_sha = corr["new_sha256"]
        for dentry in downloads:
            if dentry.get("file_name") == target:
                old = dentry.get("sha256sum")
                if old != new_sha:
                    dentry["sha256sum"] = new_sha
                    patched += 1
                    print(f"    {target}: {old} -> {new_sha}")
                else:
                    print(f"    {target}: already correct ({new_sha})")
                break
        else:
            print(f"    WARNING: file '{target}' not found in config.download; skipping.", file=sys.stderr)
    return patched


def _apply_v2_corrections(
    client: httpx.Client,
    base_url: str,
    category: str,
    entries: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> bool:
    """Fetch, patch, and PUT each model of a v2-only category via the v2 per-model API. Return True on success.

    Like the v1 path, a successful PUT returns ``202`` (queued for approval); a maintainer applies the batch.
    """
    print(f"\nCategory: {category} ({len(entries)} correction(s)) [v2-only]")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_model.setdefault(entry["model_name"], []).append(entry)

    base = base_url.rstrip("/")
    ok = True
    for model_name, corrections in by_model.items():
        url = f"{base}/api/model_references/v2/{category}/model/{model_name}"
        try:
            resp = client.get(url, timeout=30)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  WARNING: could not fetch '{model_name}' ({exc}); skipping.", file=sys.stderr)
            continue

        record = resp.json()
        if _patch_config_download(record, corrections) == 0:
            print(f"  No changes needed for '{model_name}'.")
            continue
        if dry_run:
            print(f"  [DRY-RUN] Would PUT updated record for '{model_name}'.")
            continue

        try:
            put = client.put(url, json=record, timeout=30)
        except httpx.HTTPError as exc:
            print(f"  FAILED to PUT '{model_name}': {exc}", file=sys.stderr)
            ok = False
            continue
        if put.status_code in (200, 201, 202):
            verb = "queued for approval" if put.status_code == 202 else "updated"
            print(f"  OK ({put.status_code}): '{model_name}' {verb}.")
        else:
            print(f"  FAILED ({put.status_code}): {put.text}", file=sys.stderr)
            ok = False
    return ok


# ── main ─────────────────────────────────────────────────────────────────


def main() -> int:
    """Apply every sha256 correction in the backfill report to the canonical reference. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--apikey",
        help="AI-Horde API key with write access (or set AI_HORDE_API_KEY / HORDE_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Service base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Path to the backfill report JSON (default: {DEFAULT_REPORT_PATH}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, plan, and print what would be done; do not modify anything.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    args = parser.parse_args()

    report = _load_report(args.report)
    groups = _group_by_category(report)
    print(f"Loaded {len(report)} correction(s) across {len(groups)} category/ies.")

    # API key is only required when actually writing (non-dry-run).
    apikey: str | None = None
    if groups and not args.dry_run:
        apikey = _resolve_apikey(args.apikey)

    ok = True
    headers = {"apikey": apikey, "Content-Type": "application/json"} if apikey else {}

    with httpx.Client(headers=headers, timeout=args.timeout) as client:
        for category, entries in groups.items():
            apply_fn = _apply_v2_corrections if _is_v2_only_category(category) else _apply_model_reference_corrections
            if not apply_fn(client, args.base_url, category, entries, dry_run=args.dry_run):
                ok = False

    if args.dry_run:
        print("\n--dry-run: no changes were made.")
    elif ok:
        print("\nAll corrections applied successfully.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
