"""Seed the canonical ``controlnet_annotator`` reference file on a PRIMARY deployment.

The ControlNet annotator category is PRIMARY-API canonical: there is no legacy GitHub source to fetch it
from, so a PRIMARY instance needs its ``controlnet_annotator.json`` written once from the verified in-package
catalog (:mod:`horde_model_reference.annotator_records`). After this bootstrap, sha256 backfills are applied
to the records the same way as every other category (``scripts/apply_backfill_report.py``).

Semantics are **add-missing** (idempotent and safe to re-run): a record is written only when its model name is
absent from the file, so a re-run never clobbers a record whose hashes were already backfilled. Use
``--apply`` to write; the default is a dry-run.

Usage::

    uv run python scripts/seed_annotator_reference.py --dry-run
    uv run python scripts/seed_annotator_reference.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from horde_model_reference import horde_model_reference_paths
from horde_model_reference.annotator_records import annotator_records
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write the file (default is a dry-run).")
    mode.add_argument("--dry-run", action="store_true", help="Plan only; do not write (the default).")
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="PRIMARY model-reference base path (default: the configured horde_model_reference_paths.base_path).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Seed (add-missing) the controlnet_annotator reference file. Returns 0 on success."""
    args = _parse_args(argv)
    base_path = args.base_path or horde_model_reference_paths.base_path
    file_path = horde_model_reference_paths.get_model_reference_file_path(
        MODEL_REFERENCE_CATEGORY.controlnet_annotator,
        base_path=base_path,
    )
    if file_path is None:
        sys.exit("No file path configured for controlnet_annotator.")

    existing: dict[str, Any] = {}
    if file_path.exists():
        existing = json.loads(file_path.read_text(encoding="utf-8"))

    records = annotator_records()
    to_add = {name: record for name, record in records.items() if name not in existing}

    logger.info(
        "Annotator seed: {} record(s) in catalog, {} already present, {} to add -> {}",
        len(records),
        len(records) - len(to_add),
        len(to_add),
        file_path,
    )
    for name in sorted(to_add):
        logger.info("  + {}", name)

    if not to_add:
        logger.info("Nothing to add; file is already seeded.")
        return 0
    if not args.apply:
        logger.info("--dry-run: no file written. Re-run with --apply to write.")
        return 0

    merged = dict(existing)
    for name, record in to_add.items():
        merged[name] = record.model_dump(mode="json", exclude_none=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote {} annotator record(s) to {}", len(to_add), file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
