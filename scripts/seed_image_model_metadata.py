"""Seed legacy image-generation records with metadata from the upstream model_mod_data.json.

The upstream repository tracks per-model commit dates (date_added/date_modified/date_removed).
On a legacy-canonical PRIMARY, per-record metadata lives in the legacy category file, so this
script merges those dates into it. Restart the service afterwards: the bumped legacy mtime makes
the backend rebuild the v2 projection, which carries the seeded metadata through conversion.

Merge policy: upstream data is authoritative for created_at/updated_at where an entry exists and
was never removed; entries removed upstream are skipped because a local record of the same name is
a separate lineage; server-only models keep any existing metadata untouched; models with no
metadata at all are left absent so startup population assigns current-time values once.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from horde_model_reference.util import atomic_write_json

DEFAULT_MOD_DATA_URL = (
    "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/refs/heads/main/model_mod_data.json"
)


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse command line arguments for a guarded metadata seeding run."""
    parser = argparse.ArgumentParser(
        description="Seed legacy image records with metadata from the upstream model_mod_data.json.",
    )
    parser.add_argument("--input", type=Path, required=True, help="PRIMARY legacy stable_diffusion.json to seed.")
    parser.add_argument(
        "--mod-data-url",
        default=DEFAULT_MOD_DATA_URL,
        help="URL of the upstream model_mod_data.json.",
    )
    parser.add_argument("--apply", action="store_true", help="Persist the seeding. Omit for a dry run.")
    return parser.parse_args(arguments)


def seed_image_model_metadata(
    category_records: dict[str, Any],
    mod_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Return records with upstream-derived metadata merged in, plus per-model outcome counts."""
    counts = {"from_upstream": 0, "kept_local": 0, "left_empty": 0, "skipped_removed": 0}
    seeded_records = dict(category_records)
    for name, record in seeded_records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Model record is not an object: {name}")
        entry = mod_data.get(name)
        if isinstance(entry, dict) and entry.get("date_removed"):
            counts["skipped_removed"] += 1
            entry = None
        if isinstance(entry, dict):
            created = entry.get("date_added")
            modified = entry.get("date_modified")
            if not isinstance(created, int):
                counts["left_empty"] += 1
                continue
            metadata = dict(record.get("metadata") or {})
            metadata["created_at"] = created
            metadata["updated_at"] = modified if isinstance(modified, int) else created
            record["metadata"] = metadata
            counts["from_upstream"] += 1
        elif record.get("metadata"):
            counts["kept_local"] += 1
        else:
            counts["left_empty"] += 1
    return seeded_records, counts


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate or apply metadata seeding for a legacy image-generation category file."""
    parsed_arguments = _parse_arguments(arguments)
    source_payload = json.loads(parsed_arguments.input.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("Category JSON must be a top-level object keyed by model name.")
    with urllib.request.urlopen(parsed_arguments.mod_data_url) as response:
        mod_data = json.load(response)
    if not isinstance(mod_data, dict):
        raise ValueError("model_mod_data.json must be a top-level object keyed by model name.")
    seeded_payload, counts = seed_image_model_metadata(source_payload, mod_data)
    summary = ", ".join(f"{key}={value}" for key, value in counts.items())
    if parsed_arguments.apply:
        atomic_write_json(parsed_arguments.input, seeded_payload, ensure_ascii=False)
        print(f"Seeded metadata in {parsed_arguments.input} ({summary})")
    else:
        print(f"Dry run: {summary}; re-run with --apply to persist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
