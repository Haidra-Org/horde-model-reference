"""Safely rename one canonical v2 model-record key in a PRIMARY data file."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from horde_model_reference.util import atomic_write_json


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse command line arguments for a guarded, one-record key migration."""
    parser = argparse.ArgumentParser(description="Rename one canonical v2 model-record key atomically.")
    parser.add_argument("--input", type=Path, required=True, help="PRIMARY category JSON file to migrate.")
    parser.add_argument("--old-name", required=True, help="Existing top-level model-record key.")
    parser.add_argument("--new-name", required=True, help="Replacement canonical model-record key.")
    parser.add_argument("--apply", action="store_true", help="Persist the rename. Omit for a validation-only dry run.")
    return parser.parse_args(arguments)


def rename_model_record_key(
    category_records: dict[str, object],
    *,
    old_name: str,
    new_name: str,
) -> dict[str, object]:
    """Return a renamed category mapping after enforcing identity invariants."""
    if old_name not in category_records:
        raise ValueError(f"Source model key does not exist: {old_name}")
    if new_name in category_records:
        raise ValueError(f"Destination model key already exists: {new_name}")
    record = category_records[old_name]
    if not isinstance(record, dict):
        raise ValueError(f"Source model record is not an object: {old_name}")
    if record.get("name") != new_name:
        raise ValueError(
            f"Source record name must match the replacement key: {record.get('name')!r} != {new_name!r}",
        )

    renamed_records = dict(category_records)
    del renamed_records[old_name]
    renamed_records[new_name] = record
    return renamed_records


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate or atomically apply a one-record canonical key rename."""
    parsed_arguments = _parse_arguments(arguments)
    source_payload = json.loads(parsed_arguments.input.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise ValueError("Category JSON must be a top-level object keyed by model name.")
    renamed_payload = rename_model_record_key(
        source_payload,
        old_name=parsed_arguments.old_name,
        new_name=parsed_arguments.new_name,
    )
    if parsed_arguments.apply:
        atomic_write_json(parsed_arguments.input, renamed_payload, ensure_ascii=False)
        print(f"Renamed {parsed_arguments.old_name} -> {parsed_arguments.new_name} in {parsed_arguments.input}")
    else:
        print(
            f"Validated rename {parsed_arguments.old_name} -> {parsed_arguments.new_name}; "
            "re-run with --apply to persist it.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
