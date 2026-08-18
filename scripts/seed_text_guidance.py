"""Generate the packaged text guidance bootstrap catalog from canonical text records.

The packaged catalog ships as review scaffolding: one prompt contract per distinct legacy
``instruct_format`` label plus an assignment for every model that carries one. The same heuristic
backs the ``/text_generation/guidance/migration/preview`` route, so a deployment that would rather
review the proposal than ship a generated file can submit it through the pending queue instead.

The generated artifact is a bootstrap, never a live catalog: it is produced against a throwaway
store so this script cannot mutate a deployment's published guidance.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from horde_model_reference.cli.text_guidance_report import generate_text_guidance_report
from horde_model_reference.text_backend_names import has_legacy_text_backend_prefix
from horde_model_reference.text_guidance import (
    GuidanceRecordMetadata,
    TextGuidanceAssignment,
    TextGuidanceCatalog,
    TextGuidanceCatalogMetadata,
    TextUsageProfile,
)
from horde_model_reference.text_guidance_migration import build_legacy_migration_change_set
from horde_model_reference.text_guidance_store import TextGuidanceStore

_DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "horde_model_reference" / "data" / "guidance" / "catalog.json"
)
_SEED_EDITOR_ID = "text-guidance-seed"


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse seed command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate the packaged text guidance bootstrap catalog.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--primary-api-url", help="PRIMARY base URL including its /api root.")
    source_group.add_argument("--input", type=Path, help="Saved canonical text_generation reference JSON.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH)
    parser.add_argument("--readme", type=Path, help="Markdown report path; defaults to README.md beside --output.")
    parser.add_argument(
        "--timestamp",
        type=int,
        help="Unix seconds stamped on every generated record; omit to emit null timestamps.",
    )
    return parser.parse_args(arguments)


def _load_text_records(*, primary_api_url: str | None, input_path: Path | None) -> dict[str, dict[str, object]]:
    """Return canonical text records keyed by exact model name, sorted for reproducible output."""
    if input_path is not None:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        primary_root = str(primary_api_url).rstrip("/")
        response = httpx.get(f"{primary_root}/model_references/v2/text_generation", timeout=60.0)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Text generation reference payload must be an object keyed by model name.")
    return {
        str(model_name): record
        for model_name, record in sorted(payload.items(), key=lambda item: str(item[0]))
        if isinstance(record, dict) and not has_legacy_text_backend_prefix(str(model_name))
    }


def _stamped_metadata(metadata: GuidanceRecordMetadata, timestamp: int | None) -> GuidanceRecordMetadata:
    """Return record metadata with wall-clock fields pinned to the requested value."""
    return metadata.model_copy(update={"created_at": timestamp, "updated_at": timestamp})


def _pin_timestamps(catalog: TextGuidanceCatalog, timestamp: int | None) -> TextGuidanceCatalog:
    """Return a catalog whose timestamps do not depend on when the seed was generated."""
    profiles: dict[str, TextUsageProfile] = {
        profile_id: profile.model_copy(update={"metadata": _stamped_metadata(profile.metadata, timestamp)})
        for profile_id, profile in catalog.profiles.items()
    }
    assignments: dict[str, TextGuidanceAssignment] = {
        model_name: assignment.model_copy(update={"metadata": _stamped_metadata(assignment.metadata, timestamp)})
        for model_name, assignment in catalog.assignments.items()
    }
    return catalog.model_copy(
        update={
            "metadata": TextGuidanceCatalogMetadata(
                schema_version=catalog.metadata.schema_version,
                revision=catalog.metadata.revision,
                updated_at=timestamp,
            ),
            "profiles": profiles,
            "assignments": assignments,
        },
    )


def build_bootstrap_catalog(
    records: Mapping[str, Mapping[str, object]],
    *,
    timestamp: int | None,
) -> TextGuidanceCatalog:
    """Build the seeded catalog for the given canonical records without touching a live store."""
    preview = build_legacy_migration_change_set(
        records,
        existing_profiles=[],
        current_assignment=lambda _model_name: None,
    )
    with tempfile.TemporaryDirectory(prefix="text-guidance-seed-") as temporary_root:
        store = TextGuidanceStore(root_path=Path(temporary_root))
        if preview.change_set is None:
            return _pin_timestamps(store.export(), timestamp)
        catalog = store.apply_change_set(
            preview.change_set,
            canonical_model_names=set(records),
            editor_id=_SEED_EDITOR_ID,
        )
    return _pin_timestamps(catalog, timestamp)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate the bootstrap catalog and its Markdown report."""
    parsed_arguments = _parse_arguments(arguments)
    records = _load_text_records(
        primary_api_url=parsed_arguments.primary_api_url,
        input_path=parsed_arguments.input,
    )
    catalog = build_bootstrap_catalog(records, timestamp=parsed_arguments.timestamp)
    output_path: Path = parsed_arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path: Path = parsed_arguments.readme or output_path.parent / "README.md"
    generate_text_guidance_report(catalog.model_dump(mode="json"), output_path=readme_path)
    print(f"{output_path}\n{readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
