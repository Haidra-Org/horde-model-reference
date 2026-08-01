"""Create or submit the one-time v2 model-licensing backfill from the audited migration policy.

The legacy Git repositories are never read or written by this migration.  The script
reads the canonical PRIMARY v2 API, translates the immutable 2026 audit evidence into
typed assignments, and gives every otherwise unaudited model an explicit fail-closed
``NOASSERTION`` conclusion.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import quote

import httpx
from license_expression import Licensing

from horde_model_reference.licensing import (
    LicenseEvidence,
    ModelLicensing,
    aggregate_permission_statuses,
    unknown_model_licensing,
)
from horde_model_reference.licensing_store import LicensingStore

_DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "r2_sync" / "redistribution_policy.json"
_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "src" / "horde_model_reference" / "data" / "licenses"

_BLOCKED_LICENSE_EXPRESSIONS: dict[tuple[str, str], str] = {
    ("controlnet", "control_canny_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("controlnet", "control_depth_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("controlnet", "control_hed_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("controlnet", "control_openpose_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("controlnet", "control_fakescribbles_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("controlnet", "control_scribble_sd2"): "LicenseRef-thibaud-controlnet-sd21",
    ("esrgan", "RealESRGAN_x2plus"): "LicenseRef-S-Lab-1.0",
    ("esrgan", "4x_AnimeSharp"): "CC-BY-NC-SA-4.0",
    ("codeformer", "CodeFormers"): "LicenseRef-S-Lab-1.0",
    ("controlnet_annotator", "annotator_openpose"): "LicenseRef-OpenPose-NonCommercial",
    ("controlnet_annotator", "annotator_seg"): "LicenseRef-Research-Only",
    ("controlnet_annotator", "annotator_pidinet"): "LicenseRef-Research-Only",
    ("controlnet_annotator", "annotator_teed"): "CC-BY-NC-SA-4.0",
    ("controlnet_annotator", "annotator_depth_anything_v2"): "CC-BY-NC-4.0",
}


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse migration command-line arguments."""
    parser = argparse.ArgumentParser(description="Backfill explicit licensing into canonical v2 model records.")
    parser.add_argument("--primary-api-url", required=True, help="PRIMARY base URL including /api.")
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true", help="Queue each generated PUT through the v2 API.")
    parser.add_argument("--apikey", help="Required with --submit; must be a pending-queue requestor key.")
    return parser.parse_args(arguments)


def _license_ids(expression: str) -> tuple[str, ...]:
    """Return sorted atomic identifiers from a validated SPDX expression."""
    parsed_expression = Licensing().parse(expression, validate=False)
    if parsed_expression is None:
        raise ValueError(f"Invalid license expression: {expression}")
    return tuple(sorted(str(symbol) for symbol in parsed_expression.symbols))


def _known_assignment(
    *,
    expression: str,
    policy_record: dict[str, object],
    catalog: LicensingStore,
) -> ModelLicensing:
    """Create a reviewed assignment from one immutable audit-policy record."""
    identifiers = _license_ids(expression)
    definitions = []
    for license_id in identifiers:
        definition = catalog.get_definition(license_id)
        if definition is None:
            raise ValueError(f"Migration references an unknown definition: {license_id}")
        definitions.append(definition)
    reviewed_at_value = policy_record.get("reviewed_at")
    reviewed_at = date.fromisoformat(str(reviewed_at_value)) if reviewed_at_value else None
    evidence_sources = policy_record.get("evidence", [])
    if not isinstance(evidence_sources, list):
        raise ValueError("Migration policy evidence must be an array")
    evidence = tuple(LicenseEvidence(source=str(source), checked_at=reviewed_at) for source in evidence_sources)
    obligations = tuple(sorted({obligation for definition in definitions for obligation in definition.obligations}))
    return ModelLicensing(
        license_expression=expression,
        license_ids=identifiers,
        commercial_use=aggregate_permission_statuses(
            tuple(definition.commercial_use for definition in definitions),
        ),
        redistribution=aggregate_permission_statuses(
            tuple(definition.redistribution for definition in definitions),
        ),
        obligations=obligations,
        attribution=str(policy_record["attribution"]) if policy_record.get("attribution") else None,
        evidence=evidence,
        reviewed_by=(
            str(policy_record["reviewed_by"]) if policy_record.get("reviewed_by") else "hordelib-license-audit"
        ),
        reviewed_at=reviewed_at,
        notes=str(policy_record["note"]) if policy_record.get("note") else None,
    )


def _audit_assignments(policy_path: Path) -> dict[tuple[str, str], ModelLicensing]:
    """Return reviewed model assignments derived from the immutable audit migration input."""
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    catalog = LicensingStore(root_path=_BOOTSTRAP_PATH, bootstrap_path=_BOOTSTRAP_PATH, writable=False)
    assignments: dict[tuple[str, str], ModelLicensing] = {}
    for policy_record in policy_payload["models"]:
        category = str(policy_record["category"])
        model_name = str(policy_record["name"])
        expression_value = policy_record.get("license_expression")
        expression = (
            str(expression_value) if expression_value else _BLOCKED_LICENSE_EXPRESSIONS.get((category, model_name))
        )
        if expression is None:
            continue
        assignments[(category, model_name)] = _known_assignment(
            expression=expression,
            policy_record=policy_record,
            catalog=catalog,
        )
    return assignments


def _load_primary_records(primary_api_url: str) -> dict[str, dict[str, dict[str, object]]]:
    """Return every current canonical v2 category payload from the PRIMARY API."""
    primary_root = primary_api_url.rstrip("/")
    categories_response = httpx.get(f"{primary_root}/model_references/v2/model_categories", timeout=30.0)
    categories_response.raise_for_status()
    records: dict[str, dict[str, dict[str, object]]] = {}
    for category in categories_response.json():
        category_response = httpx.get(f"{primary_root}/model_references/v2/{category}", timeout=60.0)
        if category_response.status_code == 404:
            records[str(category)] = {}
            continue
        category_response.raise_for_status()
        records[str(category)] = category_response.json()
    return records


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate the full backfill and optionally queue it through canonical model PUT routes."""
    parsed_arguments = _parse_arguments(arguments)
    if parsed_arguments.submit and not parsed_arguments.apikey:
        raise SystemExit("--apikey is required with --submit")
    assignments = _audit_assignments(parsed_arguments.policy)
    primary_records = _load_primary_records(parsed_arguments.primary_api_url)
    backfill_records: list[dict[str, object]] = []
    primary_root = str(parsed_arguments.primary_api_url).rstrip("/")
    for category in sorted(primary_records):
        category_records = primary_records[category]
        for model_name in sorted(category_records, key=str.casefold):
            model_record = category_records[model_name]
            assignment = assignments.get((category, model_name)) or unknown_model_licensing(
                note="No model-specific conclusion was present in the 2026 licensing audit.",
            )
            updated_record = {**model_record, "licensing": assignment.model_dump(mode="json", exclude_none=True)}
            backfill_records.append({"category": category, "model_name": model_name, "record": updated_record})
            if parsed_arguments.submit:
                update_url = (
                    f"{primary_root}/model_references/v2/{quote(category, safe='')}/model/{quote(model_name, safe='')}"
                )
                update_response = httpx.put(
                    update_url,
                    headers={"apikey": parsed_arguments.apikey},
                    json=updated_record,
                    timeout=60.0,
                )
                update_response.raise_for_status()
    parsed_arguments.output.parent.mkdir(parents=True, exist_ok=True)
    parsed_arguments.output.write_text(
        json.dumps({"schema_version": 1, "records": backfill_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
