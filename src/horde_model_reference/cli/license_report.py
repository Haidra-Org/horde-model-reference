"""Generate deterministic human-readable licensing reports from a PRIMARY export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference.licensing import LicenseDefinition, ModelLicensing, PermissionStatus

__all__ = ["generate_license_report", "main"]

_GENERATED_WARNING = "<!-- GENERATED FILE: edit licensing data through the v2 API, not this Markdown. -->"


class _ReportAsset(BaseModel):
    """Represents the subset of an exported asset required by the report renderer."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    asset_kind: str
    asset_identifier: str
    display_name: str
    category: str | None = None
    source_url: str | None = None
    version: str | None = None
    licensing: ModelLicensing
    notes: str | None = None


class _ReportExport(BaseModel):
    """Represents a validated licensing export accepted by the report renderer."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: int
    metadata: dict[str, Any]
    licenses: list[LicenseDefinition]
    assets: list[_ReportAsset]
    summary: dict[str, Any] = Field(default_factory=dict)


def _escape_table_cell(text: object) -> str:
    """Return text safe for a single Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _permission_label(permission_status: PermissionStatus) -> str:
    """Return a readable label for one permission status."""
    return permission_status.value.replace("_", " ").title()


def _definition_markdown(export: _ReportExport) -> str:
    """Return the generated definition and dataset overview Markdown."""
    metadata = export.metadata
    lines = [
        _GENERATED_WARNING,
        "# Licensing Reference",
        "",
        f"Data schema: `{export.schema_version}` · Dataset revision: `{metadata.get('revision', 'unknown')}`",
        "",
        "This report is an informational summary of reviewed licensing records. It is not legal advice.",
        "",
        "## Known licenses",
        "",
        "| Identifier | License | Commercial use | Redistribution | Obligations |",
        "|---|---|---|---|---|",
    ]
    for definition in sorted(export.licenses, key=lambda current_definition: current_definition.license_id.casefold()):
        obligations = ", ".join(obligation.value for obligation in definition.obligations) or "None recorded"
        license_link = f"[{definition.name}]({definition.canonical_url})"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape_table_cell(definition.license_id)}`",
                    _escape_table_cell(license_link),
                    _permission_label(definition.commercial_use),
                    _permission_label(definition.redistribution),
                    _escape_table_cell(obligations),
                ],
            )
            + " |",
        )
    return "\n".join(lines) + "\n"


def _model_markdown(export: _ReportExport) -> str:
    """Return the generated model and file licensing Markdown."""
    model_assets = [asset for asset in export.assets if asset.asset_kind == "model"]
    lines = [
        _GENERATED_WARNING,
        "# Model Weights - License Reference",
        "",
        "| Category | Model | License expression | Commercial use | Redistribution |",
        "|---|---|---|---|---|",
    ]
    for asset in sorted(
        model_assets,
        key=lambda current_asset: (
            current_asset.category or "",
            current_asset.display_name.casefold(),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(asset.category or "unknown"),
                    _escape_table_cell(asset.display_name),
                    f"`{_escape_table_cell(asset.licensing.license_expression)}`",
                    _permission_label(asset.licensing.commercial_use),
                    _permission_label(asset.licensing.redistribution),
                ],
            )
            + " |",
        )
        for file_name, file_assignment in sorted(asset.licensing.files.items()):
            lines.append(
                "|  | ↳ `"
                + _escape_table_cell(file_name)
                + "` | `"
                + _escape_table_cell(file_assignment.license_expression)
                + "` | "
                + _permission_label(file_assignment.commercial_use)
                + " | "
                + _permission_label(file_assignment.redistribution)
                + " |",
            )
    return "\n".join(lines) + "\n"


def _non_model_markdown(export: _ReportExport) -> str:
    """Return the generated custom-node and software licensing Markdown."""
    non_model_assets = [asset for asset in export.assets if asset.asset_kind != "model"]
    lines = [
        _GENERATED_WARNING,
        "# Software and Custom Nodes - License Reference",
        "",
        "| Kind | Asset | Version | License expression | Commercial use | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for asset in sorted(
        non_model_assets,
        key=lambda current_asset: (
            current_asset.asset_kind,
            current_asset.display_name.casefold(),
        ),
    ):
        display_name = (
            f"[{asset.display_name}]({asset.source_url})" if asset.source_url is not None else asset.display_name
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(asset.asset_kind),
                    _escape_table_cell(display_name),
                    _escape_table_cell(asset.version or "—"),
                    f"`{_escape_table_cell(asset.licensing.license_expression)}`",
                    _permission_label(asset.licensing.commercial_use),
                    _escape_table_cell(asset.notes or ""),
                ],
            )
            + " |",
        )
    return "\n".join(lines) + "\n"


def generate_license_report(export_payload: dict[str, Any], *, output_directory: Path) -> list[Path]:
    """Generate all human-readable reports from a licensing export.

    Args:
        export_payload: JSON-compatible response from ``GET /v2/licensing/export``.
        output_directory: Directory in which the generated Markdown files are written.

    Returns:
        Paths of the three generated Markdown files.
    """
    validated_export = _ReportExport.model_validate(export_payload)
    output_directory.mkdir(parents=True, exist_ok=True)
    rendered_files = {
        "README.md": _definition_markdown(validated_export),
        "models.md": _model_markdown(validated_export),
        "custom_nodes.md": _non_model_markdown(validated_export),
    }
    output_paths: list[Path] = []
    for file_name, rendered_markdown in rendered_files.items():
        output_path = output_directory / file_name
        output_path.write_text(rendered_markdown, encoding="utf-8")
        output_paths.append(output_path)
    return output_paths


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse report generator command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Markdown from first-class licensing data.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--primary-api-url", help="PRIMARY base URL including its /api root.")
    source_group.add_argument("--input", type=Path, help="Previously saved licensing export JSON.")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate licensing reports from a live PRIMARY or saved export."""
    parsed_arguments = _parse_arguments(arguments)
    if parsed_arguments.input is not None:
        export_payload = json.loads(parsed_arguments.input.read_text(encoding="utf-8"))
    else:
        primary_api_url = str(parsed_arguments.primary_api_url).rstrip("/")
        export_url = f"{primary_api_url}/model_references/v2/licensing/export"
        response = httpx.get(export_url, timeout=30.0)
        response.raise_for_status()
        export_payload = response.json()
    generated_paths = generate_license_report(export_payload, output_directory=parsed_arguments.output_directory)
    for generated_path in generated_paths:
        print(generated_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
