"""Generate deterministic Markdown from the text guidance catalog export."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import httpx

from horde_model_reference.text_guidance import (
    GuidanceProfileKind,
    TextGuidanceCatalog,
    TextPromptContract,
)

__all__ = ["generate_text_guidance_report", "main"]

_GENERATED_WARNING = "<!-- GENERATED FILE: edit text guidance through the v2 API, not this Markdown. -->"


def _escape_table_cell(value: object) -> str:
    """Return text safe for a single Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_list(title: str, values: list[str]) -> list[str]:
    """Render one optional prose list."""
    if not values:
        return []
    return [f"### {title}", "", *[f"- {value}" for value in values], ""]


def _profile_markdown(catalog: TextGuidanceCatalog) -> str:
    """Render the complete catalog in stable profile and assignment order."""
    lines = [
        _GENERATED_WARNING,
        "# Text Model Usage Guidance",
        "",
        f"Schema: `{catalog.metadata.schema_version}` · Catalog revision: `{catalog.metadata.revision}`",
        "",
        "## Catalog",
        "",
        "| Profile | Kind | Assigned models | Status |",
        "|---|---|---:|---|",
    ]
    assignment_counts: dict[str, int] = {}
    for assignment in catalog.assignments.values():
        for profile_id in [assignment.primary_profile_id, *assignment.supplemental_profile_ids]:
            assignment_counts[profile_id] = assignment_counts.get(profile_id, 0) + 1
    for profile_id in sorted(
        catalog.profiles,
        key=lambda current_id: (
            catalog.profiles[current_id].kind is GuidanceProfileKind.USAGE_RECIPE,
            catalog.profiles[current_id].display_name.casefold(),
        ),
    ):
        profile = catalog.profiles[profile_id]
        lines.append(
            f"| `{_escape_table_cell(profile_id)}` | "
            f"{_escape_table_cell(profile.kind.value.replace('_', ' ').title())} | "
            f"{assignment_counts.get(profile_id, 0)} | "
            f"{'Deprecated' if profile.deprecated else 'Published'} |",
        )

    for profile_id in sorted(catalog.profiles):
        profile = catalog.profiles[profile_id]
        lines.extend(["", f"## {profile.display_name}", "", profile.summary, ""])
        if profile.aliases:
            lines.extend([f"Aliases: {', '.join(f'`{alias}`' for alias in profile.aliases)}", ""])
        lines.extend(["### For users", "", profile.user.overview or "No user overview has been published.", ""])
        lines.extend(_render_list("Useful for", profile.user.use_cases))
        lines.extend(_render_list("Prompt tips", profile.user.tips))
        lines.extend(_render_list("Caveats", profile.user.caveats))
        lines.extend(
            ["### For developers", "", profile.developer.overview or "No developer overview has been published.", ""]
        )
        if isinstance(profile, TextPromptContract):
            stop_sequences = ", ".join(f"`{stop}`" for stop in profile.stop_sequences) or "None documented"
            lines.extend(
                [
                    f"Interaction modes: {', '.join(mode.value for mode in profile.interaction_modes)}",
                    "",
                    f"Stop sequences: {stop_sequences}",
                    "",
                ],
            )
            for template in profile.templates:
                lines.extend(
                    [
                        f"#### {template.name}",
                        "",
                        f"Syntax: `{template.syntax_name or template.syntax.value}`",
                        "",
                        "```text",
                        template.template,
                        "```",
                        "",
                    ],
                )
        assigned_models = [
            assignment.model_name
            for assignment in catalog.assignments.values()
            if profile_id in [assignment.primary_profile_id, *assignment.supplemental_profile_ids]
        ]
        lines.extend(_render_list("Assigned models", sorted(assigned_models)))
    return "\n".join(lines).rstrip() + "\n"


def generate_text_guidance_report(export_payload: dict[str, object], *, output_path: Path) -> Path:
    """Validate an export and write its deterministic human-readable Markdown."""
    catalog = TextGuidanceCatalog.model_validate(export_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_profile_markdown(catalog), encoding="utf-8")
    return output_path


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse report generator command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Markdown from first-class text guidance data.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--primary-api-url", help="PRIMARY base URL including its /api root.")
    source_group.add_argument("--input", type=Path, help="Previously saved guidance export JSON.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate guidance Markdown from a live PRIMARY or saved export."""
    parsed_arguments = _parse_arguments(arguments)
    if parsed_arguments.input is not None:
        export_payload = json.loads(parsed_arguments.input.read_text(encoding="utf-8"))
    else:
        primary_api_url = str(parsed_arguments.primary_api_url).rstrip("/")
        response = httpx.get(
            f"{primary_api_url}/model_references/v2/text_generation/guidance/export",
            timeout=30.0,
        )
        response.raise_for_status()
        export_payload = response.json()
    generated = generate_text_guidance_report(export_payload, output_path=parsed_arguments.output)
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
