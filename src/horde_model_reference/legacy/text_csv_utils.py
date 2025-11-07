"""Helpers for parsing legacy text generation CSV files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

_ALLOWED_PRIMITIVE_TYPES = (int, float, str, bool)

SettingsPrimitive: TypeAlias = int | float | str | bool
SettingsValue: TypeAlias = SettingsPrimitive | list[SettingsPrimitive]
SettingsMapping: TypeAlias = dict[str, SettingsValue]


@dataclass(frozen=True)
class TextCSVRow:
    """Structured representation of a single legacy text CSV row."""

    name: str
    parameters_bn: float
    parameters: int
    description: str
    version: str
    style: str
    nsfw: bool
    baseline: str
    url: str
    tags: list[str]
    settings: SettingsMapping | None
    display_name: str


@dataclass(frozen=True)
class TextCSVIssue:
    """Validation issue encountered while parsing a CSV row."""

    row_identifier: str
    message: str


def parse_legacy_text_csv(csv_path: Path) -> tuple[list[TextCSVRow], list[TextCSVIssue]]:
    """Parse legacy text-generation CSV data into structured rows."""
    rows: list[TextCSVRow] = []
    issues: list[TextCSVIssue] = []
    if not csv_path.exists():
        return rows, issues

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for line_number, raw_row in enumerate(reader, start=2):
            raw_name = (raw_row.get("name") or "").strip()
            identifier = raw_name or f"<row {line_number}>"

            if not raw_name:
                issues.append(TextCSVIssue(identifier, "missing required 'name' field; row skipped"))
                continue

            parameters_bn_str = (raw_row.get("parameters_bn") or "").strip()
            if not parameters_bn_str:
                issues.append(TextCSVIssue(identifier, "missing parameters_bn; defaulting to 0"))
                parameters_bn = 0.0
            else:
                try:
                    parameters_bn = float(parameters_bn_str)
                except ValueError:
                    issues.append(TextCSVIssue(identifier, "invalid parameters_bn value; row skipped"))
                    continue

            parameters = int(parameters_bn * 1_000_000_000)

            tags_raw = raw_row.get("tags", "")
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

            settings_str = (raw_row.get("settings") or "").strip()
            if settings_str:
                try:
                    parsed_settings = json.loads(settings_str)
                except json.JSONDecodeError as exc:  # pragma: no cover - error path exercised via tests
                    issues.append(TextCSVIssue(identifier, f"invalid settings JSON: {exc.msg}; row skipped"))
                    continue
                if not _settings_value_types_valid(parsed_settings):
                    issues.append(
                        TextCSVIssue(
                            identifier,
                            "invalid settings structure; only primitive values or lists thereof are supported",
                        )
                    )
                    continue
                settings = cast(SettingsMapping, parsed_settings)
            else:
                settings = None

            row = TextCSVRow(
                name=raw_name,
                parameters_bn=parameters_bn,
                parameters=parameters,
                description=(raw_row.get("description") or ""),
                version=(raw_row.get("version") or ""),
                style=(raw_row.get("style") or ""),
                nsfw=(raw_row.get("nsfw") or "").strip().lower() == "true",
                baseline=(raw_row.get("baseline") or ""),
                url=(raw_row.get("url") or ""),
                tags=tags,
                settings=settings,
                display_name=(raw_row.get("display_name") or ""),
            )
            rows.append(row)

    return rows, issues


def _settings_value_types_valid(settings: object) -> bool:
    """Validate that ``settings`` matches the supported flat structure."""
    if settings is None:
        return True
    if not isinstance(settings, dict):
        return False
    for value in settings.values():
        if isinstance(value, _ALLOWED_PRIMITIVE_TYPES):
            continue
        if isinstance(value, list):
            if not all(isinstance(item, _ALLOWED_PRIMITIVE_TYPES) for item in value):
                return False
            continue
        return False
    return True
