"""Helpers for parsing and writing legacy text generation CSV files.

Includes the canonical CSV→legacy-dict conversion that replicates convert.py's
algorithm, plus CSV write-back and reverse-conversion functions for maintaining
the CSV as the source of truth through write operations.

All backends that need to serve or compare text generation legacy data
should use ``csv_rows_to_legacy_dict`` rather than rolling their own conversion.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast

from loguru import logger

from horde_model_reference.text_model_write_processor import TextModelWriteProcessor, _get_defaults

_ALLOWED_PRIMITIVE_TYPES = (int, float, str, bool)

SettingsPrimitive = int | float | str | bool
SettingsValue = SettingsPrimitive | list[SettingsPrimitive]
SettingsMapping = dict[str, SettingsValue]

TEXT_CSV_FIELDNAMES: list[str] = [
    "name",
    "parameters_bn",
    "display_name",
    "url",
    "baseline",
    "description",
    "style",
    "tags",
    "instruct_format",
    "settings",
]


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
    instruct_format: str
    settings: SettingsMapping | None
    display_name: str


@dataclass(frozen=True)
class TextCSVIssue:
    """Validation issue encountered while parsing a CSV row."""

    row_identifier: str
    message: str


def parse_legacy_text_csv(stream: IO[str]) -> tuple[list[TextCSVRow], list[TextCSVIssue]]:
    """Parse legacy text-generation CSV data from a text stream into structured rows."""
    rows: list[TextCSVRow] = []
    issues: list[TextCSVIssue] = []

    reader = csv.DictReader(stream)
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
            instruct_format=(raw_row.get("instruct_format") or ""),
            settings=settings,
            display_name=(raw_row.get("display_name") or ""),
        )
        rows.append(row)

    return rows, issues


def parse_legacy_text_csv_file(csv_path: Path) -> tuple[list[TextCSVRow], list[TextCSVIssue]]:
    """Parse legacy text-generation CSV data from a file path into structured rows."""
    if not csv_path.exists():
        return [], []

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        return parse_legacy_text_csv(csvfile)


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


def csv_rows_to_legacy_dict(
    rows: list[TextCSVRow],
    *,
    with_backend_prefixes: bool = True,
) -> dict[str, Any]:
    """Convert parsed CSV rows to legacy dict format, replicating convert.py exactly.

    This is the single canonical implementation of the CSV→legacy-dict conversion.
    Field ordering, defaults merging, empty-value filtering, tag generation, and
    backend prefix duplication all match the upstream convert.py algorithm.

    Args:
        rows: Parsed CSV rows from ``parse_legacy_text_csv``.
        with_backend_prefixes: If True, generate 3 entries per model
            (base, aphrodite/, koboldcpp/) matching db.json format.
            If False, generate 1 entry per base model only.

    Returns:
        Legacy dict matching convert.py output format.
    """
    defaults = dict(_get_defaults())
    data: dict[str, Any] = {}

    for csv_row in rows:
        name = csv_row.name

        if "://" in name:
            logger.warning(f"Skipping URL-shaped model name: {name!r}")
            continue

        model_name = name.split("/")[1] if "/" in name else name

        # Build the row dict with the same key order as CSV columns
        # (after popping name and parameters_bn, which convert.py does)
        row: dict[str, Any] = {}

        row["parameters"] = csv_row.parameters

        # Tags: merge CSV tags + style + size bucket, sorted
        tags = set(csv_row.tags) if csv_row.tags else set()
        if csv_row.style:
            tags.add(csv_row.style)
        tags.add(f"{round(csv_row.parameters_bn, 0):.0f}B")
        row["tags"] = sorted(tags)

        row["settings"] = dict(csv_row.settings) if csv_row.settings is not None else {}

        # Auto-generate display_name if not provided
        if csv_row.display_name:
            row["display_name"] = csv_row.display_name
        else:
            row["display_name"] = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

        row["url"] = csv_row.url
        row["baseline"] = csv_row.baseline
        row["description"] = csv_row.description
        row["style"] = csv_row.style

        row["instruct_format"] = csv_row.instruct_format

        # Remove empty values — matches convert.py: {k: v for k, v in row.items() if v}
        row = {k: v for k, v in row.items() if v}

        if with_backend_prefixes:
            for key_format in ["{name}", "aphrodite/{name}", "koboldcpp/{model_name}"]:
                key = key_format.format(name=name, model_name=model_name)
                data[key] = {"name": key, "model_name": model_name, **defaults, **row}
        else:
            data[name] = {"name": name, "model_name": model_name, **defaults, **row}

    return data


def _parameters_to_bn_str(parameters: int) -> str:
    """Convert integer parameter count to minimal billions string for CSV.

    Uses simplest representation: 3000000000 → "3", 560000000 → "0.56".

    Args:
        parameters: Integer parameter count.

    Returns:
        Minimal string representation in billions.
    """
    bn = parameters / 1_000_000_000
    if bn == int(bn):
        return str(int(bn))
    return f"{bn:.10f}".rstrip("0").rstrip(".")


def legacy_record_to_csv_row(name: str, record: dict[str, Any]) -> TextCSVRow:
    """Reverse-convert a db.json-format record to a TextCSVRow.

    Strips auto-generated tags (style + size bucket) and reverses the
    parameter conversion so the CSV row round-trips through convert.py.

    Args:
        name: The base model name (e.g., "Org/Model-7B").
        record: A single model record from the legacy dict (db.json format).

    Returns:
        A TextCSVRow suitable for writing to CSV.
    """
    parameters = int(record.get("parameters", 0) or 0)
    parameters_bn = parameters / 1_000_000_000

    style = str(record.get("style", "") or "")

    # Strip auto-generated tags: the size bucket and style tag
    raw_tags: list[str] = record.get("tags", []) or []
    if not isinstance(raw_tags, list):
        raw_tags = []

    # Detect default-only style: convert.py/csv_rows_to_legacy_dict add explicit
    # styles to tags before applying defaults.json, so a style present on the
    # record but absent from tags was only injected by the defaults system.
    defaults = _get_defaults()
    default_style = str(defaults.get("style", "") or "")
    if style and style == default_style and style not in raw_tags:
        style = ""

    size_tag = f"{round(parameters_bn, 0):.0f}B"
    tags = [t for t in raw_tags if t != size_tag and t != style]

    # Omit display_name if it matches the auto-generated value
    model_name = TextModelWriteProcessor.extract_model_name(name)
    auto_display = TextModelWriteProcessor.generate_display_name(model_name)
    display_name = str(record.get("display_name", "") or "")
    if display_name == auto_display:
        display_name = ""

    settings_raw = record.get("settings")
    settings: SettingsMapping | None = None
    if isinstance(settings_raw, dict) and settings_raw:
        settings = cast(SettingsMapping, settings_raw)

    return TextCSVRow(
        name=name,
        parameters_bn=parameters_bn,
        parameters=parameters,
        description=str(record.get("description", "") or ""),
        version=str(record.get("version", "") or ""),
        style=style,
        nsfw=bool(record.get("nsfw", False)),
        baseline=str(record.get("baseline", "") or ""),
        url=str(record.get("url", "") or ""),
        tags=tags,
        instruct_format=str(record.get("instruct_format", "") or ""),
        settings=settings,
        display_name=display_name,
    )


def write_legacy_text_csv(rows: list[TextCSVRow], csv_path: Path) -> None:
    """Write TextCSVRow list to a CSV file in upstream models.csv format.

    Args:
        rows: The rows to write.
        csv_path: Path to write the CSV file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TEXT_CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()

    for row in rows:
        csv_dict: dict[str, str] = {
            "name": row.name,
            "parameters_bn": _parameters_to_bn_str(row.parameters),
            "display_name": row.display_name,
            "url": row.url,
            "baseline": row.baseline,
            "description": row.description,
            "style": row.style,
            "tags": ",".join(row.tags) if row.tags else "",
            "instruct_format": row.instruct_format,
            "settings": json.dumps(row.settings, separators=(",", ": ")) if row.settings else "",
        }
        writer.writerow(csv_dict)

    # csv module always writes \r\n terminators; normalize to \n for git compatibility
    csv_path.write_text(output.getvalue().replace("\r\n", "\n"), encoding="utf-8", newline="")
