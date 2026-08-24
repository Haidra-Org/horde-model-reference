"""Serialize text generation model data to upstream-compatible CSV and db.json.

Replicates the exact conversion logic from the upstream repository's convert.py,
ensuring that synced files are byte-compatible with what convert.py would produce
from the same CSV input.

The pipeline is:
    PRIMARY API records -> reverse-convert to CSV rows -> forward-convert to db.json

The forward conversion is a faithful reproduction of convert.py's algorithm,
guaranteeing field ordering, default merging, and value transformations match.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from loguru import logger

from horde_model_reference.text_backend_names import get_model_name_variants, has_legacy_text_backend_prefix
from horde_model_reference.text_model_write_processor import (
    TextModelWriteProcessor,
    _get_defaults,
    _get_generation_params,
)

# CSV column order matching the upstream models.csv header exactly.
# This order determines the key insertion order in parsed row dicts,
# which in turn controls the field order in db.json records for
# keys not present in defaults.json.
UPSTREAM_TEXT_CSV_FIELDNAMES: list[str] = [
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

# Durable text metadata columns. These exist in the PRIMARY-local CSV storage but not in
# the upstream models.csv schema, so they are only written when explicitly enabled.
TEXT_METADATA_CSV_FIELDNAMES: list[str] = [
    "context_window",
    "interaction_modes",
    "capabilities",
]

TEXT_CSV_FIELDNAMES: list[str] = [*UPSTREAM_TEXT_CSV_FIELDNAMES, *TEXT_METADATA_CSV_FIELDNAMES]

LegacyRecordDict = dict[str, Any]

# Fields where _record_to_csv_row always produces a value derived from PRIMARY data.
# An empty string means "PRIMARY says this is empty", not "unknown/missing".
# Fields NOT in this set (like instruct_format) are CSV-only metadata that
# PRIMARY may not carry - empty PRIMARY values fall back to existing CSV.
_PRIMARY_AUTHORITATIVE_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "parameters_bn",
        "display_name",
        "url",
        "baseline",
        "description",
        "style",
        "tags",
        "settings",
    }
)


@dataclass(frozen=True)
class TextGenerationSyncArtifacts:
    """Output artifacts from text generation serialization.

    Attributes:
        csv_content: The models.csv file content as a string.
        json_content: The db.json file content as a string.

    """

    csv_content: str
    json_content: str


class TextGenerationSerializer:
    """Serialize text generation records to upstream-compatible CSV and db.json.

    Produces two coordinated outputs: a models.csv and a db.json that are
    byte-compatible with the upstream repository's convert.py output.

    The serialization pipeline:
    1. Strip backend-prefixed entries to get base records
    2. Reverse-convert base records to CSV row format
    3. Merge with existing CSV (preserving row order for unchanged models)
    4. Forward-convert merged CSV rows to db.json (replicating convert.py)
    """

    def __init__(self, *, include_text_metadata_columns: bool = False) -> None:
        """Initialize with bundled defaults and generation params.

        Args:
            include_text_metadata_columns: Whether to carry ``context_window``,
                ``interaction_modes``, and ``capabilities`` into the synced CSV and
                db.json. The upstream schema does not define these columns, so they
                are withheld unless the upstream repository has been extended.

        """
        self._defaults = _get_defaults()
        self._generation_params = _get_generation_params()
        self._include_text_metadata_columns = include_text_metadata_columns
        self._fieldnames: list[str] = list(
            TEXT_CSV_FIELDNAMES if include_text_metadata_columns else UPSTREAM_TEXT_CSV_FIELDNAMES
        )
        self._primary_authoritative_fields: frozenset[str] = (
            _PRIMARY_AUTHORITATIVE_FIELDS | frozenset(TEXT_METADATA_CSV_FIELDNAMES)
            if include_text_metadata_columns
            else _PRIMARY_AUTHORITATIVE_FIELDS
        )

    @property
    def fieldnames(self) -> list[str]:
        """The CSV column names this serializer writes, in order.

        Returns:
            A copy of the ordered column list.

        """
        return list(self._fieldnames)

    def serialize(
        self,
        *,
        primary_base_records: dict[str, LegacyRecordDict],
        existing_csv_path: Path | None = None,
        existing_csv_content: str | None = None,
        preserve_absent_existing_records: bool = True,
    ) -> TextGenerationSyncArtifacts:
        """Produce models.csv and db.json from PRIMARY base records.

        Args:
            primary_base_records: Model records keyed by base name (no backend prefixes).
            existing_csv_path: Path to existing models.csv in the cloned repo.
                If provided and exists, row order of unchanged models is preserved.
            existing_csv_content: Existing models.csv content fetched before a repository
                is cloned. This is mutually exclusive with ``existing_csv_path``.
            preserve_absent_existing_records: Whether rows missing from PRIMARY are
                retained. Compatibility callers default to retaining them; the GitHub
                sync disables this so the merge cannot conceal source divergence.

        Returns:
            Artifacts containing CSV and JSON file contents.

        """
        if existing_csv_path is not None and existing_csv_content is not None:
            raise ValueError("Provide either existing_csv_path or existing_csv_content, not both")

        base_records: dict[str, LegacyRecordDict] = {}
        for name, record in primary_base_records.items():
            if has_legacy_text_backend_prefix(name):
                continue
            if "://" in name:
                logger.warning(f"Skipping URL-shaped model name during serialization: {name!r}")
                continue
            base_records[name] = record

        logger.debug(f"Serializing {len(base_records)} base text generation records")

        primary_csv_rows = {
            name: self._record_to_csv_row(name=name, record=record) for name, record in base_records.items()
        }

        existing_rows: list[dict[str, str]] = []
        if existing_csv_content is not None:
            existing_rows = self._read_existing_csv_content(existing_csv_content)
            logger.debug(f"Read {len(existing_rows)} existing CSV rows from supplied content")
        elif existing_csv_path is not None and existing_csv_path.exists():
            existing_rows = self._read_existing_csv(existing_csv_path)
            logger.debug(f"Read {len(existing_rows)} existing CSV rows from {existing_csv_path}")

        merged_rows = self._apply_changes(
            existing_rows=existing_rows,
            primary_csv_rows=primary_csv_rows,
            preserve_absent_existing_records=preserve_absent_existing_records,
        )

        csv_content = self._render_csv(merged_rows)
        db_dict = self._forward_convert(merged_rows)
        json_content = self._render_json(db_dict)

        logger.debug(
            f"Serialized {len(merged_rows)} CSV rows -> {len(db_dict)} db.json entries "
            f"(CSV: {len(csv_content)} bytes, JSON: {len(json_content)} bytes)"
        )

        return TextGenerationSyncArtifacts(
            csv_content=csv_content,
            json_content=json_content,
        )

    def _read_existing_csv(self, csv_path: Path) -> list[dict[str, str]]:
        """Parse an existing models.csv file preserving row order.

        Args:
            csv_path: Path to the CSV file.

        Returns:
            List of row dicts in file order.

        """
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            return self._read_existing_csv_stream(csvfile)

    def _read_existing_csv_content(self, csv_content: str) -> list[dict[str, str]]:
        """Parse existing CSV content preserving row order."""
        return self._read_existing_csv_stream(io.StringIO(csv_content))

    def _read_existing_csv_stream(self, stream: IO[str]) -> list[dict[str, str]]:
        """Parse an existing CSV stream preserving row order."""
        return [dict(row) for row in csv.DictReader(stream)]

    def _record_to_csv_row(
        self,
        *,
        name: str,
        record: LegacyRecordDict,
    ) -> dict[str, str]:
        """Reverse-convert a PRIMARY API record to a CSV row dict.

        Extracts only fields that belong in the CSV schema and converts
        types appropriately (int->str, list->comma-separated, dict->JSON).

        Auto-generated values (size tag, style tag, auto display_name) are
        stripped so the forward conversion can regenerate them identically
        to convert.py.

        Args:
            name: The base model name (dict key).
            record: The PRIMARY API record dict.

        Returns:
            Dict with string values keyed by CSV column names.

        """
        row: dict[str, str] = {"name": name}

        parameters = record.get("parameters")
        if parameters is not None:
            params_bn = float(parameters) / 1_000_000_000
            row["parameters_bn"] = _format_parameters_bn(params_bn)
        else:
            row["parameters_bn"] = ""

        model_name = TextModelWriteProcessor.extract_model_name(name)
        auto_display = TextModelWriteProcessor.generate_display_name(model_name)
        display_name = record.get("display_name", "")
        if display_name and display_name != auto_display:
            row["display_name"] = str(display_name)
        else:
            row["display_name"] = ""

        row["url"] = str(record.get("url", "") or "")
        row["baseline"] = str(record.get("baseline", "") or "")
        row["description"] = str(record.get("description", "") or "")

        style = record.get("style", "")

        # Detect default-only style: convert.py adds explicit styles to tags
        # before applying defaults.json, so a style present on the record but
        # absent from tags was only injected by the defaults system.
        record_tags: list[str] = record.get("tags", []) or []
        default_style = str(self._defaults.get("style", "") or "")
        if style and style == default_style and style not in record_tags:
            style = ""

        row["style"] = str(style) if style else ""

        tags = record.get("tags")
        if tags and isinstance(tags, list):
            row["tags"] = self._strip_auto_tags(
                tags=tags,
                style=str(style) if style else None,
                parameters=parameters,
            )
        else:
            row["tags"] = ""

        row["instruct_format"] = str(record.get("instruct_format", "") or "")

        settings = record.get("settings")
        if settings and isinstance(settings, dict):
            row["settings"] = json.dumps(settings)
        else:
            row["settings"] = ""

        if self._include_text_metadata_columns:
            for field_name in TEXT_METADATA_CSV_FIELDNAMES:
                field_value = record.get(field_name)
                row[field_name] = json.dumps(field_value) if isinstance(field_value, dict) and field_value else ""

        return row

    def _strip_auto_tags(
        self,
        *,
        tags: list[str],
        style: str | None,
        parameters: int | None,
    ) -> str:
        """Remove auto-generated tags (style + size bucket) and return CSV string.

        convert.py adds the style and a size tag (e.g., "3B") automatically.
        To avoid duplication on round-trip, strip them before writing CSV.

        Args:
            tags: The full tag list from the record.
            style: The style value (added as tag by convert.py).
            parameters: The parameter count (used to derive size tag).

        Returns:
            Comma-separated string of non-auto-generated tags.

        """
        auto_tags: set[str] = set()

        if style:
            auto_tags.add(style)

        if parameters is not None:
            params_bn = float(parameters) / 1_000_000_000
            size_tag = f"{round(params_bn, 0):.0f}B"
            auto_tags.add(size_tag)

        remaining = [tag for tag in tags if tag not in auto_tags]
        return ",".join(remaining)

    def _apply_changes(
        self,
        *,
        existing_rows: list[dict[str, str]],
        primary_csv_rows: dict[str, dict[str, str]],
        preserve_absent_existing_records: bool = True,
    ) -> list[dict[str, str]]:
        """Merge PRIMARY data into existing CSV rows, preserving order and CSV-only fields.

        Merge semantics:
        - Existing models present in PRIMARY: update field-by-field.
          PRIMARY non-empty values win; empty PRIMARY values fall back to existing CSV.
        - Existing models absent from PRIMARY: preserved (transition-period safety net).
        - New models in PRIMARY not in existing: appended at end.

        Args:
            existing_rows: Ordered list of CSV row dicts from the existing file.
            primary_csv_rows: PRIMARY records converted to CSV rows, keyed by name.
            preserve_absent_existing_records: Whether existing-only rows survive.

        Returns:
            Merged list of CSV row dicts.

        """
        remaining_primary = dict(primary_csv_rows)
        result: list[dict[str, str]] = []

        # Upstream currently contains a few duplicate CSV names. convert.py uses
        # last-write-wins semantics when generating db.json, so merge a PRIMARY
        # update into the last occurrence (the effective row) and leave earlier
        # duplicate rows untouched.
        last_existing_index = {
            row.get("name", ""): index for index, row in enumerate(existing_rows) if row.get("name", "")
        }

        for index, existing_row in enumerate(existing_rows):
            row_name = existing_row.get("name", "")
            if row_name in remaining_primary and last_existing_index.get(row_name) == index:
                primary_row = remaining_primary.pop(row_name)
                merged = self._merge_row_fields(existing_row, primary_row)
                result.append(merged)
            else:
                if row_name in primary_csv_rows or preserve_absent_existing_records:
                    # Keep earlier duplicates for a present PRIMARY model. For an
                    # absent model, retain the row only when the caller explicitly
                    # requests the compatibility safety net.
                    result.append(existing_row)

        for new_row in remaining_primary.values():
            result.append(new_row)

        return result

    def _merge_row_fields(
        self,
        existing_row: dict[str, str],
        primary_row: dict[str, str],
    ) -> dict[str, str]:
        """Merge a PRIMARY CSV row with an existing CSV row, field by field.

        PRIMARY non-empty values overwrite existing values. Empty/missing
        PRIMARY values fall back to the existing CSV value, preserving
        CSV-only fields like instruct_format.

        Args:
            existing_row: The row from the existing GitHub CSV.
            primary_row: The row derived from PRIMARY API data.

        Returns:
            Merged row dict.

        """
        merged = dict(existing_row)
        for key, value in primary_row.items():
            if value or key in self._primary_authoritative_fields:
                merged[key] = value
        return merged

    def _forward_convert(
        self,
        csv_rows: list[dict[str, str]],
    ) -> dict[str, LegacyRecordDict]:
        """Convert CSV rows to db.json dict, replicating convert.py exactly.

        This is a faithful reproduction of the upstream convert.py algorithm.
        Field ordering, default merging, empty-value stripping, and backend
        prefix generation all match convert.py's behavior.

        Args:
            csv_rows: List of CSV row dicts (string values).

        Returns:
            The complete db.json dict with all entries (base + backend prefixes).

        """
        defaults = dict(self._defaults)
        data: dict[str, LegacyRecordDict] = {}

        for csv_row in csv_rows:
            row: dict[str, Any] = dict(csv_row)

            name = row.pop("name")

            if "://" in name:
                logger.warning(f"Skipping URL-shaped model name in forward conversion: {name!r}")
                continue

            model_name = name.split("/")[1] if "/" in name else name

            params_str = row.pop("parameters_bn")
            params_f = float(params_str)
            row["parameters"] = int(params_f * 1_000_000_000)

            tags = set([t.strip() for t in row["tags"].split(",")] if row["tags"] else [])
            if style := row.get("style"):
                tags.add(style)
            tags.add(f"{round(params_f, 0):.0f}B")
            row["tags"] = sorted(tags)

            row["settings"] = json.loads(row["settings"]) if row["settings"] else {}
            for field_name in TEXT_METADATA_CSV_FIELDNAMES:
                if self._include_text_metadata_columns:
                    row[field_name] = json.loads(row[field_name]) if row.get(field_name) else None
                else:
                    # A previously written CSV may still carry these columns; drop them so
                    # the emitted db.json stays within the upstream schema.
                    row.pop(field_name, None)

            if not row.get("display_name"):
                row["display_name"] = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

            row = {k: v for k, v in row.items() if v}

            for key in get_model_name_variants(name):
                data[key] = {"name": key, "model_name": model_name, **defaults, **row}

        return data

    def _render_csv(self, rows: list[dict[str, str]]) -> str:
        """Render CSV rows to a string matching upstream models.csv format.

        Args:
            rows: List of CSV row dicts.

        Returns:
            CSV file content as a string.

        """
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=self._fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        # csv module always writes \r\n terminators; normalize to \n for git compatibility
        return output.getvalue().replace("\r\n", "\n")

    def _render_json(self, db_dict: dict[str, LegacyRecordDict]) -> str:
        """Render db.json dict to a string matching convert.py's output format.

        Uses 4-space indentation and a trailing newline, exactly as convert.py does.

        Args:
            db_dict: The complete db.json dict.

        Returns:
            JSON file content as a string.

        """
        return json.dumps(db_dict, indent=4) + "\n"


def _format_parameters_bn(value: float) -> str:
    """Format a parameters-in-billions value as a minimal string.

    Produces the simplest representation: no trailing zeros, no unnecessary
    decimal point. Matches how values appear in the upstream models.csv.

    Args:
        value: Parameter count in billions (e.g., 3.0, 0.56, 123.0).

    Returns:
        Minimal string representation (e.g., "3", "0.56", "123").

    Examples:
        >>> _format_parameters_bn(3.0)
        '3'
        >>> _format_parameters_bn(0.56)
        '0.56'
        >>> _format_parameters_bn(123.0)
        '123'

    """
    if value == int(value):
        return str(int(value))
    return str(value)
