# Legacy CSV Conversion for Text Generation Models

## Overview

The `text_generation` category is unique in the model reference system as it's the **only category** that uses CSV format for legacy files instead of JSON. This document explains the conversion process, common pitfalls, and implementation details.

## File Format Summary

| Category          | Legacy Format | V2 Format | Legacy Path                     | V2 Path                       |
| ----------------- | ------------- | --------- | ------------------------------- | ----------------------------- |
| `text_generation` | **CSV**       | JSON      | `{base}/legacy/models.csv`      | `{base}/text_generation.json` |
| All others        | JSON          | JSON      | `{base}/legacy/{category}.json` | `{base}/{category}.json`      |

## CSV Structure

The legacy CSV file (`models.csv`) has the following columns:

```csv
name,parameters_bn,display_name,url,baseline,description,style,tags,instruct_format,settings,context_window,interaction_modes,capabilities
```

The written column order is `TEXT_CSV_FIELDNAMES` in `legacy/text_csv_utils.py`. The sync serializer in
`sync/text_generation_serializer.py` writes only the first ten columns by default (see
[Durable Metadata Columns](#durable-metadata-columns)).

### Column Details

- **name**: Model identifier (string)
- **parameters_bn**: Parameters in billions (float, e.g., "7.0" for 7B parameters)
- **display_name**: Display name (string)
- **url**: Model URL (string)
- **baseline**: Base model/architecture (string)
- **description**: Model description (string)
- **style**: Model style/category (string)
- **tags**: Comma-separated tags (string, e.g., "tag1,tag2,tag3")
- **instruct_format**: Legacy prompt-style label (string)
- **settings**: JSON object as string (string, e.g., '{"temperature": 0.7}')
- **context_window**: JSON object as string, or empty
- **interaction_modes**: JSON object as string, or empty
- **capabilities**: JSON object as string, or empty

`TextCSVRow` also carries `version` and `nsfw`, which the parser reads from those columns when a file provides
them (`nsfw` is true only for the literal string `true`, case-insensitive). Neither is in the written column
list, so both are dropped on write-back.

### Durable Metadata Columns

`context_window`, `interaction_modes`, and `capabilities` carry the reviewed durable claims described in
[Model Reference Records](model_reference_records.md#text-generation-durable-metadata). Each holds a serialized
JSON object; an empty cell parses as `None` and is dropped from the legacy dictionary rather than stored as an
empty object.

```csv
context_window,interaction_modes,capabilities
"{""maximum_tokens"": 8192}","{""chat"": {""status"": ""supported""}}","{""tool_calling"": {""status"": ""unknown""}}"
```

These columns are strict. A cell that is neither empty, valid JSON, nor a JSON **object** produces a
`TextCSVIssue` and the whole row is skipped, so a malformed value never yields a half-populated record. This is
narrower than `settings`, where a parse failure stops conversion outright.

These columns are PRIMARY-local. The GitHub sync serializer withholds them by default because the upstream
`models.csv` schema and its `convert.py` do not define them; set `HORDE_GITHUB_SYNC_EXPORT_TEXT_METADATA_COLUMNS`
to include them once upstream accepts the wider schema. With that flag on they join
`_PRIMARY_AUTHORITATIVE_FIELDS` in the sync serializer, so a PRIMARY value overwrites whatever the GitHub CSV
holds.

Guidance profile assignments are not CSV columns. `instruct_format` remains the only prompting hint in the
legacy round trip; authored guidance lives in its own catalog
(see [Text Model Usage Guidance](../concepts/text_guidance.md)).

## Conversion Process

### 1. CSV -> Internal Dictionary

The `LegacyTextGenerationConverter._load_and_validate_legacy_records()` method reads the CSV and converts it to an internal dictionary format:

```python
# Parameters: billions -> integer
params_bn = float(row.get("parameters_bn", 0))
parameters = int(params_bn * 1_000_000_000)  # 7.0 -> 7,000,000,000

# Tags: comma-separated string -> list
tags_str = row.get("tags", "")
tags = [t.strip() for t in tags_str.split(",") if t.strip()]

# Settings: JSON string -> dict
settings_str = row.get("settings", "")
settings = json.loads(settings_str) if settings_str else None

# NSFW: string -> boolean
nsfw = row.get("nsfw", "").lower() == "true"
```

### 2. Dictionary -> Pydantic Validation

The internal dictionary is validated using `LegacyTextGenerationRecord` Pydantic model.

### 3. Pydantic -> V2 JSON Output

The base class `write_out_records()` method writes the converted records to `text_generation.json` (always JSON format).

### Backend Prefix Filtering

`LegacyTextGenerationConverter._convert_single_record()` calls `has_legacy_text_backend_prefix()` and drops any rows whose `name` uses backend-generated prefixes such as `aphrodite/` or `koboldcpp/`. These prefixed entries are duplicates that only exist for backwards compatibility and are intentionally excluded from the v2 dataset.

## Critical Constraints

### Settings Field Type Limitation

The `settings` field has a strict type constraint that **does NOT support nested dictionaries**:

```python
settings: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None
```

**Valid Settings:**

```json
{
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "stop_sequences": ["</s>", "[DONE]"],
    "enabled": true
}
```

**Invalid Settings (will fail validation):**

```json
{
    "nested": { "key": "value" },
    "complex": { "another": { "level": "here" } }
}
```

### Settings JSON Validity

The `settings` column must contain valid JSON. If a row includes malformed JSON, `json.loads()` raises `json.JSONDecodeError` and the converter stops rather than silently skipping that entry.

### Numeric Parameters Required

`parameters_bn` must parse as a floating-point number (e.g., `"7.0"`). Non-numeric strings-including blank cells-raise a `ValueError` during conversion; there is no automatic fallback beyond the explicit `0` default used when the column is truly missing.

## Common Pitfalls

### 1. Double Legacy Folder Bug

**Problem:** Passing the wrong path to converters results in `{base}/legacy/legacy/models.csv`.

**Root Cause:** The converter's `legacy_folder_path` parameter expects a BASE path (e.g., `data/`), and it automatically appends `/legacy/` internally via `get_legacy_model_reference_file_path()`.

**Wrong Usage:**

```python
# WRONG: WRONG - results in data/legacy/legacy/models.csv
converter = LegacyTextGenerationConverter(
    legacy_folder_path=Path("data/legacy"),  # Already has /legacy/
    target_file_folder=Path("data"),
)
```

**Correct Usage:**

```python
# CORRECT: CORRECT - results in data/legacy/models.csv
converter = LegacyTextGenerationConverter(
    legacy_folder_path=Path("data"),  # Base path only
    target_file_folder=Path("data"),
)
```

### 2. Empty File Handling

Empty CSV files (0 bytes or only headers) are handled gracefully and return an empty dictionary without errors.

### 3. Missing Optional Fields

CSV rows with missing optional fields are handled by using empty strings or default values:

```python
# Missing fields default to:
description=""
version=""
style=""
baseline=""
url=""
tags=[]
settings=None
display_name=""
```

`parameters_bn` is the exception-leave it blank or non-numeric and the conversion fails. Ensure every row contains a numeric value (use `0` when no parameter estimate is available).

## Testing Considerations

### Test Fixture Path Handling

In tests, the `populated_legacy_path` fixture creates files in `primary/legacy/` but when calling converters, always pass `primary_base` (not `populated_legacy_path`):

```python
# CORRECT: CORRECT
def test_converter(primary_base: Path, populated_legacy_path: Path):
    converter = LegacyTextGenerationConverter(
        legacy_folder_path=primary_base,  # Pass base, not populated_legacy_path
        target_file_folder=primary_base,
    )
```

### Comprehensive Test Coverage

The test suite (`test_text_generation_csv_conversion.py`) covers:

1. CSV reading and parsing
2. JSON output format verification (regression test)
3. Data integrity (all fields preserved)
4. Empty file handling
5. Missing optional fields
6. Complex settings (within type constraints)
7. Non-existent file handling
8. Large parameter value conversion
9. Output format verification (JSON not CSV)

## Implementation Details

### Key Files

- **Converter**: `src/horde_model_reference/legacy/classes/legacy_converters.py`
    - `LegacyTextGenerationConverter` class
    - `_load_and_validate_legacy_records()` override for CSV reading
    - `_convert_single_record()` skips backend-prefixed duplicates via `has_legacy_text_backend_prefix()`

- **Backend - GitHub**: `src/horde_model_reference/backends/github_backend.py`
    - `_read_legacy_csv_to_dict()` method for CSV parsing
    - Empty file handling

- **Backend - FileSystem**: `src/horde_model_reference/backends/filesystem_backend.py`
    - `_read_legacy_csv_to_dict()` method for CSV parsing

- **Tests**:
    - `tests/test_text_generation_csv_conversion.py` - Comprehensive CSV conversion tests
    - `tests/test_text_generation_file_paths.py` - File path and format verification
    - `tests/conftest.py` - Test fixtures with CSV generation

### Parameter Conversion Formula

```python
# CSV stores parameters in billions (float)
parameters_bn = 7.0  # From CSV

# Convert to integer parameters
parameters = int(parameters_bn * 1_000_000_000)
# Result: 7,000,000,000

# Examples:
# 0.5 -> 500,000,000 (500M)
# 7.0 -> 7,000,000,000 (7B)
# 13.0 -> 13,000,000,000 (13B)
# 70.0 -> 70,000,000,000 (70B)
```

## Best Practices

1. **Always pass base paths** to converters, never paths with `/legacy/` already included
2. **Test with empty files** to ensure graceful handling
3. **Validate settings constraints** - only flat dicts allowed
4. **Pre-validate numeric and JSON fields** - ensure `parameters_bn` values are numeric strings and `settings` cells contain valid JSON before running the converter
5. **Use CSV.DictWriter** for creating test CSV files to ensure proper formatting
6. **Verify JSON output** - output should always be JSON, never CSV
7. **Handle missing fields** with appropriate defaults

## GitHub Sync Behavior

When syncing from GitHub:

1. GitHub backend downloads `legacy/models.csv` (CSV format)
2. If file is empty (0 bytes), skip conversion
3. Parse CSV using `_read_legacy_csv_to_dict()`
4. Convert to TextGenerationModelRecord objects
5. Write to `text_generation.json` (JSON format)
6. Serve both legacy CSV and v2 JSON endpoints

## Migration Notes

If you need to add a new category with CSV format (not recommended unless necessary):

1. Override `_load_and_validate_legacy_records()` in your converter
2. Implement CSV reading logic similar to `LegacyTextGenerationConverter`
3. Add backend CSV reading support in GitHub and FileSystem backends
4. Create comprehensive tests covering all edge cases
5. Update path constants if using different filenames
6. Document the CSV structure and conversion process

**Note:** It's strongly recommended to use JSON for new categories to maintain consistency with the rest of the system.
