# Legacy CSV Conversion for Text Generation Models

## Overview

The `text_generation` category is unique in the model reference system as it's the **only category** that uses CSV format for legacy files instead of JSON. This document explains the conversion process, common pitfalls, and implementation details.

## File Format Summary

| Category | Legacy Format | V2 Format | Legacy Path | V2 Path |
|----------|---------------|-----------|-------------|---------|
| `text_generation` | **CSV** | JSON | `{base}/legacy/models.csv` | `{base}/text_generation.json` |
| All others | JSON | JSON | `{base}/legacy/{category}.json` | `{base}/{category}.json` |

## CSV Structure

The legacy CSV file (`models.csv`) has the following columns:

```csv
name,parameters_bn,description,version,style,nsfw,baseline,url,tags,settings,display_name
```

### Column Details

- **name**: Model identifier (string)
- **parameters_bn**: Parameters in billions (float, e.g., "7.0" for 7B parameters)
- **description**: Model description (string)
- **version**: Model version (string)
- **style**: Model style/category (string)
- **nsfw**: NSFW flag (string: "true" or "false")
- **baseline**: Base model/architecture (string)
- **url**: Model URL (string)
- **tags**: Comma-separated tags (string, e.g., "tag1,tag2,tag3")
- **settings**: JSON object as string (string, e.g., '{"temperature": 0.7}')
- **display_name**: Display name (string)

## Conversion Process

### 1. CSV → Internal Dictionary

The `LegacyTextGenerationConverter._load_and_validate_legacy_records()` method reads the CSV and converts it to an internal dictionary format:

```python
# Parameters: billions → integer
params_bn = float(row.get("parameters_bn", 0))
parameters = int(params_bn * 1_000_000_000)  # 7.0 → 7,000,000,000

# Tags: comma-separated string → list
tags_str = row.get("tags", "")
tags = [t.strip() for t in tags_str.split(",") if t.strip()]

# Settings: JSON string → dict
settings_str = row.get("settings", "")
settings = json.loads(settings_str) if settings_str else None

# NSFW: string → boolean
nsfw = row.get("nsfw", "").lower() == "true"
```

### 2. Dictionary → Pydantic Validation

The internal dictionary is validated using `LegacyTextGenerationRecord` Pydantic model.

### 3. Pydantic → V2 JSON Output

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
    "nested": {"key": "value"},  // ❌ Nested dicts not allowed
    "complex": {"another": {"level": "here"}}  // ❌ Nested dicts not allowed
}
```

### Settings JSON Validity

The `settings` column must contain valid JSON. If a row includes malformed JSON, `json.loads()` raises `json.JSONDecodeError` and the converter stops rather than silently skipping that entry.

### Numeric Parameters Required

`parameters_bn` must parse as a floating-point number (e.g., `"7.0"`). Non-numeric strings—including blank cells—raise a `ValueError` during conversion; there is no automatic fallback beyond the explicit `0` default used when the column is truly missing.

## Common Pitfalls

### 1. Double Legacy Folder Bug

**Problem:** Passing the wrong path to converters results in `{base}/legacy/legacy/models.csv`.

**Root Cause:** The converter's `legacy_folder_path` parameter expects a BASE path (e.g., `data/`), and it automatically appends `/legacy/` internally via `get_legacy_model_reference_file_path()`.

**Wrong Usage:**

```python
# ❌ WRONG - results in data/legacy/legacy/models.csv
converter = LegacyTextGenerationConverter(
    legacy_folder_path=Path("data/legacy"),  # Already has /legacy/
    target_file_folder=Path("data"),
)
```

**Correct Usage:**

```python
# ✅ CORRECT - results in data/legacy/models.csv
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

`parameters_bn` is the exception—leave it blank or non-numeric and the conversion fails. Ensure every row contains a numeric value (use `0` when no parameter estimate is available).

## Testing Considerations

### Test Fixture Path Handling

In tests, the `populated_legacy_path` fixture creates files in `primary/legacy/` but when calling converters, always pass `primary_base` (not `populated_legacy_path`):

```python
# ✅ CORRECT
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
# 0.5 → 500,000,000 (500M)
# 7.0 → 7,000,000,000 (7B)
# 13.0 → 13,000,000,000 (13B)
# 70.0 → 70,000,000,000 (70B)
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
