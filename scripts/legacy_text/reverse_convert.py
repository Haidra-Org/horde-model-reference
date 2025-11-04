"""Reverse conversion script: db.json (legacy GitHub format) → models.csv.

This script performs the inverse operation of convert.py, extracting unique base models
from the legacy JSON format (which includes backend-prefixed duplicates) and generating
a CSV file that can be used as the source of truth for model definitions.

The script:
1. Reads db.json from the same directory
2. Deduplicates backend-prefixed entries (aphrodite/, koboldcpp/)
3. Validates consistency between base and prefixed entries
4. Converts parameters from integers to billions (float)
5. Extracts and cleans tags (removes auto-generated style and size tags)
6. Serializes settings to compact JSON format
7. Writes models.csv with proper column ordering

Usage:
    python reverse_convert.py

The script uses strict validation and exits with code 1 on errors to ensure data integrity.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path to import horde_model_reference
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from horde_model_reference.meta_consts import has_legacy_text_backend_prefix

input_file = "db.json"
output_file = "models.csv"

# Load defaults and generation params for validation
with open("defaults.json") as f:
    defaults = json.load(f)
with open("generation_params.json") as f:
    params = json.load(f)


def validate_settings(settings: dict[str, Any], model_name: str) -> None:
    """Validate that settings keys are in generation_params.json.

    Args:
        settings: The settings dictionary to validate.
        model_name: The model name for error reporting.

    Raises:
        SystemExit: If validation fails.
    """
    if not isinstance(settings, dict):
        print(f"Error: {model_name}: settings must be a dictionary")
        sys.exit(1)

    invalid_keys = [k for k in settings if k not in params]
    if invalid_keys:
        print(f"Error: {model_name}: settings contains invalid keys: {invalid_keys}")
        print(f"Valid keys are: {list(params.keys())}")
        sys.exit(1)


def extract_base_model_name(name: str) -> str:
    """Extract the base model name without backend prefix.

    Args:
        name: The full model name (possibly with backend prefix).

    Returns:
        The base model name without prefix.
    """
    if has_legacy_text_backend_prefix(name):
        # Remove aphrodite/ or koboldcpp/ prefix
        if name.startswith("aphrodite/"):
            return name[len("aphrodite/") :]
        if name.startswith("koboldcpp/"):
            return name[len("koboldcpp/") :]
    return name


def compare_records(base_record: dict[str, Any], prefixed_record: dict[str, Any], field: str) -> bool:
    """Compare a field between base and prefixed records.

    Args:
        base_record: The base model record.
        prefixed_record: The backend-prefixed model record.
        field: The field name to compare.

    Returns:
        True if fields match or are both absent, False otherwise.
    """
    base_value = base_record.get(field)
    prefixed_value = prefixed_record.get(field)

    # Both absent is OK
    if base_value is None and prefixed_value is None:
        return True

    # One present, one absent is a mismatch
    if (base_value is None) != (prefixed_value is None):
        return False

    # Compare values - handle lists and dicts specially
    if isinstance(base_value, list) and isinstance(prefixed_value, list):
        return sorted(base_value) == sorted(prefixed_value)
    if isinstance(base_value, dict) and isinstance(prefixed_value, dict):
        return base_value == prefixed_value

    return base_value == prefixed_value


def validate_consistency(base_name: str, base_record: dict[str, Any], all_records: dict[str, dict[str, Any]]) -> None:
    """Validate that backend-prefixed entries match the base entry.

    Args:
        base_name: The base model name.
        base_record: The base model record.
        all_records: All model records from db.json.

    Raises:
        SystemExit: If inconsistencies are detected.
    """
    # Find all prefixed versions
    prefixed_names = [f"aphrodite/{base_name}", f"koboldcpp/{base_record.get('model_name', base_name)}"]

    # Fields that should be identical across all versions
    fields_to_compare = [
        "parameters",
        "baseline",
        "style",
        "nsfw",
        "description",
        "version",
        "url",
        "tags",
        "settings",
        "display_name",
    ]

    for prefixed_name in prefixed_names:
        if prefixed_name not in all_records:
            continue

        prefixed_record = all_records[prefixed_name]

        for field in fields_to_compare:
            if not compare_records(base_record, prefixed_record, field):
                print(f"Error: Inconsistency detected for {base_name}")
                print(f"  Field '{field}' differs between base and {prefixed_name}")
                print(f"  Base: {base_record.get(field)}")
                print(f"  Prefixed: {prefixed_record.get(field)}")
                sys.exit(1)


def auto_generate_display_name(model_name: str) -> str:
    """Generate display name using the same logic as convert.py.

    Args:
        model_name: The model name to generate display name from.

    Returns:
        The auto-generated display name.
    """
    return re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()


def extract_tags(record: dict[str, Any], style: str | None, params_bn: float) -> str:
    """Extract and clean tags, removing auto-generated ones.

    Args:
        record: The model record.
        style: The model style (will be removed from tags).
        params_bn: The parameter count in billions (size tag will be removed).

    Returns:
        Comma-separated tag string.
    """
    tags = record.get("tags", [])
    if not tags:
        return ""

    # Remove style tag if present
    tags = [t for t in tags if t != style]

    # Remove parameter size tag (e.g., "8B", "7B")
    size_tag = f"{round(params_bn, 0):.0f}B"
    tags = [t for t in tags if t != size_tag]

    # Return comma-separated string
    return ",".join(sorted(tags))


def serialize_settings(settings: dict[str, Any] | None) -> str:
    """Serialize settings dict to compact JSON format.

    Args:
        settings: The settings dictionary.

    Returns:
        JSON string with compact formatting, or empty string if no settings.
    """
    if not settings:
        return ""
    return json.dumps(settings, separators=(",", ":"))


def has_empty_config(record: dict[str, Any]) -> bool:
    """Check if a record has an empty config.

    Args:
        record: The model record.

    Returns:
        True if config is empty (no download entries).
    """
    config = record.get("config", {})
    if not config:
        return True

    download = config.get("download", [])
    return len(download) == 0


# Read db.json
try:
    with open(input_file) as f:
        data: dict[str, dict[str, Any]] = json.load(f)
except FileNotFoundError:
    print(f"Error: {input_file} not found")
    print("Please ensure db.json exists in the same directory as this script")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse {input_file}: {e}")
    sys.exit(1)

# Extract unique base models (deduplicate backend prefixes)
base_models: dict[str, dict[str, Any]] = {}
skipped_empty_config: list[str] = []

for model_name, record in data.items():
    # Skip backend-prefixed entries
    if has_legacy_text_backend_prefix(model_name):
        continue

    base_name = extract_base_model_name(model_name)

    # Check for empty config and skip with warning (config is optional for CSV purposes)
    # Only skip if explicitly empty, not if missing entirely
    if "config" in record and has_empty_config(record):
        skipped_empty_config.append(base_name)
        continue

    # Validate consistency with prefixed versions
    validate_consistency(base_name, record, data)

    base_models[base_name] = record

if skipped_empty_config:
    print(f"Warning: Skipped {len(skipped_empty_config)} models with empty configs:")
    for name in sorted(skipped_empty_config)[:10]:
        print(f"  - {name}")
    if len(skipped_empty_config) > 10:
        print(f"  ... and {len(skipped_empty_config) - 10} more")

# Convert to CSV format
csv_rows: list[dict[str, str]] = []

for base_name, record in base_models.items():
    # Extract and convert parameters
    parameters_int = record.get("parameters")
    if parameters_int is None:
        print(f"Error: {base_name} has no parameters count")
        sys.exit(1)

    try:
        params_bn = float(parameters_int) / 1_000_000_000
    except (ValueError, TypeError) as e:
        print(f"Error: Failed to convert parameters for {base_name}: {e}")
        sys.exit(1)

    # Get fields
    style = record.get("style")
    model_name = record.get("model_name", base_name)
    display_name = record.get("display_name")

    # Skip display_name if it matches auto-generated
    auto_display = auto_generate_display_name(model_name)
    if display_name == auto_display:
        display_name = ""

    # Extract and clean tags
    tags_str = extract_tags(record, style, params_bn)

    # Validate and serialize settings
    settings = record.get("settings")
    if settings:
        validate_settings(settings, base_name)
    settings_str = serialize_settings(settings)

    # Build CSV row with proper column ordering
    csv_row = {
        "name": base_name,
        "parameters_bn": f"{params_bn:.1f}",
        "description": record.get("description", ""),
        "version": record.get("version", ""),
        "style": style or "",
        "nsfw": str(record.get("nsfw", False)).lower(),
        "baseline": record.get("baseline", ""),
        "url": record.get("url", ""),
        "tags": tags_str,
        "settings": settings_str,
        "display_name": display_name or "",
    }

    csv_rows.append(csv_row)

# Write CSV file
TESTS_ONGOING = os.getenv("TESTS_ONGOING", False)
actual_output = output_file
output_file_existed_before = os.path.exists(output_file)

if TESTS_ONGOING and output_file_existed_before:
    # If tests are ongoing, write to a test file and compare
    actual_output = "models_test.csv"

# Define column order (must match convert.py expectations)
fieldnames = [
    "name",
    "parameters_bn",
    "description",
    "version",
    "style",
    "nsfw",
    "baseline",
    "url",
    "tags",
    "settings",
    "display_name",
]

with open(actual_output, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

if TESTS_ONGOING and output_file_existed_before:
    # Compare the test output with the existing file
    with open(output_file) as f:
        old_data = f.read()

    with open(actual_output) as f:
        new_data = f.read()

    if old_data != new_data:
        print(f"{output_file} and {actual_output} are different. Did you forget to run `reverse_convert.py`?")
        sys.exit(1)
    else:
        # Clean up test file if identical
        os.remove(actual_output)
else:
    print(f"Successfully converted {len(csv_rows)} models from {input_file} to {output_file}")
