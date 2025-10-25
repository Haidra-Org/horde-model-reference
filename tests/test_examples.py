from pathlib import Path

import scripts.create_example_json
from scripts.create_example_json import (
    STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME,
    STABLE_DIFFUSION_SCHEMA_JSON_FILENAME,
)


def test_create_example_json() -> None:
    """Test the creation of example JSON files."""
    scripts.create_example_json.create_example_json_schema()
    assert Path(STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME).exists()
    assert Path(STABLE_DIFFUSION_SCHEMA_JSON_FILENAME).exists()
