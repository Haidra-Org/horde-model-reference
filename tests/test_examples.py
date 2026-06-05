from pathlib import Path

from .create_example_json import (
    STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME,
    STABLE_DIFFUSION_SCHEMA_JSON_FILENAME,
    create_example_json_schema,
)


def test_create_example_json() -> None:
    """Test the creation of example JSON files."""
    create_example_json_schema()
    assert Path(STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME).exists()
    assert Path(STABLE_DIFFUSION_SCHEMA_JSON_FILENAME).exists()
