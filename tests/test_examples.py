from pathlib import Path

import create_example_json
from create_example_json import (
    STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME,
    STABLE_DIFFUSION_SCHEMA_JSON_FILENAME,
)


def test_create_example_json():
    create_example_json.main()
    assert Path(STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME).exists()
    assert Path(STABLE_DIFFUSION_SCHEMA_JSON_FILENAME).exists()
