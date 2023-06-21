from pathlib import Path

import pytest
from loguru import logger

from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER_NAME


def pytest_collection_modifyitems(items):
    """Modifies test items in place to ensure test modules run in a given order."""
    MODULE_ORDER = ["tests.scripts", "tests.test_convert_legacy_database", "tests.test_consts"]
    # `test.scripts` must run first because it downloads the legacy database
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = items.copy()
    # Iteratively move tests of each module to the end of the test queue
    for module in MODULE_ORDER:
        sorted_items = [it for it in sorted_items if module_mapping[it] != module] + [
            it for it in sorted_items if module_mapping[it] == module
        ]
    items[:] = sorted_items


@pytest.fixture(scope="session")
def base_path_for_tests() -> Path:
    target_path = Path(__file__).parent.joinpath("test_data_results")
    target_path.mkdir(exist_ok=True)
    return target_path


@pytest.fixture(scope="session")
def legacy_folder_for_tests(base_path_for_tests: Path) -> Path:
    legacy_folder = base_path_for_tests.joinpath(LEGACY_REFERENCE_FOLDER_NAME)
    legacy_folder.mkdir(exist_ok=True)
    return legacy_folder


@pytest.fixture(scope="session", autouse=True)
def setup_logging(base_path_for_tests: Path):
    """Set up logging for tests."""
    logger.remove()
    logger.configure(
        handlers=[
            {
                "sink": base_path_for_tests.joinpath("test_log.txt"),
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            },
        ],
    )
