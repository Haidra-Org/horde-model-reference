from pathlib import Path

import pytest
from loguru import logger

from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER_NAME


@pytest.fixture(scope="session")
def base_path_for_tests() -> Path:
    target_path = Path(__file__).parent.joinpath("test_data_results/horde_model_reference")
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


def pytest_collection_modifyitems(items):  # type: ignore
    """Modifies test items to ensure test modules run in a given order."""
    MODULES_TO_RUN_FIRST = ["tests.test_scripts"]

    MODULES_TO_RUN_LAST = []  # FIXME make dynamic
    module_mapping = {item: item.module.__name__ for item in items}

    sorted_items = []

    for module in MODULES_TO_RUN_FIRST:
        sorted_items.extend([item for item in items if module_mapping[item] == module])

    sorted_items.extend(
        [item for item in items if module_mapping[item] not in MODULES_TO_RUN_FIRST + MODULES_TO_RUN_LAST],
    )

    for module in MODULES_TO_RUN_LAST:
        sorted_items.extend([item for item in items if module_mapping[item] == module])

    items[:] = sorted_items
