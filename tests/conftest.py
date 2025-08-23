import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger
from pytest import LogCaptureFixture

from horde_model_reference.legacy.legacy_download_manager import LegacyReferenceDownloadManager
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.path_consts import LEGACY_REFERENCE_FOLDER_NAME

os.environ["TESTS_ONGOING"] = "1"


@pytest.fixture(scope="session")
def env_var_checks() -> None:
    """Check for required environment variables."""
    assert "TESTS_ONGOING" in os.environ, "Environment variable 'TESTS_ONGOING' not set."


@pytest.fixture(scope="session")
def base_path_for_tests() -> Path:
    """Return the base path for tests."""
    target_path = Path(__file__).parent.joinpath("test_data_results/horde_model_reference")
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


@pytest.fixture(scope="session")
def legacy_folder_for_tests(base_path_for_tests: Path) -> Path:
    """Return the legacy folder for tests."""
    legacy_folder = base_path_for_tests.joinpath(LEGACY_REFERENCE_FOLDER_NAME)
    legacy_folder.mkdir(parents=True, exist_ok=True)
    return legacy_folder


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Generator[LogCaptureFixture, None, None]:
    """Fixture to capture log messages during tests.

    See https://loguru.readthedocs.io/en/stable/resources/migration.html#migration-caplog for more information.
    """
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(scope="session", autouse=True)
def setup_logging(base_path_for_tests: Path) -> None:
    """Set up logging for tests."""
    logger.remove()
    logger.configure(
        handlers=[
            {
                "sink": base_path_for_tests.joinpath("test_log.txt"),
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            },
            {
                "sink": sys.stderr,
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {name}:{function}:{line} | {level} | {message}",
            },
        ],
    )


@pytest.fixture(scope="session")
def model_reference_manager() -> ModelReferenceManager:
    """Return a ModelReferenceManager instance for tests."""
    return ModelReferenceManager(
        override_existing=False,
        lazy_mode=True,
    )


@pytest.fixture(scope="session")
def legacy_reference_download_manager(base_path_for_tests: Path) -> LegacyReferenceDownloadManager:
    """Return a LegacyReferenceDownloadManager instance for tests."""
    return LegacyReferenceDownloadManager(
        base_path=base_path_for_tests,
    )


def pytest_collection_modifyitems(items) -> None:  # type: ignore #  # noqa: ANN001
    """Modify test items to ensure test modules run in a given order."""
    MODULES_TO_RUN_FIRST = ["tests.test_scripts"]

    MODULES_TO_RUN_LAST = []  # type: ignore  # FIXME make dynamic
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
