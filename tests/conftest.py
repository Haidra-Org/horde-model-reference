import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger
from pytest import LogCaptureFixture

from horde_model_reference.model_reference_manager import ModelReferenceManager

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
        lazy_mode=True,
    )


@pytest.fixture
def primary_base(tmp_path: Path) -> Path:
    """Return an isolated base directory for PRIMARY-mode file operations."""
    base = tmp_path / "primary"
    base.mkdir()
    return base


@pytest.fixture
def restore_manager_singleton() -> Generator[None, None, None]:
    """Reset the ModelReferenceManager singleton around a test."""
    previous = ModelReferenceManager._instance
    ModelReferenceManager._instance = None
    try:
        yield
    finally:
        ModelReferenceManager._instance = previous


def pytest_collection_modifyitems(items) -> None:  # type: ignore #  # noqa: ANN001
    """Modify test items to ensure test modules run in a given order."""
    MODULES_TO_RUN_FIRST: list[str] = []
    MODULES_TO_RUN_LAST: list[str] = []

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
