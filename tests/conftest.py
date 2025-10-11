import os
import sys
from collections.abc import Generator
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing any package modules
# This ensures settings singletons are initialized with test values
os.environ["TESTS_ONGOING"] = "1"
os.environ["AI_HORDE_TESTING"] = "True"

import pytest
from loguru import logger
from pytest import LogCaptureFixture

from horde_model_reference import ai_horde_ci_settings
from horde_model_reference.model_reference_manager import ModelReferenceManager

# Environment variable prefixes that should be cleared before tests
_HORDE_MODEL_REFERENCE_ENV_PREFIX = "HORDE_MODEL_REFERENCE_"

# Critical environment variables that must be cleared to avoid test interference
_CRITICAL_ENV_VARS_TO_CLEAR = [
    "HORDE_MODEL_REFERENCE_REPLICATE_MODE",
    "HORDE_MODEL_REFERENCE_REDIS_USE_REDIS",
    "HORDE_MODEL_REFERENCE_REDIS_URL",
    "HORDE_MODEL_REFERENCE_PRIMARY_API_URL",
    "HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS",
    "HORDE_MODEL_REFERENCE_MAKE_FOLDERS",
    "HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED",
]


def _clear_test_environment_variables() -> dict[str, str]:
    """Clear critical environment variables that could interfere with tests.

    Returns a dictionary of the cleared variables for potential restoration.
    """
    cleared_vars = {}
    for var_name in _CRITICAL_ENV_VARS_TO_CLEAR:
        if var_name in os.environ:
            cleared_vars[var_name] = os.environ[var_name]
            del os.environ[var_name]
    return cleared_vars


# Clear environment variables before any tests run
_CLEARED_ENV_VARS = _clear_test_environment_variables()


@pytest.fixture(scope="session")
def env_var_checks() -> None:
    """Check for required environment variables and validate test environment."""
    assert "TESTS_ONGOING" in os.environ, "Environment variable 'TESTS_ONGOING' not set."
    assert "AI_HORDE_TESTING" in os.environ, "Environment variable 'AI_HORDE_TESTING' not set."

    if not ai_horde_ci_settings.ai_horde_testing:
        pytest.fail(
            "AI_HORDE_TESTING must be set to True for tests to run. "
            "This ensures test-specific environment isolation logic is active."
        )

    # Verify critical environment variables were cleared
    remaining_critical_vars = [
        var for var in _CRITICAL_ENV_VARS_TO_CLEAR if var in os.environ
    ]
    if remaining_critical_vars:
        pytest.fail(
            f"Critical environment variables were not cleared before tests: {remaining_critical_vars}. "
            "These variables can interfere with test environment isolation. "
            "Please ensure conftest.py properly clears these variables."
        )


@pytest.fixture(scope="session")
def base_path_for_tests(env_var_checks: None) -> Path:
    """Return the base path for tests."""
    target_path = Path(__file__).parent.joinpath("test_data_results/horde_model_reference")
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path


@pytest.fixture(autouse=True)
def ensure_test_environment(env_var_checks: None) -> None:
    """Automatically ensure test environment is properly configured for every test.

    This fixture automatically runs for every test to ensure AI_HORDE_TESTING
    is set and critical environment variables are cleared.
    """
    pass


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
