import os
import sys
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

# CRITICAL: Set environment variables BEFORE importing any package modules
# This ensures settings singletons are initialized with test values
os.environ["TESTS_ONGOING"] = "1"
os.environ["AI_HORDE_TESTING"] = "True"
os.environ["HORDE_MODEL_REFERENCE_REPLICATE_MODE"] = "PRIMARY"
# Set to legacy so v1 CRUD routes are registered at import time
# v2 tests will override this via fixtures
os.environ["HORDE_MODEL_REFERENCE_CANONICAL_FORMAT"] = "legacy"

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from pytest import LogCaptureFixture

from horde_model_reference import PrefetchStrategy, ReplicateMode, ai_horde_ci_settings, horde_model_reference_settings
from horde_model_reference.backends.filesystem_backend import FileSystemBackend
from horde_model_reference.model_reference_manager import ModelReferenceManager
from horde_model_reference.service.app import app

# Environment variable prefixes that should be cleared before tests
_HORDE_MODEL_REFERENCE_ENV_PREFIX = "HORDE_MODEL_REFERENCE_"

# Critical environment variables that must be cleared to avoid test interference
_CRITICAL_ENV_VARS_TO_CLEAR = [
    "HORDE_MODEL_REFERENCE_REDIS_USE_REDIS",
    "HORDE_MODEL_REFERENCE_REDIS_URL",
    "HORDE_MODEL_REFERENCE_PRIMARY_API_URL",
    "HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS",
    "HORDE_MODEL_REFERENCE_MAKE_FOLDERS",
    "HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED",
    "HORDE_MODEL_REFERENCE_CANONICAL_FORMAT",
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
    remaining_critical_vars = [var for var in _CRITICAL_ENV_VARS_TO_CLEAR if var in os.environ]
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
        prefetch_strategy=PrefetchStrategy.LAZY,
    )


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    """Create a FastAPI test client for service tests."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def dependency_override() -> Iterator[Callable[[Callable[[], Any], Callable[[], Any]], None]]:
    """Register dependency overrides and ensure cleanup after tests."""
    overrides: list[Callable[[], Any]] = []

    def _register(dependency: Callable[[], Any], provider: Callable[[], Any]) -> None:
        app.dependency_overrides[dependency] = provider
        overrides.append(dependency)

    yield _register

    for dependency in reversed(overrides):
        app.dependency_overrides.pop(dependency, None)


@pytest.fixture
def primary_manager_override_factory(
    primary_base: Path,
    restore_manager_singleton: None,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
) -> Iterator[Callable[[Callable[[], ModelReferenceManager]], ModelReferenceManager]]:
    """Provide a factory to create PRIMARY managers and set dependency overrides."""

    def _create(dependency: Callable[[], ModelReferenceManager]) -> ModelReferenceManager:
        backend = FileSystemBackend(
            base_path=primary_base,
            cache_ttl_seconds=60,
            replicate_mode=ReplicateMode.PRIMARY,
        )
        manager = ModelReferenceManager(
            backend=backend,
            prefetch_strategy=PrefetchStrategy.LAZY,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        dependency_override(dependency, lambda: manager)
        return manager

    yield _create


@pytest.fixture
def legacy_canonical_mode(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Temporarily switch canonical_format to legacy for the duration of a test.

    This fixture uses monkeypatch to ensure proper cleanup and isolation.
    """
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "legacy")
    yield


@pytest.fixture
def v2_canonical_mode(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Temporarily switch canonical_format to v2 for the duration of a test.

    This fixture uses monkeypatch to ensure proper cleanup and isolation.
    """
    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "v2")
    yield


@pytest.fixture
def v1_canonical_manager(
    primary_base: Path,
    restore_manager_singleton: None,
    dependency_override: Callable[[Callable[[], Any], Callable[[], Any]], None],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[ModelReferenceManager]:
    """Create a PRIMARY mode manager with canonical_format='legacy' for v1 API tests.

    This fixture:
    1. Sets canonical_format to 'legacy' via monkeypatch
    2. Creates a fresh PRIMARY manager with isolated base_path
    3. Registers it as a dependency override for v1 API endpoints
    4. Cleans up automatically after the test
    """
    from horde_model_reference.service.v1.routers.shared import get_model_reference_manager

    monkeypatch.setattr(horde_model_reference_settings, "canonical_format", "legacy")

    backend = FileSystemBackend(
        base_path=primary_base,
        cache_ttl_seconds=60,
        replicate_mode=ReplicateMode.PRIMARY,
    )
    manager = ModelReferenceManager(
        backend=backend,
        prefetch_strategy=PrefetchStrategy.LAZY,
        replicate_mode=ReplicateMode.PRIMARY,
    )

    dependency_override(get_model_reference_manager, lambda: manager)
    yield manager


@pytest.fixture
def primary_base(tmp_path: Path) -> Path:
    """Return an isolated base directory for PRIMARY-mode file operations."""
    base = tmp_path / "primary"
    base.mkdir()
    return base


@pytest.fixture
def legacy_path(primary_base: Path) -> Path:
    """Return the legacy directory path for tests, creating it if needed."""
    legacy_dir = primary_base / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    return legacy_dir


@pytest.fixture
def minimal_legacy_stable_diffusion_data() -> dict[str, Any]:
    """Return minimal legacy stable diffusion test data."""
    return {
        "test_model_sd_1": {
            "name": "test_model_sd_1",
            "description": "Test stable diffusion model 1",
            "version": "1.0",
            "baseline": "stable_diffusion_1",
            "inpainting": False,
            "style": "generalist",
            "nsfw": False,
            "tags": ["tag1", "tag2", "tag3"],
            "config": {
                "files": [
                    {"path": "test_model_sd_1.ckpt", "sha256sum": "a" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_1.ckpt",
                        "file_url": "https://example.com/test_model_sd_1.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
        "test_model_sd_2": {
            "name": "test_model_sd_2",
            "description": "Test stable diffusion model 2",
            "version": "1.0",
            "baseline": "stable_diffusion_2_512",
            "inpainting": True,
            "style": "anime",
            "nsfw": False,
            "tags": ["tag4", "tag5", "tag6"],
            "config": {
                "files": [
                    {"path": "test_model_sd_2.ckpt", "sha256sum": "b" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_2.ckpt",
                        "file_url": "https://example.com/test_model_sd_2.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
        "test_model_sd_3": {
            "name": "test_model_sd_3",
            "description": "Test stable diffusion model 3",
            "version": "1.0",
            "baseline": "stable_diffusion_xl",
            "inpainting": False,
            "style": "realistic",
            "nsfw": True,
            "tags": ["tag7", "tag8", "tag9"],
            "config": {
                "files": [
                    {"path": "test_model_sd_3.ckpt", "sha256sum": "c" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_3.ckpt",
                        "file_url": "https://example.com/test_model_sd_3.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
        "test_model_sd_4": {
            "name": "test_model_sd_4",
            "description": "Test stable diffusion model 4",
            "version": "1.0",
            "baseline": "stable_cascade",
            "inpainting": False,
            "style": "artistic",
            "nsfw": False,
            "tags": ["tag10", "tag11", "tag12"],
            "config": {
                "files": [
                    {"path": "test_model_sd_4.ckpt", "sha256sum": "d" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_4.ckpt",
                        "file_url": "https://example.com/test_model_sd_4.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
        "test_model_sd_5": {
            "name": "test_model_sd_5",
            "description": "Test stable diffusion model 5",
            "version": "1.0",
            "baseline": "flux_1",
            "inpainting": False,
            "style": "furry",
            "nsfw": False,
            "tags": ["tag13"],
            "config": {
                "files": [
                    {"path": "test_model_sd_5.ckpt", "sha256sum": "e" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_5.ckpt",
                        "file_url": "https://example.com/test_model_sd_5.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
        "test_model_sd_6": {
            "name": "test_model_sd_6",
            "description": "Test stable diffusion model 6",
            "version": "1.0",
            "baseline": "flux_schnell",
            "inpainting": False,
            "style": "other",
            "nsfw": False,
            "tags": ["tag14"],
            "config": {
                "files": [
                    {"path": "test_model_sd_6.ckpt", "sha256sum": "f" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_model_sd_6.ckpt",
                        "file_url": "https://example.com/test_model_sd_6.ckpt",
                        "file_path": "",
                    },
                ],
            },
        },
    }


@pytest.fixture
def minimal_legacy_clip_data() -> dict[str, Any]:
    """Return minimal legacy CLIP test data."""
    return {
        "test_clip_1": {
            "name": "test_clip_1",
            "description": "Test CLIP model 1",
            "version": "1.0",
            "pretrained_name": "openai/clip-vit-base-patch32",
            "config": {
                "files": [
                    {"path": "test_clip_1.safetensors", "sha256sum": "b" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_clip_1.safetensors",
                        "file_url": "https://example.com/test_clip_1.safetensors",
                        "file_path": "",
                    },
                ],
            },
        },
    }


@pytest.fixture
def minimal_legacy_controlnet_data() -> dict[str, Any]:
    """Return minimal legacy ControlNet test data."""
    return {
        "test_controlnet_1": {
            "name": "test_controlnet_1",
            "description": "Test ControlNet model 1",
            "version": "1.0",
            "style": "control_canny",
            "config": {
                "files": [
                    {"path": "test_controlnet_1.safetensors", "sha256sum": "c" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_controlnet_1.safetensors",
                        "file_url": "https://example.com/test_controlnet_1.safetensors",
                        "file_path": "",
                    },
                ],
            },
        },
    }


@pytest.fixture
def minimal_legacy_text_generation_data() -> dict[str, Any]:
    """Return minimal legacy text generation test data."""
    return {
        "test_text_gen_1": {
            "name": "test_text_gen_1",
            "description": "Test text generation model 1",
            "version": "1.0",
            "baseline": "gpt2",
            "parameters": 124000000,
            "style": "generalist",
            "nsfw": False,
            "config": {
                "files": [
                    {"path": "test_text_gen_1.safetensors", "sha256sum": "d" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_text_gen_1.safetensors",
                        "file_url": "https://example.com/test_text_gen_1.safetensors",
                        "file_path": "",
                    },
                ],
            },
        },
    }


@pytest.fixture
def minimal_legacy_generic_data() -> dict[str, Any]:
    """Return minimal legacy generic model test data."""
    return {
        "test_generic_1": {
            "name": "test_generic_1",
            "description": "Test generic model 1",
            "version": "1.0",
            "config": {
                "files": [
                    {"path": "test_generic_1.pth", "sha256sum": "e" * 64},
                ],
                "download": [
                    {
                        "file_name": "test_generic_1.pth",
                        "file_url": "https://example.com/test_generic_1.pth",
                        "file_path": "",
                    },
                ],
            },
        },
    }


@pytest.fixture
def populated_legacy_path(
    legacy_path: Path,
    minimal_legacy_stable_diffusion_data: dict[str, Any],
    minimal_legacy_clip_data: dict[str, Any],
    minimal_legacy_controlnet_data: dict[str, Any],
    minimal_legacy_text_generation_data: dict[str, Any],
) -> Path:
    """Return a legacy path populated with minimal test files.

    IMPORTANT: This fixture creates files in the legacy/ subdirectory but returns
    the legacy_path itself (e.g., 'primary/legacy/').

    When using converters, pass primary_base (NOT this populated_legacy_path) to the
    legacy_folder_path parameter, because converters automatically append '/legacy/'.

    Example:
        # CORRECT:
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=primary_base,  # e.g., 'primary/'
            target_file_folder=primary_base,
        )

        # WRONG - results in 'primary/legacy/legacy/' double nesting:
        converter = LegacyTextGenerationConverter(
            legacy_folder_path=populated_legacy_path,  # e.g., 'primary/legacy/'
            target_file_folder=primary_base,
        )
    """
    import csv
    import json

    # Create stable_diffusion.json
    (legacy_path / "stable_diffusion.json").write_text(
        json.dumps(minimal_legacy_stable_diffusion_data, indent=2),
    )

    # Create clip.json
    (legacy_path / "clip.json").write_text(
        json.dumps(minimal_legacy_clip_data, indent=2),
    )

    # Create controlnet.json
    (legacy_path / "controlnet.json").write_text(
        json.dumps(minimal_legacy_controlnet_data, indent=2),
    )

    # Create models.csv for text_generation (CSV format, not JSON like other categories)
    # IMPORTANT: text_generation is the only category using CSV for legacy format.
    # The CSV structure matches the format downloaded from GitHub.
    csv_path = legacy_path / "models.csv"
    fieldnames = [
        "name",
        "parameters_bn",  # Parameters in billions (converted to int in processing)
        "description",
        "version",
        "style",
        "nsfw",  # String "true"/"false", converted to boolean
        "baseline",
        "url",
        "tags",  # Comma-separated string, converted to list
        "settings",  # JSON string, converted to dict (flat only, no nested dicts)
        "display_name",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Convert each model from JSON to CSV row
        for model_name, model_data in minimal_legacy_text_generation_data.items():
            # Extract parameters_bn from parameters (convert back from integer)
            params = model_data.get("parameters", 0)
            params_bn = params / 1_000_000_000 if params else 0

            # Convert tags list to comma-separated string
            tags = model_data.get("tags", [])
            tags_str = ",".join(tags) if tags else ""

            # Convert settings dict to JSON string
            settings = model_data.get("settings")
            settings_str = json.dumps(settings) if settings else ""

            writer.writerow(
                {
                    "name": model_data.get("name", model_name),
                    "parameters_bn": str(params_bn),
                    "description": model_data.get("description", ""),
                    "version": model_data.get("version", ""),
                    "style": model_data.get("style", ""),
                    "nsfw": str(model_data.get("nsfw", False)).lower(),
                    "baseline": model_data.get("baseline", ""),
                    "url": model_data.get("url", ""),
                    "tags": tags_str,
                    "settings": settings_str,
                    "display_name": model_data.get("display_name", ""),
                }
            )

    return legacy_path


@pytest.fixture
def mock_auth_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock auth_against_horde to always return True for testing write operations."""

    async def _mock_auth(apikey: str, client: Any) -> bool:  # noqa: ANN401
        return True

    monkeypatch.setattr("horde_model_reference.service.v1.routers.create_update.auth_against_horde", _mock_auth)
    monkeypatch.setattr("horde_model_reference.service.v1.routers.shared.auth_against_horde", _mock_auth)


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
