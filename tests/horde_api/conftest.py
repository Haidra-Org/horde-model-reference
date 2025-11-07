"""Pytest configuration for Horde API integration tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from horde_model_reference import ReplicateMode
from horde_model_reference.model_reference_manager import ModelReferenceManager


@pytest.fixture(scope="module", autouse=True)
def setup_model_reference_files(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Set up model reference files for audit tests using GitHub seeding.

    This fixture creates a temporary directory and enables GitHub seeding to
    automatically download model reference files. It overrides the manager
    dependency to use this test instance.

    Returns:
        Path to the base directory containing model reference files.
    """
    from horde_model_reference import horde_model_reference_settings

    # Create a temporary directory for model reference files
    # Use module scope so it persists across all tests in the module
    base_path = tmp_path_factory.mktemp("horde_model_reference_audit")

    # Reset the singleton so we can create a fresh instance
    previous_instance = ModelReferenceManager._instance
    ModelReferenceManager._instance = None

    # Store original setting
    original_seed_setting = horde_model_reference_settings.github_seed_enabled

    try:
        # Enable GitHub seeding for this test session
        # This will automatically download missing files from GitHub
        horde_model_reference_settings.github_seed_enabled = True

        # Create a manager using PRIMARY mode with GitHub seeding
        # This will automatically fetch missing categories from GitHub
        manager = ModelReferenceManager(
            backend=None,  # Let it auto-create with seeding enabled
            lazy_mode=False,
            base_path=base_path,
            replicate_mode=ReplicateMode.PRIMARY,
        )

        # Override the dependency to use our test manager
        from horde_model_reference.service.app import app
        from horde_model_reference.service.statistics.routers.audit import get_model_reference_manager

        app.dependency_overrides[get_model_reference_manager] = lambda: manager

        yield base_path

    finally:
        # Cleanup: remove the override
        from horde_model_reference.service.app import app
        from horde_model_reference.service.statistics.routers.audit import get_model_reference_manager

        app.dependency_overrides.pop(get_model_reference_manager, None)

        # Restore original settings
        horde_model_reference_settings.github_seed_enabled = original_seed_setting
        ModelReferenceManager._instance = previous_instance
