"""Tests for GitHub sync configuration."""

from __future__ import annotations

import pytest

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.sync import HordeGitHubSyncSettings


class TestHordeGitHubSyncSettings:
    """Test suite for HordeGitHubSyncSettings."""

    def test_default_settings(self) -> None:
        """Test default configuration values."""
        settings = HordeGitHubSyncSettings()

        assert settings.primary_api_url is None
        assert settings.primary_api_timeout == 30
        assert settings.github_token is None
        assert settings.categories_to_sync is None
        assert settings.pr_labels == ["automated", "sync", "ready-for-review"]
        assert settings.min_changes_threshold == 1
        assert settings.dry_run is False
        assert settings.verbose_logging is False

    def test_should_sync_category_no_whitelist(self) -> None:
        """Test category syncing when no whitelist is configured."""
        settings = HordeGitHubSyncSettings(categories_to_sync=None)

        assert settings.should_sync_category(MODEL_REFERENCE_CATEGORY.image_generation)
        assert settings.should_sync_category(MODEL_REFERENCE_CATEGORY.text_generation)
        assert settings.should_sync_category(MODEL_REFERENCE_CATEGORY.clip)

    def test_should_sync_category_with_whitelist(self) -> None:
        """Test category syncing with whitelist configured."""
        settings = HordeGitHubSyncSettings(categories_to_sync=["image_generation", "text_generation"])

        assert settings.should_sync_category(MODEL_REFERENCE_CATEGORY.image_generation)
        assert settings.should_sync_category(MODEL_REFERENCE_CATEGORY.text_generation)
        assert not settings.should_sync_category(MODEL_REFERENCE_CATEGORY.clip)

    def test_env_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables use correct prefix."""
        monkeypatch.setenv("HORDE_GITHUB_SYNC_PRIMARY_API_URL", "https://test.api.com")
        monkeypatch.setenv("HORDE_GITHUB_SYNC_DRY_RUN", "true")
        monkeypatch.setenv("HORDE_GITHUB_SYNC_MIN_CHANGES_THRESHOLD", "5")

        settings = HordeGitHubSyncSettings()

        assert settings.primary_api_url == "https://test.api.com"
        assert settings.dry_run is True
        assert settings.min_changes_threshold == 5
        # Note: github_branch is a computed property from horde_model_reference_settings,
        # not directly configurable via HordeGitHubSyncSettings env vars

    def test_min_changes_threshold_validation(self) -> None:
        """Test validation of min_changes_threshold."""
        settings = HordeGitHubSyncSettings(min_changes_threshold=0)

        assert settings.min_changes_threshold == 1

        settings = HordeGitHubSyncSettings(min_changes_threshold=-5)
        assert settings.min_changes_threshold == 1

        settings = HordeGitHubSyncSettings(min_changes_threshold=10)
        assert settings.min_changes_threshold == 10

    def test_pr_configuration(self) -> None:
        """Test PR-related configuration."""
        settings = HordeGitHubSyncSettings(
            pr_reviewers=["user1", "user2"],
            pr_labels=["sync", "automated", "urgent"],
            pr_auto_assign_team="my-org/my-team",
        )

        assert settings.pr_reviewers == ["user1", "user2"]
        assert settings.pr_labels == ["sync", "automated", "urgent"]
        assert settings.pr_auto_assign_team == "my-org/my-team"
