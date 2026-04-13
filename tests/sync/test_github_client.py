"""Tests for GitHubSyncClient commit guard logic.

Validates that sync operations handle the case where the comparator detects
changes but the actual file writes produce no git diff (false positives).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.sync import GitHubSyncClient, ModelReferenceDiff
from horde_model_reference.sync.text_generation_serializer import TextGenerationSyncArtifacts

_CategoriesData = dict[
    MODEL_REFERENCE_CATEGORY,
    tuple[ModelReferenceDiff, dict[str, dict[str, Any]], TextGenerationSyncArtifacts | None],
]


@pytest.fixture
def sample_diff() -> ModelReferenceDiff:
    """Create a diff that reports changes."""
    diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
    diff.modified_models = {
        "model1": {"name": "model1", "description": "modified"},
    }
    return diff


@pytest.fixture
def sample_primary_data() -> dict[str, dict[str, Any]]:
    """Return a simple primary data for testing."""
    return {
        "model1": {"name": "model1", "description": "modified"},
    }


class TestCommitChangesReturnValue:
    """Test _commit_changes returns a boolean indicating if changes were committed."""

    def test_commit_changes_returns_true_when_dirty(self) -> None:
        """When there are actual file changes, _commit_changes returns True."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        client._current_repo = mock_repo

        diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
        diff.modified_models = {"m1": {"name": "m1"}}

        result = client._commit_changes(MODEL_REFERENCE_CATEGORY.image_generation, diff)

        assert result is True
        mock_repo.git.add.assert_called_once_with(".")
        mock_repo.git.commit.assert_called_once()

    def test_commit_changes_returns_false_when_clean(self) -> None:
        """When there are no file changes, _commit_changes returns False."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = False
        client._current_repo = mock_repo

        diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
        diff.modified_models = {"m1": {"name": "m1"}}

        result = client._commit_changes(MODEL_REFERENCE_CATEGORY.image_generation, diff)

        assert result is False
        mock_repo.git.add.assert_called_once_with(".")
        mock_repo.git.commit.assert_not_called()


class TestCommitMultiCategoryReturnValue:
    """Test _commit_multi_category_changes returns a boolean."""

    def test_returns_true_when_dirty(self) -> None:
        """Returns True when there are actual file changes."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = True
        client._current_repo = mock_repo

        diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
        diff.modified_models = {"m1": {"name": "m1"}}

        primary_data: dict[str, dict[str, Any]] = {"m1": {"name": "m1"}}
        categories_data: _CategoriesData = {
            MODEL_REFERENCE_CATEGORY.image_generation: (diff, primary_data, None),
        }

        result = client._commit_multi_category_changes(categories_data)

        assert result is True
        mock_repo.git.commit.assert_called_once()

    def test_returns_false_when_clean(self) -> None:
        """Returns False when there are no file changes."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        mock_repo = MagicMock()
        mock_repo.is_dirty.return_value = False
        client._current_repo = mock_repo

        diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
        diff.modified_models = {"m1": {"name": "m1"}}

        primary_data: dict[str, dict[str, Any]] = {"m1": {"name": "m1"}}
        categories_data: _CategoriesData = {
            MODEL_REFERENCE_CATEGORY.image_generation: (diff, primary_data, None),
        }

        result = client._commit_multi_category_changes(categories_data)

        assert result is False
        mock_repo.git.commit.assert_not_called()


class TestSyncCategorySkipsPROnFalsePositive:
    """Test that sync_category_to_github skips PR creation when commit returns no changes."""

    def test_returns_none_on_false_positive(self, sample_diff: ModelReferenceDiff) -> None:
        """When _commit_changes returns False, sync returns None instead of creating a PR."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        client.settings = MagicMock()
        client.settings.dry_run = False
        client.settings.min_changes_threshold = 1

        # Mock all the internal methods
        mock_repo_settings = MagicMock()
        mock_repo_settings.repo_owner_and_name = "test/repo"

        with (
            patch.object(client, "_clone_repository"),
            patch.object(client, "_branch_operation") as mock_branch_ctx,
            patch.object(client, "_create_sync_branch", return_value="sync-branch"),
            patch.object(client, "_update_category_file"),
            patch.object(client, "_commit_changes", return_value=False),
            patch.object(client, "_push_branch") as mock_push,
            patch.object(client, "_create_pull_request") as mock_pr,
            patch.object(client, "cleanup"),
            patch("horde_model_reference.sync.github_client.horde_model_reference_settings") as mock_settings,
        ):
            mock_settings.get_repo_by_category.return_value = mock_repo_settings
            mock_branch_ctx.return_value.__enter__ = MagicMock()
            mock_branch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            result = client.sync_category_to_github(
                category=MODEL_REFERENCE_CATEGORY.image_generation,
                diff=sample_diff,
                primary_data={"model1": {"name": "model1"}},
            )

        assert result is None
        mock_push.assert_not_called()
        mock_pr.assert_not_called()


class TestSyncMultipleCategoriesSkipsPROnFalsePositive:
    """Test sync_multiple_categories_to_github handles false positives for multi-category PRs."""

    def test_returns_none_on_false_positive(self) -> None:
        """When _commit_multi_category_changes returns False, sync returns None."""
        client = GitHubSyncClient.__new__(GitHubSyncClient)
        client.settings = MagicMock()
        client.settings.dry_run = False
        client.settings.min_changes_threshold = 1

        diff = ModelReferenceDiff(category=MODEL_REFERENCE_CATEGORY.image_generation)
        diff.modified_models = {"m1": {"name": "m1"}}

        primary_data: dict[str, dict[str, Any]] = {"m1": {"name": "m1"}}
        categories_data: _CategoriesData = {
            MODEL_REFERENCE_CATEGORY.image_generation: (diff, primary_data, None),
        }

        mock_repo_settings = MagicMock()
        mock_repo_settings.repo_owner_and_name = "test/repo"

        with (
            patch.object(client, "_clone_repository"),
            patch.object(client, "_branch_operation") as mock_branch_ctx,
            patch.object(client, "_create_multi_category_sync_branch", return_value="sync-branch"),
            patch.object(client, "_update_category_file"),
            patch.object(client, "_commit_multi_category_changes", return_value=False),
            patch.object(client, "_push_branch") as mock_push,
            patch.object(client, "_create_pull_request") as mock_pr,
            patch.object(client, "cleanup"),
            patch("horde_model_reference.sync.github_client.horde_model_reference_settings") as mock_settings,
        ):
            mock_settings.get_repo_by_category.return_value = mock_repo_settings
            mock_branch_ctx.return_value.__enter__ = MagicMock()
            mock_branch_ctx.return_value.__exit__ = MagicMock(return_value=False)

            result = client.sync_multiple_categories_to_github(
                categories_data=categories_data,
                repo_name="test/repo",
            )

        assert result is None
        mock_push.assert_not_called()
        mock_pr.assert_not_called()


class TestFetchGithubDataUrlResolution:
    """Test that fetch_github_data resolves URLs for all syncable categories.

    The sync script calls fetch_github_data for every category except
    text_generation. Each must resolve to a valid URL from either
    legacy_image_model_github_urls or legacy_text_model_github_urls.
    """

    def _get_syncable_categories(self) -> list[MODEL_REFERENCE_CATEGORY]:
        """Return all categories except text_generation (which uses separate path)."""
        return [c for c in MODEL_REFERENCE_CATEGORY if c != MODEL_REFERENCE_CATEGORY.text_generation]

    def test_all_syncable_categories_have_urls(self) -> None:
        """Every non-text_generation category must resolve to a GitHub URL."""
        from horde_model_reference.path_consts import horde_model_reference_paths

        image_urls = horde_model_reference_paths.legacy_image_model_github_urls
        text_urls = horde_model_reference_paths.legacy_text_model_github_urls

        missing = []
        for category in self._get_syncable_categories():
            url = image_urls.get(category) or text_urls.get(category)
            if not url:
                missing.append(category)

        assert not missing, f"Categories with no GitHub URL: {missing}"

    def test_fetch_github_data_raises_for_unknown_category(self) -> None:
        """fetch_github_data raises ValueError for a category with no URL."""
        from scripts.sync.sync_github_references import GithubSynchronizer

        synchronizer = GithubSynchronizer()

        fake_cat = MagicMock(spec=MODEL_REFERENCE_CATEGORY)
        fake_cat.value = "nonexistent_category"

        with (
            patch("scripts.sync.sync_github_references.horde_model_reference_paths") as mock_paths,
        ):
            mock_paths.legacy_image_model_github_urls = {}
            mock_paths.legacy_text_model_github_urls = {}

            with pytest.raises(ValueError, match="No known GitHub URL"):
                synchronizer.fetch_github_data(category=fake_cat)
