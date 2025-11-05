"""GitHub client for automating PR creation and git operations."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlparse

from git import Repo
from github import Auth, Github, GithubException
from loguru import logger

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    GithubRepoSettings,
    horde_model_reference_paths,
    horde_model_reference_settings,
)
from horde_model_reference.sync.comparator import ModelReferenceDiff
from horde_model_reference.sync.config import github_app_settings
from horde_model_reference.sync.legacy_text_validator import LegacyTextValidator


class GitHubSyncClient:
    """Client for syncing model references to GitHub legacy repositories via PRs."""

    def __init__(
        self,
    ) -> None:
        """Initialize the GitHub sync client."""
        from horde_model_reference.sync.config import github_sync_settings

        self.settings = github_sync_settings

        self._github_client: Github | None
        self._installation_auth: Auth.AppInstallationAuth | None = None

        # Try GitHub App authentication first, then fall back to token
        if github_app_settings.is_configured():
            logger.info("Using GitHub App installation authentication")
            self._github_client, self._installation_auth = self._create_app_authenticated_client()
        elif self.settings.github_token:
            logger.info("Using GitHub token authentication")
            auth = Auth.Token(self.settings.github_token)
            self._github_client = Github(auth=auth)
        else:
            raise RuntimeError("No GitHub authentication method configured")

        self._temp_dir: Path | None = None
        self._current_repo: Repo | None = None
        self._is_persistent_dir: bool = False
        self._original_branch: str | None = None

    def _create_app_authenticated_client(self) -> tuple[Github, Auth.AppInstallationAuth]:
        """Create a GitHub client using App installation authentication.

        Returns:
            Tuple of (Github client, installation auth object) for later token access.

        Raises:
            RuntimeError: If GitHub App settings are not properly configured.
        """
        if not github_app_settings.is_configured():
            raise RuntimeError("GitHub App settings are not fully configured")

        try:
            private_key = github_app_settings.get_private_key_content()

            # Type assertions - is_configured() ensures these are not None
            assert github_app_settings.github_app_id is not None
            assert github_app_settings.github_installation_id is not None

            # Create App authentication
            app_auth = Auth.AppAuth(
                app_id=github_app_settings.github_app_id,
                private_key=private_key,
            )

            # Get installation authentication with all permissions
            # token_permissions can be specified if you want to limit permissions
            auth = app_auth.get_installation_auth(
                installation_id=github_app_settings.github_installation_id,
            )

            logger.debug(
                f"Created GitHub App installation auth for app_id={github_app_settings.github_app_id}, "
                f"installation_id={github_app_settings.github_installation_id}"
            )

            return Github(auth=auth), auth

        except Exception as e:
            logger.error(f"Failed to create GitHub App authenticated client: {e}")
            raise RuntimeError(f"GitHub App authentication failed: {e}") from e

    def __enter__(self) -> GitHubSyncClient:
        """Context manager entry - creates temporary directory."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - cleans up temporary directory."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources, preserving persistent directories.

        SIGNIFICANCE:
        - Persistent target directories must NOT be deleted (reused across runs)
        - Only temp directories (self._is_persistent_dir == False) should be removed
        """
        if self._current_repo:
            self._current_repo.close()
            self._current_repo = None

        if self._temp_dir and self._temp_dir.exists():
            if self._is_persistent_dir:
                logger.debug(f"Preserving persistent clone at {self._temp_dir}")
            else:
                logger.debug(f"Cleaning up temporary directory: {self._temp_dir}")
                try:
                    shutil.rmtree(self._temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {self._temp_dir}: {e}")

        self._temp_dir = None
        self._is_persistent_dir = False
        self._original_branch = None

    @contextmanager
    def _branch_operation(self) -> Generator[None, None, None]:
        """Context manager to ensure repository is returned to original branch.

        Captures the current branch before operations and restores it afterwards,
        even if an exception occurs. This prevents leaving repositories in a
        detached or temporary branch state.

        Usage:
            with self._branch_operation():
                # Create and work on temporary branch
                # Push changes and create PR
                # Branch will be restored automatically
        """
        if not self._current_repo:
            raise RuntimeError("No repository available for branch operation")

        try:
            self._original_branch = self._current_repo.active_branch.name
            logger.debug(f"Saved original branch: {self._original_branch}")
        except Exception as e:
            raise RuntimeError(f"Failed to determine current branch: {e}") from e

        try:
            yield
        finally:
            if self._current_repo and self._original_branch:
                try:
                    current_branch = self._current_repo.active_branch.name
                    if current_branch != self._original_branch:
                        logger.debug(f"Restoring original branch: {self._original_branch}")
                        self._current_repo.git.checkout(self._original_branch)
                except Exception as e:
                    logger.warning(f"Failed to restore original branch {self._original_branch}: {e}")

    def sync_category_to_github(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY,
        diff: ModelReferenceDiff,
        primary_data: dict[str, dict[str, Any]],
    ) -> str | None:
        """Sync a category's model references to GitHub by creating a PR.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to sync.
            diff: The detected differences for this category.
            primary_data: The complete PRIMARY data for this category (legacy format).

        Returns:
            The PR URL if created, None if no PR was needed or dry run.
        """
        if not diff.has_changes():
            logger.info(f"No changes detected for {category}, skipping PR creation")
            return None

        if diff.total_changes() < self.settings.min_changes_threshold:
            logger.info(
                f"Only {diff.total_changes()} changes for {category} "
                f"(threshold: {self.settings.min_changes_threshold}), skipping PR"
            )
            return None

        if self.settings.dry_run:
            logger.info(f"[DRY RUN] Would create PR for {category} with {diff.total_changes()} changes")
            logger.info(f"[DRY RUN] Summary:\n{diff.summary()}")
            return None

        try:
            github_repo_settings = horde_model_reference_settings.get_repo_by_category(category)
            repo_owner_and_name = github_repo_settings.repo_owner_and_name
            logger.info(f"Starting sync for {category} to {repo_owner_and_name}")

            self._clone_repository(github_repo_settings)

            with self._branch_operation():
                branch_name = self._create_sync_branch(category)
                self._update_category_file(category, primary_data)
                self._commit_changes(category, diff)
                self._push_branch(branch_name)
                pr_url = self._create_pull_request(
                    category, diff, repo_owner_and_name, branch_name, github_repo_settings
                )

            logger.success(f"Successfully created PR for {category}: {pr_url}")
            return pr_url

        except Exception as e:
            logger.error(f"Failed to sync {category} to GitHub: {e}")
            raise
        finally:
            self.cleanup()

    def sync_multiple_categories_to_github(
        self,
        *,
        repo_name: str,
        categories_data: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]],
    ) -> str | None:
        """Sync multiple categories to GitHub in a single PR.

        Args:
            repo_name: Repository in 'owner/repo' format.
            categories_data: Dict mapping categories to (diff, primary_data) tuples.

        Returns:
            The PR URL if created, None if no PR was needed or dry run.
        """
        total_changes = sum(diff.total_changes() for diff, _ in categories_data.values())

        if total_changes < self.settings.min_changes_threshold:
            logger.info(
                f"Only {total_changes} total changes across categories "
                f"(threshold: {self.settings.min_changes_threshold}), skipping PR"
            )
            return None

        if self.settings.dry_run:
            logger.info(
                f"[DRY RUN] Would create PR for {len(categories_data)} categories with {total_changes} changes"
            )
            for category, (diff, _) in categories_data.items():
                logger.info(f"[DRY RUN] {category}:\n{diff.summary()}")
            return None

        try:
            logger.info(f"Starting multi-category sync to {repo_name}")

            # Get the GitHub settings from the first category (all categories in this batch use same repo)
            first_category = next(iter(categories_data.keys()))
            github_repo_settings = horde_model_reference_settings.get_repo_by_category(first_category)

            self._clone_repository(github_repo_settings)

            with self._branch_operation():
                branch_name = self._create_multi_category_sync_branch(list(categories_data.keys()))

                for category, (diff, primary_data) in categories_data.items():
                    logger.info(f"Updating {category} with {diff.total_changes()} changes")
                    self._update_category_file(category, primary_data)

                self._commit_multi_category_changes(categories_data)
                self._push_branch(branch_name)
                pr_url = self._create_multi_category_pull_request(
                    categories_data, repo_name, branch_name, github_repo_settings
                )

            logger.success(f"Successfully created multi-category PR: {pr_url}")
            return pr_url

        except Exception as e:
            logger.error(f"Failed to sync multiple categories to GitHub: {e}")
            raise
        finally:
            self.cleanup()

    def _get_target_dir_for_repo(self, github_settings: GithubRepoSettings) -> Path | None:
        """Compute per-repository directory path within target_clone_dir.

        Args:
            github_settings: GitHub repository settings containing owner/name.

        Returns:
            Path to {target_clone_dir}/{owner}/{repo}/ or None if target_clone_dir not configured.

        Example:
            github_settings.owner = "Haidra-Org"
            github_settings.name = "AI-Horde-image-model-reference"
            target_clone_dir = "/path/to/clones"
            Returns: Path("/path/to/clones/Haidra-Org/AI-Horde-image-model-reference")
        """
        if not self.settings.target_clone_dir:
            return None

        base_dir = Path(self.settings.target_clone_dir)
        return base_dir / github_settings.repo_owner_and_name

    def _verify_existing_repo(self, repo_path: Path, expected_github_settings: GithubRepoSettings) -> None:
        """Verify existing directory matches expected repository identity.

        Extracts and compares repository owner/name from git remote URL and current branch.
        Ignores authentication information in URLs when comparing.

        Args:
            repo_path: Path to existing git repository.
            expected_github_settings: Expected GitHub repository settings.

        Raises:
            RuntimeError: If directory is not a git repository.
            ValueError: If owner/repo or branch doesn't match expected values.
        """
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"Target directory exists but is not a git repository: {repo_path}")

        try:
            repo = Repo(repo_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open git repository at {repo_path}: {e}") from e

        try:
            remote_url = repo.remote("origin").url
        except Exception as e:
            raise RuntimeError(f"Failed to get remote URL from repository at {repo_path}: {e}") from e

        # Strip any authentication information before parsing
        clean_url = self._strip_auth_from_url(remote_url)
        actual_repo_name = self._parse_repo_name_from_url(clean_url)
        expected_repo_name = expected_github_settings.repo_owner_and_name

        if actual_repo_name != expected_repo_name:
            raise ValueError(
                f"Repository mismatch - Expected: {expected_repo_name}, Found: {actual_repo_name}. "
                f"Aborting to prevent data corruption."
            )

        logger.info(f"✓ Repository: {actual_repo_name} (matches expected)")

    def _strip_auth_from_url(self, url: str) -> str:
        """Strip authentication information from a git remote URL.

        Args:
            url: Git remote URL that may contain authentication credentials.

        Returns:
            URL with authentication information removed.

        Example:
            "https://token@github.com/owner/repo.git" -> "https://github.com/owner/repo.git"
            "https://user:pass@github.com/owner/repo.git" -> "https://github.com/owner/repo.git"
        """
        if "github.com" in url and url.startswith("https://") and "@github.com" in url:
            return "https://github.com/" + url.split("@github.com/", 1)[1]
        return url

    def _parse_repo_name_from_url(self, url: str) -> str:
        """Parse owner/repo from a git remote URL.

        Args:
            url: Git remote URL (https:// or git@ format), should be cleaned of auth first.

        Returns:
            Repository name in 'owner/repo' format.

        Example:
            "https://github.com/Haidra-Org/AI-Horde-image-model-reference.git"
            -> "Haidra-Org/AI-Horde-image-model-reference"
        """
        if "github.com" in url:
            if url.startswith("https://"):
                repo_path = url.replace("https://github.com/", "").replace(".git", "")
            elif url.startswith("git@"):
                repo_path = url.split(":")[-1].replace(".git", "")
            else:
                repo_path = url.replace("github.com/", "").replace(".git", "")
            return repo_path.strip("/")
        raise ValueError(f"Unable to parse repository name from URL: {url}")

    def _check_for_local_changes(self, repo: Repo) -> bool:
        """Check if repository has uncommitted changes.

        Args:
            repo: GitPython Repo object.

        Returns:
            True if uncommitted changes or untracked files exist.
        """
        return repo.is_dirty(untracked_files=True)

    def _reset_existing_repo(self, repo: Repo, github_settings: GithubRepoSettings) -> None:
        """Reset existing repository to match remote branch state.

        Sequence:
        1. Fetch latest from origin
        2. Check for local changes
        3. If changes exist, prompt user to continue or abort
        4. Hard reset to origin/{branch}
        5. Clean untracked files

        Args:
            repo: GitPython Repo object for repository to reset.
            github_settings: GitHub repository settings containing branch name.

        Raises:
            RuntimeError: If user chooses to abort or on git operation failure.
        """
        logger.info("Fetching latest changes from origin...")
        try:
            repo.remotes.origin.fetch()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch from origin: {e}") from e

        if self._check_for_local_changes(repo):
            logger.warning(f"Local changes detected in {repo.working_dir}")

            changed_files = []
            if repo.is_dirty():
                changed_files.extend([item.a_path for item in repo.index.diff(None) if item.a_path])
                changed_files.extend([item.a_path for item in repo.index.diff("HEAD") if item.a_path])
            if repo.untracked_files:
                changed_files.extend(repo.untracked_files)

            if changed_files:
                logger.warning("Changed files:")
                for file_path in sorted(set(changed_files))[:10]:
                    logger.warning(f"  {file_path}")
                if len(changed_files) > 10:
                    logger.warning(f"  ... and {len(changed_files) - 10} more")

            try:
                response = input("Continue and discard all local changes? (y/n): ").strip().lower()
                if response != "y":
                    raise RuntimeError("Aborting to preserve local changes")
            except EOFError as e:
                raise RuntimeError("Aborting: no user input available") from e

        target_ref = f"origin/{github_settings.branch}"
        logger.info(f"Resetting to {target_ref}...")

        try:
            repo.head.reset(target_ref, index=True, working_tree=True)
            repo.git.clean("-fdx")
            logger.info(f"✓ Reset to {target_ref} and cleaned untracked files")
        except Exception as e:
            raise RuntimeError(f"Failed to reset repository: {e}") from e

    def _clone_repository(self, github_settings: GithubRepoSettings) -> None:
        """Clone or reuse GitHub repository.

        SIGNIFICANCE:
        - Uses HordeModelReferenceSettings as single source of truth for GitHub URLs
        - github_settings contains owner, name, branch, and constructs all URLs
        - Multiple categories may map to same repo (persistence is per-repo, not per-category)

        Flow:
        1. Determine target directory via _get_target_dir_for_repo(github_settings)
        2. If no persistent dir configured: use temp directory logic
        3. If persistent dir exists: verify identity and reset
        4. If persistent dir doesn't exist: clone to persistent dir

        Args:
            github_settings: GitHub repository settings from HordeModelReferenceSettings.
        """
        target_dir = self._get_target_dir_for_repo(github_settings)
        repo_name = github_settings.repo_owner_and_name

        if target_dir is None:
            if self.settings.sync_temp_dir:
                temp_base = Path(self.settings.sync_temp_dir)
                temp_base.mkdir(parents=True, exist_ok=True)
                self._temp_dir = Path(tempfile.mkdtemp(dir=temp_base))
            else:
                self._temp_dir = Path(tempfile.mkdtemp())

            logger.debug(f"Created temporary directory: {self._temp_dir}")
            self._is_persistent_dir = False

            repo_url = github_settings.git_clone_url
            logger.info(f"Cloning {repo_url} to {self._temp_dir}")

            try:
                self._current_repo = Repo.clone_from(
                    url=repo_url,
                    to_path=self._temp_dir,
                    branch=github_settings.branch,
                    depth=1,
                )
                logger.debug(f"Successfully cloned {repo_name}")
            except Exception as e:
                logger.error(f"Failed to clone repository {repo_name}: {e}")
                raise

        elif target_dir.exists():
            logger.info(f"Found existing clone at {target_dir}, verifying identity...")

            self._verify_existing_repo(target_dir, github_settings)

            try:
                self._current_repo = Repo(target_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to open existing repository at {target_dir}: {e}") from e

            logger.info(f"Verified repository identity: {repo_name} (branch: {github_settings.branch})")

            self._reset_existing_repo(self._current_repo, github_settings)

            self._temp_dir = target_dir
            self._is_persistent_dir = True

        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            repo_url = github_settings.git_clone_url
            logger.info(f"Cloning {repo_url} to {target_dir}...")

            try:
                self._current_repo = Repo.clone_from(
                    url=repo_url,
                    to_path=target_dir,
                    branch=github_settings.branch,
                )
                logger.info(f"Successfully cloned {repo_name} (branch: {github_settings.branch})")
            except Exception as e:
                logger.error(f"Failed to clone repository {repo_name}: {e}")
                raise

            self._temp_dir = target_dir
            self._is_persistent_dir = True

    def _create_sync_branch(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Create a new branch for the sync operation.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.

        Returns:
            The name of the created branch.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"sync/{category}/{timestamp}"

        logger.debug(f"Creating branch: {branch_name}")
        self._current_repo.git.checkout("-b", branch_name)

        return branch_name

    def _update_category_file(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        primary_data: dict[str, dict[str, Any]],
    ) -> None:
        """Update the category file with PRIMARY data.

        SIGNIFICANCE:
        - For text_generation category, GitHub repos use 'db.json', not 'text_generation.json'
        - We must write to the filename that exists in the GitHub repository
        - This follows the legacy naming convention used by the GitHub repos
        - For text_generation, applies LegacyTextValidator to ensure convert.py compatibility

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to update.
            primary_data: The complete PRIMARY data in legacy format.
        """
        if not self._current_repo or not self._temp_dir:
            raise RuntimeError("No repository cloned")

        # Use GitHub legacy filename for text_generation category
        filename: str
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            filename = "db.json"
            logger.debug(f"Using legacy GitHub filename 'db.json' for {category}")
        else:
            filename = str(horde_model_reference_paths.get_model_reference_filename(category))

        file_path = self._temp_dir / filename

        logger.debug(f"Updating {file_path} with PRIMARY data")

        # Apply legacy text validation for text_generation category
        # This ensures the data matches convert.py expectations (generation_params.json,
        # defaults.json, backend prefixes, tag auto-generation, etc.)
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            logger.debug("Applying LegacyTextValidator for text_generation category")
            try:
                validator = LegacyTextValidator()
                primary_data = validator.validate_and_transform(primary_data)
                logger.debug(f"LegacyTextValidator applied: {len(primary_data)} records after transformation")
            except Exception as e:
                logger.error(f"LegacyTextValidator failed: {e}")
                raise

        serialized_data = json.dumps(primary_data, indent=4, sort_keys=False)
        serialized_data = serialized_data + "\n"

        file_path.write_text(serialized_data, encoding="utf-8")
        logger.debug(f"Wrote {len(primary_data)} models to {file_path}")

    def _commit_changes(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        diff: ModelReferenceDiff,
    ) -> None:
        """Commit the changes to the repository.

        Uses --no-gpg-sign to bypass GPG signing requirements for automated commits.
        This prevents issues when running in environments without GPG configured.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.
            diff: The diff summary for generating commit message.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned")

        self._current_repo.git.add(".")

        if not self._current_repo.is_dirty():
            logger.warning("No changes to commit")
            return

        commit_message = self._generate_commit_message(category, diff)
        logger.debug(f"Committing with message:\n{commit_message}")

        self._current_repo.git.commit("-m", commit_message, "--no-gpg-sign")
        logger.debug("Changes committed successfully")

    def _push_branch(self, branch_name: str) -> None:
        """Push the branch to the remote repository.

        Args:
            branch_name: The name of the branch to push.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned?")

        try:
            repo_url_with_auth = self._get_authenticated_repo_url()

            logger.info(f"Pushing branch {branch_name}")
            self._current_repo.git.push(repo_url_with_auth, branch_name)
            logger.debug("Branch pushed successfully")
        except Exception as e:
            logger.error(f"Failed to push branch {branch_name}: {e}")
            raise

    def _get_authenticated_repo_url(self) -> str:
        """Get the repository URL with authentication token.

        Returns:
            The authenticated repository URL.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned")

        remote_url = self._current_repo.remote("origin").url

        # Strip any existing authentication before adding new token
        clean_url = self._strip_auth_from_url(remote_url)

        hostname = urlparse(clean_url).hostname
        if hostname and hostname.lower() == "github.com":
            repo_path = self._parse_repo_name_from_url(clean_url)

            # Try GitHub App authentication first
            if self._installation_auth is not None:
                try:
                    # Use the installation auth object we stored during initialization
                    token = self._installation_auth.token
                    return f"https://x-access-token:{token}@github.com/{repo_path}.git"
                except Exception as e:
                    logger.warning(f"Failed to get GitHub App token for push: {e}")
                    # Fall through to token auth

            # Fall back to personal access token
            if self.settings.github_token:
                return f"https://{self.settings.github_token}@github.com/{repo_path}.git"

        return remote_url

    def _find_existing_sync_prs(self, repo_name: str, category: MODEL_REFERENCE_CATEGORY | None = None) -> list[Any]:
        """Find existing open PRs created by the sync service.

        Args:
            repo_name: Repository in 'owner/repo' format.
            category: Optional category to filter PRs. If None, finds all sync PRs.

        Returns:
            List of open pull request objects created by the sync service.
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        try:
            repo = self._github_client.get_repo(repo_name)
            open_prs = repo.get_pulls(state="open", sort="created", direction="desc")

            sync_prs = []
            for pr in open_prs:
                # Check if PR is from a sync branch and matches category filter (if specified)
                if pr.head.ref.startswith("sync/") and (
                    category is None or f"sync/{category}/" in pr.head.ref or "sync/multi-category/" in pr.head.ref
                ):
                    sync_prs.append(pr)

            return sync_prs

        except GithubException as e:
            logger.warning(f"Failed to find existing sync PRs: {e}")
            return []

    def _close_existing_sync_prs(self, repo_name: str, category: MODEL_REFERENCE_CATEGORY | None = None) -> None:
        """Close existing open PRs created by the sync service.

        Args:
            repo_name: Repository in 'owner/repo' format.
            category: Optional category to filter PRs. If None, closes all sync PRs.
        """
        existing_prs = self._find_existing_sync_prs(repo_name, category)

        if not existing_prs:
            logger.debug(f"No existing sync PRs found for {repo_name}")
            return

        logger.info(f"Found {len(existing_prs)} existing sync PR(s) to close")

        for pr in existing_prs:
            try:
                # Add a comment explaining why it's being closed
                comment = (
                    "This PR is being automatically closed because a new sync operation has been initiated.\n\n"
                    "A new PR with updated changes will be created shortly."
                )
                pr.create_issue_comment(comment)

                # Close the PR
                pr.edit(state="closed")
                logger.info(f"Closed PR #{pr.number}: {pr.title}")

            except GithubException as e:
                logger.warning(f"Failed to close PR #{pr.number}: {e}")

    def _create_pull_request(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        diff: ModelReferenceDiff,
        repo_name: str,
        branch_name: str,
        github_settings: GithubRepoSettings,
    ) -> str:
        """Create a pull request for the sync.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.
            diff: The diff summary for generating PR description.
            repo_name: Repository in 'owner/repo' format.
            branch_name: The name of the branch to create PR from.
            github_settings: GitHub repository settings containing branch name.

        Returns:
            The URL of the created PR.
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        # Close any existing sync PRs for this category
        self._close_existing_sync_prs(repo_name, category)

        try:
            repo = self._github_client.get_repo(repo_name)
            title = self._generate_pr_title(category)
            body = self._generate_pr_body(category, diff)

            logger.info(f"Creating PR: {title}")

            pr = repo.create_pull(
                title=title,
                body=body,
                head=branch_name,
                base=github_settings.branch,
            )

            if self.settings.pr_labels:
                pr.add_to_labels(*self.settings.pr_labels)

            if self.settings.pr_reviewers:
                try:
                    pr.create_review_request(reviewers=self.settings.pr_reviewers)
                except GithubException as e:
                    logger.warning(f"Failed to assign reviewers: {e}")

            if self.settings.pr_auto_assign_team:
                try:
                    team_slug = self.settings.pr_auto_assign_team.split("/")[-1]
                    pr.create_review_request(team_reviewers=[team_slug])
                except GithubException as e:
                    logger.warning(f"Failed to assign team: {e}")

            return pr.html_url

        except GithubException as e:
            logger.error(f"Failed to create PR: {e}")
            raise

    def _generate_commit_message(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        diff: ModelReferenceDiff,
    ) -> str:
        """Generate a commit message from the diff.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.
            diff: The diff summary.

        Returns:
            The commit message.
        """
        lines = [f"Sync {category} from PRIMARY instance"]
        lines.append("")

        if diff.added_models:
            lines.append(f"Added {len(diff.added_models)} models:")
            for model_name in sorted(diff.added_models.keys())[:10]:
                lines.append(f"  + {model_name}")
            if len(diff.added_models) > 10:
                lines.append(f"  ... and {len(diff.added_models) - 10} more")

        if diff.removed_models:
            lines.append(f"\nRemoved {len(diff.removed_models)} models:")
            for model_name in sorted(diff.removed_models.keys())[:10]:
                lines.append(f"  - {model_name}")
            if len(diff.removed_models) > 10:
                lines.append(f"  ... and {len(diff.removed_models) - 10} more")

        if diff.modified_models:
            lines.append(f"\nModified {len(diff.modified_models)} models:")
            for model_name in sorted(diff.modified_models.keys())[:10]:
                lines.append(f"  ~ {model_name}")
            if len(diff.modified_models) > 10:
                lines.append(f"  ... and {len(diff.modified_models) - 10} more")

        lines.append("")
        lines.append("Generated by horde-model-reference GitHub sync service")

        return "\n".join(lines)

    def _generate_pr_title(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        """Generate a PR title.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.

        Returns:
            The PR title.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"Sync {category} from PRIMARY instance - {date_str}"

    def _generate_pr_body(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        diff: ModelReferenceDiff,
    ) -> str:
        """Generate a PR description from the diff.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category being synced.
            diff: The diff summary.

        Returns:
            The PR body in Markdown format.
        """
        lines = [
            "## Automated Sync from PRIMARY Instance",
            "",
            f"This PR synchronizes the `{category}` model references from the PRIMARY instance.",
            "",
            "### Changes Summary",
            "",
        ]

        if self.settings.primary_api_url:
            lines.append(f"**Source:** {self.settings.primary_api_url}")
            lines.append("")

        lines.append(f"- **Added:** {len(diff.added_models)} models")
        lines.append(f"- **Removed:** {len(diff.removed_models)} models")
        lines.append(f"- **Modified:** {len(diff.modified_models)} models")
        lines.append(f"- **Total Changes:** {diff.total_changes()}")
        lines.append("")

        if diff.added_models:
            lines.append("#### Added Models")
            lines.append("")
            for model_name in sorted(diff.added_models.keys())[:20]:
                lines.append(f"- `{model_name}`")
            if len(diff.added_models) > 20:
                lines.append(f"- ... and {len(diff.added_models) - 20} more")
            lines.append("")

        if diff.removed_models:
            lines.append("#### Removed Models")
            lines.append("")
            for model_name in sorted(diff.removed_models.keys())[:20]:
                lines.append(f"- `{model_name}`")
            if len(diff.removed_models) > 20:
                lines.append(f"- ... and {len(diff.removed_models) - 20} more")
            lines.append("")

        if diff.modified_models:
            lines.append("#### Modified Models")
            lines.append("")
            for model_name in sorted(diff.modified_models.keys())[:20]:
                lines.append(f"- `{model_name}`")
            if len(diff.modified_models) > 20:
                lines.append(f"- ... and {len(diff.modified_models) - 20} more")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*This PR was automatically generated by the horde-model-reference GitHub sync service.*")
        lines.append("")
        lines.append(
            "Please review the changes carefully before merging. "
            "If you notice any issues, contact the PRIMARY instance administrator."
        )

        return "\n".join(lines)

    def _create_multi_category_sync_branch(self, categories: list[MODEL_REFERENCE_CATEGORY]) -> str:
        """Create a new branch for multi-category sync operation.

        Args:
            categories: The list of categories being synced.

        Returns:
            The name of the created branch.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"sync/multi-category/{timestamp}"

        logger.debug(f"Creating multi-category branch: {branch_name}")
        self._current_repo.git.checkout("-b", branch_name)

        return branch_name

    def _commit_multi_category_changes(
        self,
        categories_data: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]],
    ) -> None:
        """Commit changes for multiple categories.

        Uses --no-gpg-sign to bypass GPG signing requirements for automated commits.
        This prevents issues when running in environments without GPG configured.

        Args:
            categories_data: Dict mapping categories to (diff, primary_data) tuples.
        """
        if not self._current_repo:
            raise RuntimeError("No repository cloned")

        self._current_repo.git.add(".")

        if not self._current_repo.is_dirty():
            logger.warning("No changes to commit")
            return

        commit_message = self._generate_multi_category_commit_message(categories_data)
        logger.debug(f"Committing with message:\n{commit_message}")

        self._current_repo.git.commit("-m", commit_message, "--no-gpg-sign")
        logger.debug("Changes committed successfully")

    def _generate_multi_category_commit_message(
        self,
        categories_data: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]],
    ) -> str:
        """Generate a commit message for multi-category sync.

        Args:
            categories_data: Dict mapping categories to (diff, primary_data) tuples.

        Returns:
            The commit message.
        """
        category_names = ", ".join(str(cat) for cat in sorted(categories_data.keys()))
        total_changes = sum(diff.total_changes() for diff, _ in categories_data.values())

        lines = ["Sync multiple categories from PRIMARY instance"]
        lines.append("")
        lines.append(f"Categories: {category_names}")
        lines.append(f"Total changes: {total_changes}")
        lines.append("")

        for category in sorted(categories_data.keys()):
            diff, _ = categories_data[category]
            lines.append(f"## {category}")

            if diff.added_models:
                lines.append(f"Added {len(diff.added_models)} models:")
                for model_name in sorted(diff.added_models.keys())[:5]:
                    lines.append(f"  + {model_name}")
                if len(diff.added_models) > 5:
                    lines.append(f"  ... and {len(diff.added_models) - 5} more")

            if diff.removed_models:
                lines.append(f"Removed {len(diff.removed_models)} models:")
                for model_name in sorted(diff.removed_models.keys())[:5]:
                    lines.append(f"  - {model_name}")
                if len(diff.removed_models) > 5:
                    lines.append(f"  ... and {len(diff.removed_models) - 5} more")

            if diff.modified_models:
                lines.append(f"Modified {len(diff.modified_models)} models:")
                for model_name in sorted(diff.modified_models.keys())[:5]:
                    lines.append(f"  ~ {model_name}")
                if len(diff.modified_models) > 5:
                    lines.append(f"  ... and {len(diff.modified_models) - 5} more")

            lines.append("")

        lines.append("Generated by horde-model-reference GitHub sync service")

        return "\n".join(lines)

    def _create_multi_category_pull_request(
        self,
        categories_data: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]],
        repo_name: str,
        branch_name: str,
        github_settings: GithubRepoSettings,
    ) -> str:
        """Create a pull request for multi-category sync.

        Args:
            categories_data: Dict mapping categories to (diff, primary_data) tuples.
            repo_name: Repository in 'owner/repo' format.
            branch_name: The name of the branch to create PR from.
            github_settings: GitHub repository settings containing branch name.

        Returns:
            The URL of the created PR.
        """
        if not self._github_client:
            raise RuntimeError("GitHub client not initialized")

        # Close any existing sync PRs for this repository
        # For multi-category PRs, we close all sync PRs regardless of category
        self._close_existing_sync_prs(repo_name, category=None)

        try:
            repo = self._github_client.get_repo(repo_name)
            title = self._generate_multi_category_pr_title(list(categories_data.keys()))
            body = self._generate_multi_category_pr_body(categories_data)

            logger.info(f"Creating multi-category PR: {title}")

            pr = repo.create_pull(
                title=title,
                body=body,
                head=branch_name,
                base=github_settings.branch,
            )

            if self.settings.pr_labels:
                pr.add_to_labels(*self.settings.pr_labels)

            if self.settings.pr_reviewers:
                try:
                    pr.create_review_request(reviewers=self.settings.pr_reviewers)
                except GithubException as e:
                    logger.warning(f"Failed to assign reviewers: {e}")

            if self.settings.pr_auto_assign_team:
                try:
                    team_slug = self.settings.pr_auto_assign_team.split("/")[-1]
                    pr.create_review_request(team_reviewers=[team_slug])
                except GithubException as e:
                    logger.warning(f"Failed to assign team: {e}")

            return pr.html_url

        except GithubException as e:
            logger.error(f"Failed to create PR: {e}")
            raise

    def _generate_multi_category_pr_title(self, categories: list[MODEL_REFERENCE_CATEGORY]) -> str:
        """Generate a PR title for multi-category sync.

        Args:
            categories: The list of categories being synced.

        Returns:
            The PR title.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"Sync multiple categories from PRIMARY instance - {date_str}"

    def _generate_multi_category_pr_body(
        self,
        categories_data: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]],
    ) -> str:
        """Generate a PR description for multi-category sync.

        Args:
            categories_data: Dict mapping categories to (diff, primary_data) tuples.

        Returns:
            The PR body in Markdown format.
        """
        total_added = sum(len(diff.added_models) for diff, _ in categories_data.values())
        total_removed = sum(len(diff.removed_models) for diff, _ in categories_data.values())
        total_modified = sum(len(diff.modified_models) for diff, _ in categories_data.values())
        total_changes = total_added + total_removed + total_modified

        lines = [
            "## Automated Multi-Category Sync from PRIMARY Instance",
            "",
            f"This PR synchronizes **{len(categories_data)} categories** from the PRIMARY instance.",
            "",
            f"**Categories:** {', '.join(f'`{cat}`' for cat in sorted(categories_data.keys()))}",
            "",
            "### Overall Changes Summary",
            "",
        ]

        if self.settings.primary_api_url:
            lines.append(f"**Source:** {self.settings.primary_api_url}")
            lines.append("")

        lines.append(f"- **Total Added:** {total_added} models")
        lines.append(f"- **Total Removed:** {total_removed} models")
        lines.append(f"- **Total Modified:** {total_modified} models")
        lines.append(f"- **Total Changes:** {total_changes}")
        lines.append("")

        for category in sorted(categories_data.keys()):
            diff, _ = categories_data[category]
            lines.append(f"### {category}")
            lines.append("")
            lines.append(f"- **Added:** {len(diff.added_models)} models")
            lines.append(f"- **Removed:** {len(diff.removed_models)} models")
            lines.append(f"- **Modified:** {len(diff.modified_models)} models")
            lines.append("")

            if diff.added_models:
                lines.append("**Added Models:**")
                for model_name in sorted(diff.added_models.keys())[:10]:
                    lines.append(f"- `{model_name}`")
                if len(diff.added_models) > 10:
                    lines.append(f"- ... and {len(diff.added_models) - 10} more")
                lines.append("")

            if diff.removed_models:
                lines.append("**Removed Models:**")
                for model_name in sorted(diff.removed_models.keys())[:10]:
                    lines.append(f"- `{model_name}`")
                if len(diff.removed_models) > 10:
                    lines.append(f"- ... and {len(diff.removed_models) - 10} more")
                lines.append("")

            if diff.modified_models:
                lines.append("**Modified Models:**")
                for model_name in sorted(diff.modified_models.keys())[:10]:
                    lines.append(f"- `{model_name}`")
                if len(diff.modified_models) > 10:
                    lines.append(f"- ... and {len(diff.modified_models) - 10} more")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("*This PR was automatically generated by the horde-model-reference GitHub sync service.*")
        lines.append("")
        lines.append(
            "Please review the changes carefully before merging. "
            "If you notice any issues, contact the PRIMARY instance administrator."
        )

        return "\n".join(lines)
