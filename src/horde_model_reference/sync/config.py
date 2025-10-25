"""Configuration settings for GitHub synchronization service."""

from __future__ import annotations

import os

from loguru import logger
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from horde_model_reference import MODEL_REFERENCE_CATEGORY


class GithubAppSettings(BaseSettings):
    """Settings for GitHub App authentication."""

    github_app_id: int | None = None
    """GitHub App ID for authentication."""

    github_installation_id: int | None = None
    """GitHub App Installation ID for the target organization/repository."""

    github_app_private_key: str | None = None
    """GitHub App private key content in PEM format. Use this for inline key content."""

    github_app_private_key_path: str | None = None
    """Path to GitHub App private key file (.pem). Use this to load key from a file."""

    def is_configured(self) -> bool:
        """Check if GitHub App authentication is fully configured.

        Returns:
            True if app_id, installation_id, and either private_key or private_key_path is set.
        """
        has_key = self.github_app_private_key is not None or self.github_app_private_key_path is not None
        return self.github_app_id is not None and self.github_installation_id is not None and has_key

    def get_private_key_content(self) -> str:
        """Get the private key content, loading from file if necessary.

        Returns:
            The private key content in PEM format.

        Raises:
            ValueError: If neither private_key nor private_key_path is configured,
                       or if both are set (mutually exclusive), or if file cannot be read.
        """
        # Check for mutually exclusive configuration
        if self.github_app_private_key and self.github_app_private_key_path:
            raise ValueError(
                "Both GITHUB_APP_PRIVATE_KEY and GITHUB_APP_PRIVATE_KEY_PATH are set. "
                "These are mutually exclusive. Please use only one."
            )

        if not self.github_app_private_key and not self.github_app_private_key_path:
            raise ValueError(
                "GitHub App private key is not configured. "
                "Set either GITHUB_APP_PRIVATE_KEY (for inline key) or "
                "GITHUB_APP_PRIVATE_KEY_PATH (for file path)."
            )

        # Handle file path
        if self.github_app_private_key_path:
            logger.debug(f"Loading GitHub App private key from file: {self.github_app_private_key_path}")
            if not os.path.exists(self.github_app_private_key_path):
                raise ValueError(
                    f"Private key file not found at: {self.github_app_private_key_path}. "
                    "Ensure the file exists and the path is correct."
                )
            try:
                with open(self.github_app_private_key_path, encoding="utf-8") as f:
                    content = f.read()
                # Ensure the key has proper newlines and format
                if not content.strip().startswith("-----BEGIN"):
                    logger.error(f"Private key file does not start with -----BEGIN. First 50 chars: {content[:50]!r}")
                    raise ValueError("Private key file does not appear to be in PEM format")
                logger.debug("Successfully loaded and validated private key from file")
                return content
            except Exception as e:
                logger.error(f"Failed to read private key from file {self.github_app_private_key_path}: {e}")
                raise ValueError(
                    f"Failed to read private key from file {self.github_app_private_key_path}: {e}"
                ) from e

        # Handle inline key content
        logger.debug("Using inline GitHub App private key content")
        assert self.github_app_private_key is not None  # Type checker hint - we know it's not None here
        content = self.github_app_private_key

        # Log the first part to help debug
        first_chars = content[:50] if len(content) > 50 else content
        logger.debug(f"Private key content starts with: {first_chars!r}")

        if not content.strip().startswith("-----BEGIN"):
            logger.error(f"Private key does not start with -----BEGIN. Value looks like: {first_chars!r}.")
            raise ValueError("Private key does not appear to be in PEM format (should start with -----BEGIN).")
        return content


github_app_settings = GithubAppSettings()


class HordeGitHubSyncSettings(BaseSettings):
    """Settings for syncing model references from PRIMARY to GitHub legacy repos."""

    model_config = SettingsConfigDict(
        env_prefix="HORDE_GITHUB_SYNC_",
        use_attribute_docstrings=True,
    )

    suppress_meta_warnings: bool = True
    """Suppress pydantic meta warnings about unknown fields, for example, for use in scripts."""

    primary_api_url: str | None = None
    """PRIMARY instance v1 API base URL (e.g., https://stablehorde.net/api). Required for sync to work."""

    primary_api_timeout: int = 30
    """Timeout in seconds for HTTP requests to PRIMARY API."""

    github_token: str | None = None
    """GitHub personal access token with repo write permissions. Required for creating PRs. \
Set via HORDE_GITHUB_SYNC_GITHUB_TOKEN or GITHUB_TOKEN environment variable."""

    categories_to_sync: list[str] | None = None
    """Whitelist of categories to sync (e.g., ['image_generation', 'text_generation']). \
If None, syncs all available categories."""

    pr_reviewers: list[str] | None = None
    """Default GitHub usernames to assign as PR reviewers (e.g., ['username1', 'username2'])."""

    pr_labels: list[str] = ["automated", "sync", "ready-for-review"]
    """Default labels to apply to sync PRs."""

    pr_auto_assign_team: str | None = None
    """GitHub team to auto-assign for review (e.g., 'org-name/team-name')."""

    min_changes_threshold: int = 1
    """Minimum number of changes required to trigger PR creation."""

    sync_temp_dir: str | None = None
    """Temporary directory for git operations. If None, uses system temp directory."""

    target_clone_dir: str | None = None
    """Base directory for persistent repository clones. If set, repos will be cloned to \
{target_clone_dir}/{owner}/{repo}/ and reused across sync runs. The repository identity \
(owner/repo/branch) will be verified on each run and must match the configured github_image_repo, \
github_text_repo, and github_branch settings."""

    dry_run: bool = False
    """If True, perform comparison but don't create PRs. Useful for testing."""

    verbose_logging: bool = False
    """Enable detailed logging for sync operations."""

    watch_mode: bool = False
    """Enable watch mode to continuously monitor for metadata changes and trigger syncs."""

    watch_interval_seconds: int = 60
    """Interval in seconds between metadata checks in watch mode."""

    watch_initial_delay_seconds: int = 0
    """Initial delay in seconds before starting watch mode polling (useful for startup synchronization)."""

    watch_enable_startup_sync: bool = False
    """If True, perform a full sync immediately when watch mode starts, before entering the watch loop."""

    @model_validator(mode="after")
    def validate_sync_configuration(self) -> HordeGitHubSyncSettings:
        """Validate sync configuration and provide helpful warnings."""
        if not self.suppress_meta_warnings:
            if not self.primary_api_url and not self.dry_run:
                logger.error(
                    "PRIMARY API URL is not configured. "
                    "Set HORDE_GITHUB_SYNC_PRIMARY_API_URL to enable sync operations. "
                    "Example: export HORDE_GITHUB_SYNC_PRIMARY_API_URL=https://stablehorde.net/api"
                )

            if not self.github_token and not self.dry_run:
                logger.warning(
                    "GitHub token is not configured. "
                    "PR creation will fail without authentication. "
                    "Set HORDE_GITHUB_SYNC_GITHUB_TOKEN or GITHUB_TOKEN environment variable, "
                    "or configure GitHub App (GITHUB_APP_ID, GITHUB_APP_INSTALLATION_ID, GITHUB_APP_PRIVATE_KEY)."
                )

            if self.categories_to_sync:
                invalid_categories = [
                    cat for cat in self.categories_to_sync if cat not in MODEL_REFERENCE_CATEGORY.__members__
                ]
                if invalid_categories:
                    logger.warning(
                        f"Invalid categories in categories_to_sync: {invalid_categories}. "
                        f"Valid categories: {list(MODEL_REFERENCE_CATEGORY.__members__.keys())}"
                    )

        if self.min_changes_threshold < 1:
            logger.warning(f"min_changes_threshold is {self.min_changes_threshold}, but must be >= 1. Setting to 1.")
            self.min_changes_threshold = 1

        if self.verbose_logging:
            logger.info("Verbose logging enabled for GitHub sync operations")

        # Handle GitHub authentication (token or app)
        has_token = False
        has_app = False

        if self.github_token is None:
            logger.debug("Loading GitHub token from GITHUB_TOKEN environment variable for authentication")
            self.github_token = os.getenv("GITHUB_TOKEN")
        else:
            if "GITHUB_TOKEN" not in os.environ:
                os.environ["GITHUB_TOKEN"] = self.github_token
            else:
                logger.warning(
                    "Both HORDE_GITHUB_SYNC_GITHUB_TOKEN and GITHUB_TOKEN are set. "
                    "Using HORDE_GITHUB_SYNC_GITHUB_TOKEN."
                )

        if self.github_token is not None:
            logger.debug("GitHub token successfully loaded for authentication")
            has_token = True

        # Check for GitHub App configuration
        if github_app_settings.github_app_id is None:
            github_app_settings.github_app_id = int(os.getenv("GITHUB_APP_ID", "0")) or None
        if github_app_settings.github_installation_id is None:
            github_app_settings.github_installation_id = int(os.getenv("GITHUB_APP_INSTALLATION_ID", "0")) or None

        # Load private key - check for path first, then inline key
        if github_app_settings.github_app_private_key_path is None:
            github_app_settings.github_app_private_key_path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")

        if github_app_settings.github_app_private_key is None:
            private_key_str = os.getenv("GITHUB_APP_PRIVATE_KEY")
            if private_key_str:
                # Handle escaped newlines in environment variables
                # Environment variables might contain literal \n which need to be converted to actual newlines
                github_app_settings.github_app_private_key = private_key_str.replace("\\n", "\n")
                logger.debug("Loaded GitHub App private key from GITHUB_APP_PRIVATE_KEY environment variable")

        if github_app_settings.is_configured():
            logger.debug("GitHub App authentication configured")
            has_app = True

        if has_token and has_app:
            logger.warning(
                "Both GitHub token and GitHub App credentials are configured. GitHub App will take precedence."
            )

        return self

    def should_sync_category(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
        """Check if a category should be synced based on the whitelist.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The model reference category.

        Returns:
            True if the category should be synced, False otherwise.
        """
        if self.categories_to_sync is None:
            return True

        return str(category) in self.categories_to_sync


github_sync_settings = HordeGitHubSyncSettings()
"""Global instance of GitHub sync settings."""
