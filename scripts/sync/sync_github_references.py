#!/usr/bin/env python3
r"""CLI script for syncing model references from PRIMARY to GitHub legacy repositories.

This script:
1. Fetches current state from PRIMARY v1 API
2. Fetches current state from GitHub legacy repos
3. Compares and detects drift
4. Creates PRs to sync changes back to GitHub

The script supports two modes:
- **One-shot mode** (default): Runs once and exits
- **Watch mode** (--watch): Continuously monitors metadata changes and syncs automatically

Usage:
    python scripts/sync_github_references.py [OPTIONS]

Environment variables:
    HORDE_GITHUB_SYNC_PRIMARY_API_URL - PRIMARY API URL (required)
    HORDE_GITHUB_SYNC_GITHUB_TOKEN - GitHub token (required for PR creation)
    HORDE_GITHUB_SYNC_CATEGORIES_TO_SYNC - Comma-separated list of categories
    HORDE_GITHUB_SYNC_DRY_RUN - Set to 'true' for dry run mode
    HORDE_GITHUB_SYNC_VERBOSE_LOGGING - Set to 'true' for verbose logging
    HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS - Polling interval for watch mode (default: 60)
    HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC - Run sync on startup in watch mode (default: false)

    For testing with forks, also set:
    HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_OWNER - Owner of image repo fork (e.g., 'tazlin')
    HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_NAME - Name of image repo (if different)
    HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_OWNER - Owner of text repo fork (e.g., 'tazlin')
    HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_NAME - Name of text repo (if different)

Examples:
    # Sync all categories (one-shot mode)
    python scripts/sync_github_references.py

    # Dry run to see what would change
    HORDE_GITHUB_SYNC_DRY_RUN=true python scripts/sync_github_references.py

    # Sync specific categories
    HORDE_GITHUB_SYNC_CATEGORIES_TO_SYNC=image_generation,text_generation \\
        python scripts/sync_github_references.py

    # Verbose mode
    HORDE_GITHUB_SYNC_VERBOSE_LOGGING=true python scripts/sync_github_references.py

    # Watch mode - continuously monitor for changes
    python scripts/sync_github_references.py --watch

    # Watch mode with custom interval (5 minutes)
    python scripts/sync_github_references.py --watch --watch-interval 300

    # Watch mode with startup sync
    python scripts/sync_github_references.py --watch --watch-startup-sync

    # Test with forks (PowerShell example)
    $env:HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_OWNER="tazlin"
    $env:HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_OWNER="tazlin"
    python scripts/sync_github_references.py --primary-url http://localhost:19800 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import httpx
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
    level="INFO",
)

from horde_model_reference import MODEL_REFERENCE_CATEGORY, horde_model_reference_settings  # noqa: E402
from horde_model_reference.backends.github_backend import GitHubBackend  # noqa: E402
from horde_model_reference.sync import (  # noqa: E402
    GitHubSyncClient,
    ModelReferenceComparator,
    ModelReferenceDiff,
    WatchModeManager,
    github_sync_settings,
)

if github_sync_settings.verbose_logging:
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
        level="DEBUG",
    )


class GithubSynchronizer:
    """Helper class for syncing model references from PRIMARY to GitHub."""

    backend: GitHubBackend

    def __init__(self) -> None:
        """Initialize the synchronizer with a GitHub backend."""
        self.backend = GitHubBackend()

    def fetch_primary_data(
        self,
        *,
        api_url: str,
        category: MODEL_REFERENCE_CATEGORY,
        timeout: int = 30,
    ) -> dict[str, dict[str, Any]]:
        """Fetch model reference data from PRIMARY v1 API.

        Args:
            api_url: Base URL of PRIMARY API (e.g., https://stablehorde.net/api).
            category (MODEL_REFERENCE_CATEGORY): The category to fetch.
            timeout: Request timeout in seconds.

        Returns:
            Dictionary of model records in legacy format.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        endpoint = f"{api_url.rstrip('/')}/model_references/v1/{category}"
        logger.debug(f"Fetching PRIMARY data from {endpoint}")

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                data: dict[str, dict[str, Any]] = response.json()

            logger.debug(f"Fetched {len(data)} models for {category} from PRIMARY")
            return data

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch PRIMARY data for {category}: {e}")
            raise

    def fetch_github_data(
        self,
        *,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> dict[str, dict[str, Any]]:
        """Fetch model reference data from GitHub legacy repos.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category to fetch.

        Returns:
            Dictionary of model records in legacy format.

        Raises:
            Exception: If the fetch fails.
        """
        logger.debug(f"Fetching GitHub data for {category}")

        try:
            data: dict[str, Any] | None = self.backend.get_legacy_json(category)

            if data is None:
                logger.warning(f"No data found for category {category} in GitHub")
                raise ValueError(f"No data for category {category}")

            logger.debug(f"Fetched {len(data)} models for {category} from GitHub")
            return data

        except Exception as e:
            logger.error(f"Failed to fetch GitHub data for {category}: {e}")
            raise


def main() -> int:
    """Enter the GitHub sync service.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Sync model references from PRIMARY to GitHub legacy repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--primary-url",
        type=str,
        default=None,
        help="PRIMARY API base URL (default: from env HORDE_GITHUB_SYNC_PRIMARY_API_URL)",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories to sync (default: all)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview changes without creating PRs",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Create PR even if changes are below threshold",
    )

    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Base directory for persistent repo clones (default: from env HORDE_GITHUB_SYNC_TARGET_CLONE_DIR)",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Enable watch mode to continuously monitor for metadata changes",
    )

    parser.add_argument(
        "--watch-interval",
        type=int,
        default=None,
        help="Polling interval in seconds for watch mode (default: 60)",
    )

    parser.add_argument(
        "--watch-startup-sync",
        action="store_true",
        default=False,
        help="Run a sync immediately when watch mode starts (before entering watch loop)",
    )

    args = parser.parse_args()

    if args.primary_url:
        github_sync_settings.primary_api_url = args.primary_url

    if args.categories:
        github_sync_settings.categories_to_sync = args.categories.split(",")

    if args.dry_run:
        github_sync_settings.dry_run = True

    if args.force:
        github_sync_settings.min_changes_threshold = 0

    if args.target_dir:
        github_sync_settings.target_clone_dir = args.target_dir

    if args.watch:
        github_sync_settings.watch_mode = True

    if args.watch_interval is not None:
        github_sync_settings.watch_interval_seconds = args.watch_interval

    if args.watch_startup_sync:
        github_sync_settings.watch_enable_startup_sync = True

    primary_api_url = github_sync_settings.primary_api_url
    if not primary_api_url:
        logger.error(
            "PRIMARY API URL is required. "
            "Set via --primary-url or HORDE_GITHUB_SYNC_PRIMARY_API_URL environment variable."
        )
        return 1

    # Watch mode
    if github_sync_settings.watch_mode:
        watch_manager = WatchModeManager(
            api_url=primary_api_url,
            sync_callback=run_sync_once,
            interval_seconds=github_sync_settings.watch_interval_seconds,
            enable_startup_sync=github_sync_settings.watch_enable_startup_sync,
        )
        return watch_manager.run()

    # One-shot mode
    return run_sync_once()


def run_sync_once() -> int:
    """Run a single sync operation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    primary_api_url = github_sync_settings.primary_api_url
    if not primary_api_url:
        logger.error("PRIMARY API URL not configured")
        return 1

    logger.info("=" * 80)
    logger.info("GitHub Model Reference Sync Service")
    logger.info("=" * 80)
    logger.info(f"Image repo: {horde_model_reference_settings.image_github_repo.repo_owner_and_name}")
    logger.info(f"Text repo: {horde_model_reference_settings.text_github_repo.repo_owner_and_name}")
    logger.info("=" * 80)

    comparator = ModelReferenceComparator()
    github_synchronizer = GithubSynchronizer()

    logger.info("Phase 1: Scanning all categories for changes...")
    logger.info("-" * 80)

    category_changes: dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]] = {}

    for category in MODEL_REFERENCE_CATEGORY:
        if not github_sync_settings.should_sync_category(category):
            logger.debug(f"Skipping {category} (not in categories_to_sync filter)")
            continue

        try:
            primary_data = github_synchronizer.fetch_primary_data(
                api_url=primary_api_url,
                category=category,
                timeout=30,
            )

            github_data = github_synchronizer.fetch_github_data(category=category)

            diff = comparator.compare_categories(
                category=category,
                primary_data=primary_data,
                github_data=github_data,
            )

            if diff.has_changes():
                category_changes[category] = (diff, primary_data)
                logger.info(f"✓ {category}: {diff.total_changes()} changes detected")
            else:
                logger.info(f"✓ {category}: No changes needed")

        except Exception as e:
            logger.error(f"✗ {category}: Failed to scan - {e}")

    if not category_changes:
        logger.success("\n✓ No changes detected across any categories")
        return 0

    logger.info("-" * 80)
    logger.info(f"Phase 2: Grouping {len(category_changes)} categories by repository...")
    logger.info("-" * 80)

    repo_groups: dict[str, dict[MODEL_REFERENCE_CATEGORY, tuple[ModelReferenceDiff, dict[str, dict[str, Any]]]]] = {}

    for category, (diff, primary_data) in category_changes.items():
        try:
            repo_owner_and_name = horde_model_reference_settings.get_repo_by_category(category).repo_owner_and_name
            if repo_owner_and_name not in repo_groups:
                repo_groups[repo_owner_and_name] = {}
            repo_groups[repo_owner_and_name][category] = (diff, primary_data)
        except ValueError as e:
            logger.warning(f"Skipping {category}: {e}")

    for repo_owner_and_name, categories in repo_groups.items():
        total_changes = sum(diff.total_changes() for diff, _ in categories.values())
        category_list = ", ".join(str(cat) for cat in categories)
        logger.info(f"Repository: {repo_owner_and_name}")
        logger.info(f"  Categories: {category_list}")
        logger.info(f"  Total changes: {total_changes}")

    logger.info("-" * 80)
    logger.info("Phase 3: Creating pull requests...")
    logger.info("-" * 80)

    results: dict[str, tuple[bool, str | None]] = {}

    with GitHubSyncClient() as github_client:
        for repo_owner_and_name, categories_data in repo_groups.items():
            try:
                if len(categories_data) == 1:
                    category = next(iter(categories_data))
                    diff, primary_data = categories_data[category]

                    logger.info(f"Creating single-category PR for {category} in {repo_owner_and_name}")

                    pr_url = github_client.sync_category_to_github(
                        category=category,
                        diff=diff,
                        primary_data=primary_data,
                    )

                    results[str(category)] = (True, pr_url)

                else:
                    category_names = ", ".join(str(cat) for cat in categories_data)
                    logger.info(f"Creating multi-category PR for {category_names} in {repo_owner_and_name}")

                    pr_url = github_client.sync_multiple_categories_to_github(
                        repo_name=repo_owner_and_name,
                        categories_data=categories_data,
                    )

                    for category in categories_data:
                        results[str(category)] = (True, pr_url)

            except Exception as e:
                logger.error(f"Failed to create PR for {repo_owner_and_name}: {e}")
                for category in categories_data:
                    results[str(category)] = (False, None)

    logger.info("=" * 80)
    logger.info("Sync Summary")
    logger.info("=" * 80)

    successful = sum(1 for success, _ in results.values() if success)
    failed = len(results) - successful
    prs_created = len({pr_url for _, pr_url in results.values() if pr_url})

    logger.info(f"Categories processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"PRs created: {prs_created}")

    if prs_created > 0:
        logger.info("\nCreated PRs:")
        seen_prs = set()
        for _, pr_url in results.values():
            if pr_url and pr_url not in seen_prs:
                logger.info(f"  {pr_url}")
                seen_prs.add(pr_url)

    if failed > 0:
        logger.error(f"\n{failed} category(ies) failed to sync")
        return 1

    logger.success("\n✓ All categories synced successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
