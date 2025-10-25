"""Watch mode manager for monitoring metadata changes and triggering syncs."""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from typing import Any

import httpx
from loguru import logger

from horde_model_reference.sync.config import github_sync_settings


class WatchModeManager:
    """Manages watch mode for continuously monitoring metadata changes.

    Polls the PRIMARY v1 metadata /last_updated endpoint at regular intervals
    and triggers sync operations when changes are detected.
    """

    def __init__(
        self,
        *,
        api_url: str,
        sync_callback: Callable[[], int],
        interval_seconds: int | None = None,
        initial_delay_seconds: int | None = None,
        enable_startup_sync: bool | None = None,
    ) -> None:
        """Initialize the watch mode manager.

        Args:
            api_url: Base URL of PRIMARY API (e.g., https://stablehorde.net/api).
            sync_callback: Function to call when changes are detected. Should return exit code (0 for success).
            interval_seconds: Polling interval in seconds (default: from settings).
            initial_delay_seconds: Initial delay before starting watch loop (default: from settings).
            enable_startup_sync: Whether to run sync immediately on startup (default: from settings).
        """
        self.api_url = api_url.rstrip("/")
        self.sync_callback = sync_callback
        self.interval_seconds = interval_seconds or github_sync_settings.watch_interval_seconds
        self.initial_delay_seconds = initial_delay_seconds or github_sync_settings.watch_initial_delay_seconds
        self.enable_startup_sync = (
            enable_startup_sync
            if enable_startup_sync is not None
            else github_sync_settings.watch_enable_startup_sync
        )

        self.last_known_timestamp: int | None = None
        self.running = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:  # noqa: ANN401
        """Handle shutdown signals gracefully.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"\n{signal_name} received. Shutting down watch mode gracefully...")
        self.running = False

    def fetch_last_updated_timestamp(self) -> int | None:
        """Fetch the last_updated timestamp from PRIMARY metadata endpoint.

        Returns:
            Unix timestamp of last update, or None if fetch fails or no metadata exists.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        endpoint = f"{self.api_url}/model_references/v1/metadata/last_updated"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(endpoint)
                response.raise_for_status()
                data = response.json()

            timestamp: int | None = data.get("last_updated")
            if timestamp is None:
                logger.debug("Metadata endpoint returned null timestamp (no metadata exists yet)")
            else:
                logger.debug(f"Fetched last_updated timestamp: {timestamp}")

            return timestamp

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching metadata: {e.response.status_code} - {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Network error fetching metadata: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching metadata: {e}")
            raise

    def check_for_changes(self) -> bool:
        """Check if metadata has changed since last check.

        Returns:
            True if changes were detected (or first run), False otherwise.
        """
        try:
            current_timestamp = self.fetch_last_updated_timestamp()

            # First run - initialize but don't trigger sync unless enable_startup_sync is True
            if self.last_known_timestamp is None:
                self.last_known_timestamp = current_timestamp
                logger.info(f"Initialized last known timestamp: {current_timestamp}")

                # Return True only if startup sync is enabled
                return self.enable_startup_sync

            # No metadata exists yet
            if current_timestamp is None:
                logger.debug("No metadata available yet")
                return False

            # Check for changes
            if current_timestamp > self.last_known_timestamp:
                logger.info(
                    f"Changes detected! Timestamp changed from {self.last_known_timestamp} to {current_timestamp}"
                )
                self.last_known_timestamp = current_timestamp
                return True

            logger.debug(f"No changes detected (timestamp: {current_timestamp})")
            return False

        except Exception as e:
            self.consecutive_errors += 1
            logger.error(
                f"Error checking for changes (error {self.consecutive_errors}/{self.max_consecutive_errors}): {e}"
            )

            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.critical(
                    f"Exceeded maximum consecutive errors ({self.max_consecutive_errors}). "
                    "Check network connectivity and PRIMARY API availability."
                )
                return False

            return False

    def run(self) -> int:
        """Run the watch mode loop.

        Continuously monitors metadata changes and triggers sync operations.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        self.running = True

        logger.info("=" * 80)
        logger.info("GitHub Model Reference Sync Service - WATCH MODE")
        logger.info("=" * 80)
        logger.info(f"Monitoring PRIMARY API: {self.api_url}")
        logger.info(f"Polling interval: {self.interval_seconds} seconds")
        logger.info(f"Initial delay: {self.initial_delay_seconds} seconds")
        logger.info(f"Startup sync: {'enabled' if self.enable_startup_sync else 'disabled'}")
        logger.info("=" * 80)

        # Initial delay if configured
        if self.initial_delay_seconds > 0:
            logger.info(f"Waiting {self.initial_delay_seconds} seconds before starting watch loop...")
            time.sleep(self.initial_delay_seconds)

        logger.info("Starting watch loop. Press Ctrl+C to stop.")
        logger.info("-" * 80)

        watch_iteration = 0

        while self.running:
            watch_iteration += 1
            logger.debug(f"Watch iteration {watch_iteration}")

            try:
                # Check for changes
                has_changes = self.check_for_changes()

                # Reset error counter on successful check
                if self.consecutive_errors > 0:
                    logger.info("Connection restored after errors")
                    self.consecutive_errors = 0

                # Trigger sync if changes detected
                if has_changes:
                    logger.info("Triggering sync operation due to detected changes...")
                    logger.info("-" * 80)

                    try:
                        exit_code = self.sync_callback()

                        if exit_code == 0:
                            logger.success("Sync completed successfully")
                        else:
                            logger.error(f"Sync failed with exit code {exit_code}")

                    except Exception as e:
                        logger.error(f"Sync operation raised an exception: {e}")

                    logger.info("-" * 80)
                    logger.info("Resuming watch loop...")

            except Exception as e:
                logger.error(f"Unexpected error in watch loop: {e}")
                self.consecutive_errors += 1

                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.critical("Too many consecutive errors. Exiting watch mode.")
                    return 1

            # Sleep until next check (if still running)
            if self.running:
                logger.debug(f"Sleeping for {self.interval_seconds} seconds...")
                try:
                    time.sleep(self.interval_seconds)
                except KeyboardInterrupt:
                    logger.info("\nKeyboard interrupt detected during sleep. Shutting down...")
                    self.running = False

        logger.info("=" * 80)
        logger.info("Watch mode stopped")
        logger.info("=" * 80)
        return 0
