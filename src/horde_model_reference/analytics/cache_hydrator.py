"""Background cache hydration for deletion risk and statistics caches.

Proactively refreshes caches on a timer to ensure clients always receive
fast cached responses instead of waiting for slow Horde API fetches.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Literal

from loguru import logger

from horde_model_reference import ModelReferenceManager, horde_model_reference_settings
from horde_model_reference.analytics.deletion_risk_analysis import CategoryDeletionRiskResponse
from horde_model_reference.analytics.statistics import CategoryStatistics
from horde_model_reference.integrations.horde_api_integration import HordeAPIIntegration
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class CacheHydrator:
    """Background service that proactively refreshes deletion risk and statistics caches.

    Runs on a configurable interval to ensure caches remain warm. When hydration
    is enabled, clients always receive fast cached responses while fresh data
    is computed in the background.

    This implements a "stale-while-revalidate" pattern:
    - Clients receive cached data immediately (even if stale)
    - Background hydration refreshes caches before TTL expiry
    - Stale data is served during hydration to avoid blocking requests

    Examples:
        ```python
        # Start hydration on service startup
        hydrator = CacheHydrator()
        await hydrator.start()

        # Stop on service shutdown
        await hydrator.stop()
        ```

    """

    _instance: CacheHydrator | None = None
    _task: asyncio.Task[None] | None
    _running: bool = False
    _shutdown_event: asyncio.Event

    def __new__(cls) -> CacheHydrator:
        """Singleton pattern for cache hydrator."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._task = None
            cls._instance._running = False
            cls._instance._shutdown_event = asyncio.Event()
        return cls._instance

    @property
    def is_running(self) -> bool:
        """Check if hydration is currently running."""
        return self._running

    @classmethod
    def get_instance(cls) -> CacheHydrator:
        """Get the singleton CacheHydrator instance."""
        return cls()

    async def start(self) -> None:
        """Start the background hydration task.

        Does nothing if hydration is disabled in settings or already running.
        """
        if not horde_model_reference_settings.cache_hydration_enabled:
            logger.info("Cache hydration is disabled in settings")
            return

        if self._running:
            logger.warning("Cache hydration is already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._hydration_loop())
        logger.info(
            f"Cache hydration started with interval={horde_model_reference_settings.cache_hydration_interval_seconds}s"
        )

    async def stop(self) -> None:
        """Stop the background hydration task gracefully."""
        if not self._running:
            return

        logger.info("Stopping cache hydration...")
        self._running = False
        self._shutdown_event.set()

        if self._task:
            try:
                # Give the task a moment to notice the shutdown event
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                logger.warning("Cache hydration task did not stop gracefully, cancelling...")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

        logger.info("Cache hydration stopped")

    async def _hydration_loop(self) -> None:
        """Run the main hydration loop until stopped."""
        # Initial delay to let service fully start
        startup_delay = horde_model_reference_settings.cache_hydration_startup_delay_seconds
        logger.debug(f"Cache hydration waiting {startup_delay}s for startup...")

        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=startup_delay)
            # If we get here, shutdown was requested during startup delay
            return
        except TimeoutError:
            # Normal case - startup delay completed
            pass

        interval = horde_model_reference_settings.cache_hydration_interval_seconds

        while self._running:
            try:
                await self._hydrate_all_caches()
            except Exception as e:
                logger.exception(f"Error during cache hydration: {e}")

            # Wait for interval or shutdown
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                # Shutdown requested
                break
            except TimeoutError:
                # Normal case - interval completed, continue loop
                continue

    async def _hydrate_all_caches(self) -> None:
        """Hydrate all deletion risk and statistics caches for supported categories."""
        logger.debug("Starting cache hydration cycle...")

        # Categories that support deletion risk/statistics
        supported_categories = [
            MODEL_REFERENCE_CATEGORY.image_generation,
            MODEL_REFERENCE_CATEGORY.text_generation,
        ]

        base_variants = (
            (False, False),  # grouped=False, include_backend_variations=False
            (True, False),
        )

        for category in supported_categories:
            try:
                for grouped, include_backend_variations in base_variants:
                    if not self._running:
                        break

                    await self._hydrate_deletion_risk_cache(
                        category,
                        grouped=grouped,
                        include_backend_variations=include_backend_variations,
                    )

                if not self._running:
                    break

                if category == MODEL_REFERENCE_CATEGORY.text_generation and self._running:
                    await self._hydrate_deletion_risk_cache(
                        category,
                        grouped=False,
                        include_backend_variations=True,
                    )

                # TODO: Hydrate statistics cache when implemented
                # await self._hydrate_statistics_cache(category)

            except Exception as e:
                logger.exception(f"Error hydrating cache for {category}: {e}")

            if not self._running:
                break

        logger.debug("Cache hydration cycle completed")

    async def _hydrate_deletion_risk_cache(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        grouped: bool,
        include_backend_variations: bool,
    ) -> None:
        """Hydrate deletion risk cache for a specific category and configuration.

        Args:
            category: The model reference category.
            grouped: Whether to use grouped text model view.
            include_backend_variations: Whether to include backend variations.

        """
        from horde_model_reference.analytics.deletion_risk_cache import DeletionRiskCache

        cache = DeletionRiskCache()

        logger.debug(
            f"Hydrating deletion risk cache: {category.value}, grouped={grouped}, "
            f"backend_variations={include_backend_variations}"
        )

        try:
            # Compute fresh deletion risk data
            risk_response = await self._compute_deletion_risk_response(
                category, grouped=grouped, include_backend_variations=include_backend_variations
            )

            if risk_response:
                # Store in cache (this updates both Redis and in-memory)
                cache.set(
                    category,
                    risk_response,
                    grouped=grouped,
                    include_backend_variations=include_backend_variations,
                )
                logger.info(
                    f"Hydrated deletion risk cache: {category.value} "
                    f"(grouped={grouped}, variations={include_backend_variations}, "
                    f"models={risk_response.total_count})"
                )
        except Exception as e:
            logger.warning(f"Failed to hydrate deletion risk cache for {category}: {e}")

    async def _compute_deletion_risk_response(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        grouped: bool,
        include_backend_variations: bool,
    ) -> CategoryDeletionRiskResponse | None:
        """Compute fresh deletion risk response data.

        This mirrors the logic in the deletion risk endpoint but is designed for
        background execution without HTTP context.

        Args:
            category: The model reference category.
            grouped: Whether to use grouped text model view.
            include_backend_variations: Whether to include backend variations.

        Returns:
            CategoryDeletionRiskResponse if successful, None on error.

        """
        from horde_model_reference.analytics.deletion_risk_analysis import ModelDeletionRiskInfoFactory
        from horde_model_reference.analytics.text_model_grouping import apply_text_model_grouping_to_risk_response
        from horde_model_reference.integrations.data_merger import merge_category_with_horde_data

        manager = ModelReferenceManager()
        horde_api = HordeAPIIntegration()

        # Determine effective backend variations flag
        is_text_category = category == MODEL_REFERENCE_CATEGORY.text_generation
        effective_include_backend_variations = include_backend_variations and is_text_category and not grouped

        # Get model names and records
        model_names = manager.get_model_names(category)
        if not model_names:
            logger.warning(f"No models found for category {category}")
            return None

        model_records = manager.get_model_reference(category)

        # Determine model type for Horde API
        model_type: Literal["image", "text"] = (
            "image" if category == MODEL_REFERENCE_CATEGORY.image_generation else "text"
        )

        # Fetch Horde API data (force refresh to get latest)
        try:
            status_data = await horde_api.get_model_status_indexed(model_type, force_refresh=True)
            stats_data = await horde_api.get_model_stats_indexed(model_type, force_refresh=True)
        except Exception as e:
            logger.warning(f"Cache hydration skipped for {category}: Horde API unavailable ({e})")
            return None

        # Merge with model reference data
        model_statistics = merge_category_with_horde_data(
            model_names=model_names,
            horde_status=status_data,
            horde_stats=stats_data,
            workers=None,
            include_backend_variations=effective_include_backend_variations,
        )

        # Calculate total category usage
        category_total_month_usage = sum(
            stats.usage_stats.month for stats in model_statistics.values() if stats.usage_stats
        )

        # Create deletion risk response
        factory = ModelDeletionRiskInfoFactory.create_default()
        risk_response = factory.create_deletion_risk_response(
            model_records,
            model_statistics,
            category_total_month_usage,
            category,
            include_backend_variations=effective_include_backend_variations,
        )

        # Apply text model grouping if requested
        if grouped and is_text_category:
            risk_response = apply_text_model_grouping_to_risk_response(risk_response)

        return risk_response

    async def _hydrate_statistics_cache(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        grouped: bool = False,
    ) -> None:
        """Hydrate statistics cache for a specific category.

        Args:
            category: The model reference category.
            grouped: Whether to use grouped text model view.

        """
        from horde_model_reference.analytics.statistics_cache import StatisticsCache

        cache = StatisticsCache()

        logger.debug(f"Hydrating statistics cache: {category.value}, grouped={grouped}")

        try:
            statistics = await self._compute_statistics(category, grouped=grouped)

            if statistics:
                cache.set(category, statistics, grouped=grouped)
                logger.info(f"Hydrated statistics cache: {category.value} (grouped={grouped})")
        except Exception as e:
            logger.warning(f"Failed to hydrate statistics cache for {category}: {e}")

    async def _compute_statistics(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        grouped: bool = False,
    ) -> CategoryStatistics | None:
        """Compute fresh statistics data.

        Args:
            category: The model reference category.
            grouped: Whether to use grouped text model view.

        Returns:
            CategoryStatistics if successful, None on error.

        """
        # TODO: Implement statistics computation when statistics endpoint logic is refactored
        # This would mirror the statistics endpoint logic
        logger.debug(f"Statistics hydration not yet implemented for {category}")
        return None


# Module-level singleton accessor
def get_cache_hydrator() -> CacheHydrator:
    """Get the singleton CacheHydrator instance.

    Returns:
        CacheHydrator singleton.

    """
    return CacheHydrator()
