"""Pydantic models for AI Horde public API responses.

API Documentation: https://aihorde.net/api/
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel

HordeModelType = Literal["image", "text"]
HordeWorkerType = Literal["image", "text", "interrogation", "alchemy"]  # All worker types from API
HordeModelState = Literal["known", "custom", "all"]


class HordeModelStatus(BaseModel):
    """Model status from Horde API status endpoint."""

    performance: float = Field(description="Performance metric (varies by model type)")
    queued: int = Field(description="Number of queued requests (pixelsteps for image, tokens for text)")
    jobs: int = Field(description="Number of active jobs")
    eta: int = Field(description="Estimated time to completion in seconds")
    type: HordeModelType = Field(description="Model type (image or text)")
    name: str = Field(description="Model name")
    count: int = Field(description="Number of workers serving this model")


class HordeModelStatsResponse(BaseModel):
    """Model statistics from Horde API stats endpoint."""

    day: dict[str, int] = Field(default_factory=dict, description="Statistics for the past day (model_name -> count)")
    month: dict[str, int] = Field(
        default_factory=dict, description="Statistics for the past month (model_name -> count)"
    )
    total: dict[str, int] = Field(default_factory=dict, description="All-time statistics (model_name -> count)")


class HordeTotalStatsTimePeriod(BaseModel):
    """Statistics for a specific time period."""

    images: int | None = Field(default=None, description="Number of images generated (for image type)")
    requests: int | None = Field(default=None, description="Number of requests processed")
    tokens: int | None = Field(default=None, description="Number of tokens processed (for text type)")
    ps: int | None = Field(default=None, description="Pixelsteps (for image type)")


class HordeTotalStatsResponse(BaseModel):
    """Total statistics across all models from Horde API."""

    minute: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past minute"
    )
    hour: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past hour"
    )
    day: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past day"
    )
    month: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past month"
    )
    total: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="All-time statistics"
    )


class HordeModelUsageStats(BaseModel):
    """Usage statistics for a specific model."""

    minute: int | None = Field(default=None, description="Usage count for the past minute")
    hour: int | None = Field(default=None, description="Usage count for the past hour")
    day: int = Field(description="Usage count for the past day")
    month: int = Field(description="Usage count for the past month")
    total: int = Field(description="All-time usage count")


class BackendVariation(BaseModel):
    """Per-backend statistics for a text generation model variant.

    This tracks statistics for a specific backend (e.g., 'aphrodite' or 'koboldcpp')
    serving a particular model. Used to show backend-specific details in the UI
    while still providing aggregated totals at the model level.
    """

    backend: str = Field(description="Backend name (e.g., 'aphrodite', 'koboldcpp', or 'canonical' for non-prefixed)")
    variant_name: str = Field(description="Full model name as reported by Horde API (may include backend prefix)")
    worker_count: int = Field(description="Number of workers serving this backend variant")
    performance: float | None = Field(default=None, description="Performance metric for this variant")
    queued: int | None = Field(default=None, description="Number of queued requests for this variant")
    queued_jobs: int | None = Field(default=None, description="Number of active jobs for this variant")
    eta: int | None = Field(default=None, description="Estimated time to completion for this variant")
    usage_day: int = Field(default=0, description="Usage count for the past day")
    usage_month: int = Field(default=0, description="Usage count for the past month")
    usage_total: int = Field(default=0, description="All-time usage count")


class HordeWorkerTeam(BaseModel):
    """Worker team information."""

    name: str | None = Field(default=None, description="Team name")
    id: str | None = Field(default=None, description="Team ID (UUID)")


class HordeKudosDetails(BaseModel):
    """Kudos breakdown by source."""

    generated: float | None = Field(default=None, description="Kudos generated from work completed")
    uptime: float | None = Field(default=None, description="Kudos from uptime")


class IndexedHordeModelStatus(RootModel[dict[str, HordeModelStatus]]):
    """Indexed model status for O(1) lookups by model name.

    This wraps the status list and provides case-insensitive dictionary access.
    Time complexity: O(1) for lookups instead of O(n) for list iteration.

    Usage:
        indexed = IndexedHordeModelStatus([status1, status2, ...])
        status = indexed.get("model_name")  # Case-insensitive lookup
        all_statuses = indexed.get_all()    # Get all as list
    """

    root: dict[str, HordeModelStatus]

    def __init__(self, status_list: list[HordeModelStatus]) -> None:
        """Build indexed lookup from status list.

        Args:
            status_list: List of HordeModelStatus from API
        """
        # Build case-insensitive lookup dictionary
        status_dict = {s.name.lower(): s for s in status_list}
        super().__init__(root=status_dict)

    def get(self, model_name: str) -> HordeModelStatus | None:
        """Get status for a model by name (case-insensitive).

        Time Complexity: O(1)

        Args:
            model_name: Model name to look up

        Returns:
            HordeModelStatus if found, None otherwise
        """
        return self.root.get(model_name.lower())

    def get_all(self) -> list[HordeModelStatus]:
        """Get all status entries as a list.

        Returns:
            List of all HordeModelStatus objects
        """
        return list(self.root.values())

    def get_aggregated_status(self, canonical_name: str) -> HordeModelStatus | None:
        """Get aggregated status across all backend variants of a model.

        This method aggregates status from all possible backend-prefixed variants,
        taking the maximum worker count (count field) and the first non-None value
        for other fields.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Aggregated HordeModelStatus or None if no variants have status.
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)

        statuses: list[HordeModelStatus] = []
        for variant in variants:
            status = self.get(variant)
            if status is not None:
                statuses.append(status)

        if not statuses:
            return None

        return max(statuses, key=lambda s: s.count)

    def get_status_with_variations(
        self, canonical_name: str
    ) -> tuple[HordeModelStatus | None, dict[str, HordeModelStatus]]:
        """Get aggregated status and individual backend variations.

        This method returns both the aggregated status (same as get_aggregated_status)
        and a dictionary of individual backend statuses keyed by backend name.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (aggregated_status, variations_dict) where:
            - aggregated_status: Combined status or None if no variants found
            - variations_dict: Dict of backend_name -> HordeModelStatus
              Keys are 'canonical', 'aphrodite', 'koboldcpp' depending on what's found
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)
        variations: dict[str, HordeModelStatus] = {}

        # Look up each variant and store by backend name
        for variant in variants:
            status = self.get(variant)
            if status is not None:
                # Determine backend name from variant
                if variant == canonical_name:
                    backend_name = "canonical"
                elif variant.startswith("aphrodite/"):
                    backend_name = "aphrodite"
                elif variant.startswith("koboldcpp/"):
                    backend_name = "koboldcpp"
                else:
                    backend_name = "unknown"

                variations[backend_name] = status

        if not variations:
            return None, {}

        # Aggregate: take max by worker count
        aggregated = max(variations.values(), key=lambda s: s.count)
        return aggregated, variations


class _StatsLookup(BaseModel):
    """Internal structure for indexed stats lookups."""

    day: dict[str, int] = Field(default_factory=dict)
    month: dict[str, int] = Field(default_factory=dict)
    total: dict[str, int] = Field(default_factory=dict)


class IndexedHordeModelStats(RootModel[_StatsLookup]):
    """Indexed model stats for O(1) lookups by model name.

    This wraps the stats response and provides case-insensitive dictionary access.
    Time complexity: O(1) for lookups instead of O(n) for dict iteration.

    Usage:
        indexed = IndexedHordeModelStats(stats_response)
        day_count = indexed.get_day("model_name")      # Case-insensitive
        has_data = indexed.has_stats("model_name")     # Check existence
    """

    root: _StatsLookup

    def __init__(self, stats_response: HordeModelStatsResponse) -> None:
        """Build indexed lookups from stats response.

        Args:
            stats_response: HordeModelStatsResponse from API
        """
        # Build case-insensitive lookup dictionaries for each time period
        lookups = _StatsLookup(
            day={k.lower(): v for k, v in stats_response.day.items()},
            month={k.lower(): v for k, v in stats_response.month.items()},
            total={k.lower(): v for k, v in stats_response.total.items()},
        )
        super().__init__(root=lookups)

    def get_day(self, model_name: str) -> int | None:
        """Get day count for a model (case-insensitive). O(1)."""
        return self.root.day.get(model_name.lower())

    def get_month(self, model_name: str) -> int | None:
        """Get month count for a model (case-insensitive). O(1)."""
        return self.root.month.get(model_name.lower())

    def get_total(self, model_name: str) -> int | None:
        """Get total count for a model (case-insensitive). O(1)."""
        return self.root.total.get(model_name.lower())

    def has_stats(self, model_name: str) -> bool:
        """Check if model has any stats (case-insensitive). O(1)."""
        name_lower = model_name.lower()
        return name_lower in self.root.day or name_lower in self.root.month or name_lower in self.root.total

    def get_aggregated_stats(self, canonical_name: str) -> tuple[int, int, int]:
        """Get aggregated stats across all backend variants of a model.

        This method aggregates stats from all possible backend-prefixed variants:
        - Canonical name (e.g., "ReadyArt/Broken-Tutu-24B")
        - Aphrodite variant (e.g., "aphrodite/ReadyArt/Broken-Tutu-24B")
        - KoboldCPP variant (e.g., "koboldcpp/Broken-Tutu-24B")

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (day_total, month_total, total_total) aggregated across all variants.

        Example:
            >>> indexed = IndexedHordeModelStats(stats_response)
            >>> day, month, total = indexed.get_aggregated_stats("ReadyArt/Broken-Tutu-24B")
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)

        day_total = 0
        month_total = 0
        total_total = 0

        for variant in variants:
            day_total += self.get_day(variant) or 0
            month_total += self.get_month(variant) or 0
            total_total += self.get_total(variant) or 0

        return (day_total, month_total, total_total)

    def get_stats_with_variations(
        self, canonical_name: str
    ) -> tuple[tuple[int, int, int], dict[str, tuple[int, int, int]]]:
        """Get aggregated stats and individual backend variations.

        This method returns both the aggregated stats (same as get_aggregated_stats)
        and a dictionary of individual backend stats keyed by backend name.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (aggregated_stats, variations_dict) where:
            - aggregated_stats: (day_total, month_total, total_total) aggregated
            - variations_dict: Dict of backend_name -> (day, month, total)
              Keys are 'canonical', 'aphrodite', 'koboldcpp' depending on what's found
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)
        variations: dict[str, tuple[int, int, int]] = {}

        day_total = 0
        month_total = 0
        total_total = 0

        # Look up each variant and store by backend name
        for variant in variants:
            day = self.get_day(variant) or 0
            month = self.get_month(variant) or 0
            total = self.get_total(variant) or 0

            # Only store if there's actual data
            if day > 0 or month > 0 or total > 0:
                # Determine backend name from variant
                if variant == canonical_name:
                    backend_name = "canonical"
                elif variant.startswith("aphrodite/"):
                    backend_name = "aphrodite"
                elif variant.startswith("koboldcpp/"):
                    backend_name = "koboldcpp"
                else:
                    backend_name = "unknown"

                variations[backend_name] = (day, month, total)

            day_total += day
            month_total += month
            total_total += total

        return (day_total, month_total, total_total), variations


class HordeWorker(BaseModel):
    """Worker information from Horde API.

    This is a simplified model that captures the common fields needed for merging.
    The API returns additional type-specific fields that aren't needed for basic integration.
    """

    name: str = Field(description="Worker name")
    id: str = Field(description="Worker ID (UUID)")
    type: HordeWorkerType = Field(description="Worker type (image, text, interrogation, or alchemy)")
    performance: str = Field(description="Performance metric as a string")
    requests_fulfilled: int = Field(description="Number of requests fulfilled by this worker")
    kudos_rewards: float = Field(description="Total kudos rewards earned")
    kudos_details: HordeKudosDetails = Field(description="Kudos breakdown by source")
    threads: int = Field(description="Number of worker threads")
    uptime: int = Field(description="Total uptime in seconds")
    uncompleted_jobs: int = Field(description="Number of jobs not yet completed")
    maintenance_mode: bool = Field(description="Whether worker is in maintenance mode")
    nsfw: bool = Field(description="Whether worker serves NSFW content")
    trusted: bool = Field(description="Whether worker is trusted")
    flagged: bool = Field(description="Whether worker is flagged")
    online: bool = Field(description="Whether worker is currently online")
    models: list[str] = Field(default_factory=list, description="List of model names this worker serves")
    team: HordeWorkerTeam = Field(description="Team information (if worker belongs to a team)")
    bridge_agent: str = Field(description="Bridge agent information (software/version)")

    # Type-specific optional fields (not all workers have these)
    max_pixels: int | None = Field(default=None, description="Maximum pixels (image workers)")
    megapixelsteps_generated: float | None = Field(default=None, description="Total megapixelsteps (image workers)")
    img2img: bool | None = Field(default=None, description="Supports img2img (image workers)")
    painting: bool | None = Field(default=None, description="Supports inpainting (image workers)")
    lora: bool | None = Field(default=None, description="Supports LoRA (image workers)")
    controlnet: bool | None = Field(default=None, description="Supports ControlNet (image workers)")
    sdxl_controlnet: bool | None = Field(default=None, description="Supports SDXL ControlNet (image workers)")
    max_length: int | None = Field(default=None, description="Maximum token length (text workers)")
    max_context_length: int | None = Field(default=None, description="Maximum context length (text workers)")
    info: str | None = Field(default=None, description="Additional worker information")

    # Handle post-processing as aliased field (hyphen in API response)
    post_processing: bool | None = Field(default=None, alias="post-processing", description="Supports post-processing")


class IndexedHordeWorkers(RootModel[dict[str, list[HordeWorker]]]):
    """Indexed workers for O(1) lookups by model name.

    This wraps the workers list and provides case-insensitive dictionary access
    where keys are model names and values are lists of workers serving that model.
    Time complexity: O(1) for lookups instead of O(w*m) iteration.

    Usage:
        indexed = IndexedHordeWorkers([worker1, worker2, ...])
        workers = indexed.get("model_name")  # Case-insensitive lookup
        all_workers = indexed.get_all()      # Get all unique workers
    """

    root: dict[str, list[HordeWorker]]

    def __init__(self, workers_list: list[HordeWorker]) -> None:
        """Build indexed lookup from workers list.

        Args:
            workers_list: List of HordeWorker from API
        """
        # Build case-insensitive lookup dictionary by model name
        workers_by_model: dict[str, list[HordeWorker]] = {}
        for worker in workers_list:
            for model_name in worker.models:
                model_name_lower = model_name.lower()
                if model_name_lower not in workers_by_model:
                    workers_by_model[model_name_lower] = []
                workers_by_model[model_name_lower].append(worker)
        super().__init__(root=workers_by_model)

    def get(self, model_name: str) -> list[HordeWorker]:
        """Get workers for a model by name (case-insensitive).

        Time Complexity: O(1)

        Args:
            model_name: Model name to look up

        Returns:
            List of HordeWorker serving this model (empty list if none)
        """
        return self.root.get(model_name.lower(), [])

    def get_all(self) -> list[HordeWorker]:
        """Get all unique workers as a list.

        Returns:
            List of all HordeWorker objects (deduplicated)
        """
        seen_ids = set()
        all_workers = []
        for workers in self.root.values():
            for worker in workers:
                if worker.id not in seen_ids:
                    seen_ids.add(worker.id)
                    all_workers.append(worker)
        return all_workers
