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
