"""Pure functions for merging model reference data with Horde runtime data.

This module provides utilities to combine static model reference data with
dynamic runtime data from the AI Horde API (status, statistics, workers).
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field, computed_field

from horde_model_reference.integrations.horde_api_models import (
    HordeModelStatsResponse,
    HordeModelStatus,
    HordeWorker,
    IndexedHordeModelStats,
    IndexedHordeModelStatus,
    IndexedHordeWorkers,
)


class WorkerSummary(BaseModel):
    """Summary of a worker serving a model."""

    id: str = Field(description="Worker ID (UUID)")
    name: str = Field(description="Worker name")
    performance: str = Field(description="Performance metric as a string")
    online: bool = Field(description="Whether worker is currently online")
    trusted: bool = Field(description="Whether worker is trusted")
    uptime: int = Field(description="Total uptime in seconds")


class UsageStats(BaseModel):
    """Usage statistics for a specific model."""

    day: int = Field(description="Usage count for the past day")
    month: int = Field(description="Usage count for the past month")
    total: int = Field(description="All-time usage count")


class CombinedModelStatistics(BaseModel):
    """Horde usage statistics and data for a model, aggregated from multiple sources."""

    # Horde runtime fields
    @computed_field  # type: ignore[prop-decorator]
    @property
    def worker_count(self) -> int:
        """Number of workers serving this model.

        Returns worker count from detailed worker_summaries if available.
        Returns 0 if worker_summaries is None (detailed worker info not fetched).
        """
        if self.worker_summaries:
            return len(self.worker_summaries)
        return 0

    queued_jobs: int | None = Field(default=None, description="Number of active jobs")
    performance: float | None = Field(default=None, description="Performance metric")
    eta: int | None = Field(default=None, description="Estimated time to completion in seconds")
    queued: int | None = Field(default=None, description="Number of queued requests")
    usage_stats: UsageStats | None = Field(default=None, description="Usage statistics from Horde")
    worker_summaries: dict[str, WorkerSummary] | None = Field(default=None, description="Workers serving this model")
    worker_count_from_status: int | None = Field(
        default=None,
        description="Worker count from HordeModelStatus (used when worker_summaries not available)",
        exclude=True,  # Don't include in serialization by default
    )


def merge_model_with_horde_data(
    model_name: str,
    horde_status: list[HordeModelStatus] | IndexedHordeModelStatus,
    horde_stats: HordeModelStatsResponse | IndexedHordeModelStats,
    workers: list[HordeWorker] | IndexedHordeWorkers | None = None,
) -> CombinedModelStatistics:
    """Merge a single model reference with Horde runtime data.

    **Optimization**: Pass IndexedHordeModelStatus, IndexedHordeModelStats, and
    IndexedHordeWorkers instead of raw lists to skip the indexing overhead.
    This is especially beneficial when merging multiple models sequentially.

    Args:
        model_name: The model name to look up in Horde data.
        horde_status: Model status from Horde API (list or indexed).
        horde_stats: Model statistics from Horde API (response or indexed).
        workers: Optional workers from Horde API (list or indexed).

    Returns:
        CombinedModelStatistics with runtime fields:
            - worker_count: Number of workers serving this model (computed field)
            - queued_jobs: Number of active jobs
            - performance: Performance metric
            - eta: Estimated time to completion in seconds
            - queued: Number of queued requests
            - usage_stats: UsageStats with {day, month, total} usage counts
            - worker_summaries: Dict of worker_id -> WorkerSummary (if workers provided)
    """
    indexed_status = IndexedHordeModelStatus(horde_status) if isinstance(horde_status, list) else horde_status
    indexed_stats = (
        IndexedHordeModelStats(horde_stats) if isinstance(horde_stats, HordeModelStatsResponse) else horde_stats
    )

    indexed_workers: IndexedHordeWorkers | None = None
    if workers is not None:
        indexed_workers = IndexedHordeWorkers(workers) if isinstance(workers, list) else workers

    # Extract status data
    status = indexed_status.get(model_name)
    queued_jobs = status.jobs if status else None
    performance = status.performance if status else None
    eta = status.eta if status else None
    queued = status.queued if status else None
    worker_count_from_status = status.count if status else None

    # Extract usage stats
    usage_stats = None
    if indexed_stats.has_stats(model_name):
        usage_stats = UsageStats(
            day=indexed_stats.get_day(model_name) or 0,
            month=indexed_stats.get_month(model_name) or 0,
            total=indexed_stats.get_total(model_name) or 0,
        )

    # Extract worker summaries
    worker_summaries = None
    if indexed_workers:
        workers_for_model = indexed_workers.get(model_name)
        if workers_for_model:
            worker_summaries = {
                w.id: WorkerSummary(
                    id=w.id,
                    name=w.name,
                    performance=w.performance,
                    online=w.online,
                    trusted=w.trusted,
                    uptime=w.uptime,
                )
                for w in workers_for_model
            }

    return CombinedModelStatistics(
        queued_jobs=queued_jobs,
        performance=performance,
        eta=eta,
        queued=queued,
        usage_stats=usage_stats,
        worker_summaries=worker_summaries,
        worker_count_from_status=worker_count_from_status,
    )


def merge_category_with_horde_data(
    model_names: Iterable[str],
    horde_status: list[HordeModelStatus] | IndexedHordeModelStatus,
    horde_stats: HordeModelStatsResponse | IndexedHordeModelStats,
    workers: list[HordeWorker] | IndexedHordeWorkers | None = None,
) -> dict[str, CombinedModelStatistics]:
    """Merge all models in a category with Horde runtime data.

    **Optimization**: Pass IndexedHordeModelStatus, IndexedHordeModelStats, and
    IndexedHordeWorkers instead of raw lists to skip the O(s+t+w*p) indexing overhead.
    This is especially beneficial when merging multiple categories sequentially.

    Args:
        model_names: Iterable of model names to merge.
        horde_status: Model status from Horde API (list or indexed).
        horde_stats: Model statistics from Horde API (response or indexed).
        workers: Optional workers from Horde API (list or indexed).

    Returns:
        Dict of model_name -> CombinedModelStatistics with added runtime fields.
    """
    # Convert to indexed types if needed (supports both old and new API)
    indexed_status = IndexedHordeModelStatus(horde_status) if isinstance(horde_status, list) else horde_status
    indexed_stats = (
        IndexedHordeModelStats(horde_stats) if isinstance(horde_stats, HordeModelStatsResponse) else horde_stats
    )

    indexed_workers: IndexedHordeWorkers | None = None
    if workers is not None:
        indexed_workers = IndexedHordeWorkers(workers) if isinstance(workers, list) else workers

    all_merged_data: dict[str, CombinedModelStatistics] = {}

    for model_name in model_names:
        merged_data = merge_model_with_horde_data(
            model_name=model_name,
            horde_status=indexed_status,
            horde_stats=indexed_stats,
            workers=indexed_workers,
        )

        all_merged_data[model_name] = merged_data

    return all_merged_data
