"""Tests for data_merger module."""

from __future__ import annotations

from typing import Any

import pytest

from horde_model_reference.integrations.data_merger import (
    CombinedModelStatistics,
    UsageStats,
    WorkerSummary,
    merge_category_with_horde_data,
    merge_model_with_horde_data,
)
from horde_model_reference.integrations.horde_api_models import (
    HordeKudosDetails,
    HordeModelStatsResponse,
    HordeModelStatus,
    HordeWorker,
    HordeWorkerTeam,
    IndexedHordeModelStats,
    IndexedHordeModelStatus,
    IndexedHordeWorkers,
)


@pytest.fixture
def sample_reference_model() -> dict[str, Any]:
    """Sample model reference data."""
    return {
        "name": "test_model",
        "description": "A test model",
        "baseline": "stable_diffusion_xl",
        "nsfw": False,
    }


@pytest.fixture
def sample_horde_status() -> list[HordeModelStatus]:
    """Sample Horde API status data."""
    return [
        HordeModelStatus(
            name="test_model",
            count=5,
            jobs=10,
            performance=100.5,
            eta=30,
            queued=500,
            type="image",
        ),
        HordeModelStatus(
            name="other_model",
            count=3,
            jobs=5,
            performance=50.0,
            eta=60,
            queued=300,
            type="image",
        ),
    ]


@pytest.fixture
def sample_horde_stats() -> HordeModelStatsResponse:
    """Sample Horde API statistics data."""
    return HordeModelStatsResponse(
        day={"test_model": 100, "other_model": 50},
        month={"test_model": 3000, "other_model": 1500},
        total={"test_model": 50000, "other_model": 25000},
    )


@pytest.fixture
def sample_workers() -> list[HordeWorker]:
    """Sample Horde API worker data."""
    return [
        HordeWorker(
            id="worker-1",
            name="Test Worker 1",
            type="image",
            performance="100.5",
            requests_fulfilled=1000,
            kudos_rewards=5000.0,
            kudos_details=HordeKudosDetails(generated=4500.0, uptime=500.0),
            threads=2,
            uptime=86400,
            uncompleted_jobs=0,
            maintenance_mode=False,
            nsfw=True,
            trusted=True,
            flagged=False,
            online=True,
            models=["test_model", "other_model"],
            team=HordeWorkerTeam(name="Test Team", id="team-1"),
            bridge_agent="AI Horde Worker:1.0.0",
        ),
        HordeWorker(
            id="worker-2",
            name="Test Worker 2",
            type="image",
            performance="50.0",
            requests_fulfilled=500,
            kudos_rewards=2500.0,
            kudos_details=HordeKudosDetails(generated=2250.0, uptime=250.0),
            threads=1,
            uptime=43200,
            uncompleted_jobs=0,
            maintenance_mode=False,
            nsfw=False,
            trusted=False,
            flagged=False,
            online=True,
            models=["test_model"],
            team=HordeWorkerTeam(),
            bridge_agent="AI Horde Worker:1.0.0",
        ),
    ]


def test_merge_model_with_status_data(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging model with status data."""
    result = merge_model_with_horde_data(
        "test_model",
        sample_horde_status,
        sample_horde_stats,
    )

    # Result should be CombinedModelStatistics Pydantic model
    assert isinstance(result, CombinedModelStatistics)

    # Status fields present
    assert result.worker_count == 0  # Computed from worker_summaries (None)
    assert result.queued_jobs == 10
    assert result.performance == 100.5
    assert result.eta == 30
    assert result.queued == 500


def test_merge_model_with_stats_data(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging model with statistics data."""
    result = merge_model_with_horde_data(
        "test_model",
        sample_horde_status,
        sample_horde_stats,
    )

    # Usage stats added as UsageStats Pydantic model
    assert result.usage_stats is not None
    assert isinstance(result.usage_stats, UsageStats)
    assert result.usage_stats.day == 100
    assert result.usage_stats.month == 3000
    assert result.usage_stats.total == 50000


def test_merge_model_with_workers(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
    sample_workers: list[HordeWorker],
) -> None:
    """Test merging model with worker data."""
    result = merge_model_with_horde_data(
        "test_model",
        sample_horde_status,
        sample_horde_stats,
        workers=sample_workers,
    )

    # Workers added as dict of WorkerSummary
    assert result.worker_summaries is not None
    assert len(result.worker_summaries) == 2  # Both workers serve test_model
    assert all(isinstance(w, WorkerSummary) for w in result.worker_summaries.values())

    # Check first worker
    worker1 = result.worker_summaries["worker-1"]
    assert worker1.id == "worker-1"
    assert worker1.name == "Test Worker 1"
    assert worker1.performance == "100.5"
    assert worker1.online is True
    assert worker1.trusted is True


def test_merge_model_case_insensitive(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test that model name matching is case-insensitive."""
    # Use uppercase model name
    result = merge_model_with_horde_data(
        "TEST_MODEL",
        sample_horde_status,
        sample_horde_stats,
    )

    # Should still match and merge data
    assert result.worker_count == 0  # Computed from worker_summaries (None)
    assert result.usage_stats is not None
    assert result.usage_stats.day == 100


def test_merge_model_no_match(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging when model not found in Horde data."""
    result = merge_model_with_horde_data(
        "nonexistent_model",
        sample_horde_status,
        sample_horde_stats,
    )

    # No Horde data added - all fields should be None
    assert result.worker_count == 0  # Computed from worker_summaries (None)
    assert result.queued_jobs is None
    assert result.performance is None
    assert result.eta is None
    assert result.queued is None
    assert result.usage_stats is None
    assert result.worker_summaries is None


def test_merge_category_with_horde_data(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging entire category with Horde data."""
    reference_models = {
        "test_model": {
            "name": "test_model",
            "description": "Test model 1",
        },
        "other_model": {
            "name": "other_model",
            "description": "Test model 2",
        },
    }

    result = merge_category_with_horde_data(
        reference_models,
        sample_horde_status,
        sample_horde_stats,
    )

    # Check both models merged
    assert len(result) == 2
    assert "test_model" in result
    assert "other_model" in result

    # Check test_model - result values are CombinedModelStatistics
    test_model = result["test_model"]
    assert isinstance(test_model, CombinedModelStatistics)
    assert test_model.worker_count == 0  # Computed from worker_summaries (None)
    assert test_model.usage_stats is not None
    assert test_model.usage_stats.day == 100

    # Check other_model
    other_model = result["other_model"]
    assert isinstance(other_model, CombinedModelStatistics)
    assert other_model.worker_count == 0  # Computed from worker_summaries (None)
    assert other_model.usage_stats is not None
    assert other_model.usage_stats.day == 50


def test_merge_category_preserves_unmergeable_models(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test that models without Horde data are still included in result."""
    reference_models = {
        "test_model": {
            "name": "test_model",
            "description": "Test model 1",
        },
        "unknown_model": {
            "name": "unknown_model",
            "description": "Model not in Horde",
        },
    }

    result = merge_category_with_horde_data(
        reference_models,
        sample_horde_status,
        sample_horde_stats,
    )

    # Both models should be in result
    assert len(result) == 2
    assert "test_model" in result
    assert "unknown_model" in result

    # test_model should have Horde data
    test_model = result["test_model"]
    assert test_model.queued_jobs is not None
    assert test_model.queued_jobs == 10

    # unknown_model should not have Horde data
    unknown_model = result["unknown_model"]
    assert unknown_model.worker_count == 0  # Computed from worker_summaries (None)
    assert unknown_model.queued_jobs is None


def test_merge_category_with_indexed_types(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
    sample_workers: list[HordeWorker],
) -> None:
    """Test that merge_category_with_horde_data works with pre-indexed types."""
    reference_models = {
        "test_model": {
            "name": "test_model",
            "description": "Test model 1",
        },
        "other_model": {
            "name": "other_model",
            "description": "Test model 2",
        },
    }

    # Create indexed types (simulating what HordeAPIIntegration.get_*_indexed() returns)
    indexed_status = IndexedHordeModelStatus(sample_horde_status)
    indexed_stats = IndexedHordeModelStats(sample_horde_stats)
    indexed_workers = IndexedHordeWorkers(sample_workers)

    # Merge using indexed types
    result = merge_category_with_horde_data(
        reference_models,
        indexed_status,
        indexed_stats,
        indexed_workers,
    )

    # Verify results are the same as with lists
    assert len(result) == 2

    test_model = result["test_model"]
    assert test_model.worker_count == 2  # Computed from worker_summaries dict
    assert test_model.usage_stats is not None
    assert test_model.usage_stats.day == 100
    assert test_model.worker_summaries is not None
    assert len(test_model.worker_summaries) == 2

    other_model = result["other_model"]
    assert other_model.worker_count == 1  # Computed from worker_summaries dict
    assert other_model.usage_stats is not None
    assert other_model.usage_stats.day == 50
    assert other_model.worker_summaries is not None
    assert len(other_model.worker_summaries) == 1


def test_merge_model_with_status_but_no_stats(
    sample_horde_status: list[HordeModelStatus],
) -> None:
    """Test merging when model has status data but no stats data."""
    # Create empty stats response
    empty_stats = HordeModelStatsResponse(day={}, month={}, total={})

    result = merge_model_with_horde_data(
        "test_model",
        sample_horde_status,
        empty_stats,
    )

    # Status fields should be present
    assert result.queued_jobs == 10
    assert result.performance == 100.5
    assert result.eta == 30
    assert result.queued == 500

    # Stats should be None
    assert result.usage_stats is None
    assert result.worker_count == 0
    assert result.worker_summaries is None


def test_merge_model_with_stats_but_no_status(
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging when model has stats data but no status data."""
    # Create empty status list
    empty_status: list[HordeModelStatus] = []

    result = merge_model_with_horde_data(
        "test_model",
        empty_status,
        sample_horde_stats,
    )

    # Status fields should be None
    assert result.queued_jobs is None
    assert result.performance is None
    assert result.eta is None
    assert result.queued is None

    # Stats should be present
    assert result.usage_stats is not None
    assert result.usage_stats.day == 100
    assert result.usage_stats.month == 3000
    assert result.usage_stats.total == 50000

    # Worker fields should be None
    assert result.worker_count == 0
    assert result.worker_summaries is None


def test_merge_model_with_workers_but_model_not_in_workers(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
    sample_workers: list[HordeWorker],
) -> None:
    """Test merging when workers are provided but model is not served by any worker."""
    result = merge_model_with_horde_data(
        "other_model",  # Only worker-1 serves this model
        sample_horde_status,
        sample_horde_stats,
        workers=sample_workers,
    )

    # Should have status and stats
    assert result.queued_jobs == 5
    assert result.usage_stats is not None
    assert result.usage_stats.day == 50

    # Should have worker summaries for worker-1 only
    assert result.worker_summaries is not None
    assert len(result.worker_summaries) == 1
    assert "worker-1" in result.worker_summaries
    assert result.worker_count == 1


def test_merge_model_with_none_workers_explicitly(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
) -> None:
    """Test merging when workers=None is explicitly passed."""
    result = merge_model_with_horde_data(
        "test_model",
        sample_horde_status,
        sample_horde_stats,
        workers=None,
    )

    # Should have status and stats
    assert result.queued_jobs == 10
    assert result.usage_stats is not None
    assert result.usage_stats.day == 100

    # Workers should be None
    assert result.worker_summaries is None
    assert result.worker_count == 0
