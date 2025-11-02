"""Tests for indexed Horde API types."""

from __future__ import annotations

import pytest

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
def sample_horde_status() -> list[HordeModelStatus]:
    """Sample Horde API status data."""
    return [
        HordeModelStatus(
            name="TestModel",
            count=5,
            jobs=10,
            performance=100.5,
            eta=30,
            queued=500,
            type="image",
        ),
        HordeModelStatus(
            name="AnotherModel",
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
        day={"TestModel": 100, "AnotherModel": 50},
        month={"TestModel": 3000, "AnotherModel": 1500},
        total={"TestModel": 50000, "AnotherModel": 25000},
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
            models=["TestModel", "AnotherModel"],
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
            models=["TestModel"],
            team=HordeWorkerTeam(),
            bridge_agent="AI Horde Worker:1.0.0",
        ),
    ]


def test_indexed_status_lookups(sample_horde_status: list[HordeModelStatus]) -> None:
    """Test IndexedHordeModelStatus provides O(1) lookups."""
    indexed = IndexedHordeModelStatus(sample_horde_status)

    # Case-insensitive exact match
    result = indexed.get("TestModel")
    assert result is not None
    assert result.name == "TestModel"
    assert result.count == 5

    # Case-insensitive lowercase
    result = indexed.get("testmodel")
    assert result is not None
    assert result.count == 5

    # Case-insensitive uppercase
    result = indexed.get("TESTMODEL")
    assert result is not None
    assert result.count == 5

    # Not found
    result = indexed.get("NonExistent")
    assert result is None

    # Get all
    all_status = indexed.get_all()
    assert len(all_status) == 2


def test_indexed_stats_lookups(sample_horde_stats: HordeModelStatsResponse) -> None:
    """Test IndexedHordeModelStats provides O(1) lookups."""
    indexed = IndexedHordeModelStats(sample_horde_stats)

    # Case-sensitive exact match
    assert indexed.get_day("TestModel") == 100
    assert indexed.get_month("TestModel") == 3000
    assert indexed.get_total("TestModel") == 50000

    # Case-insensitive lowercase
    assert indexed.get_day("testmodel") == 100
    assert indexed.get_month("testmodel") == 3000
    assert indexed.get_total("testmodel") == 50000

    # Check has_stats
    assert indexed.has_stats("TestModel") is True
    assert indexed.has_stats("testmodel") is True
    assert indexed.has_stats("NonExistent") is False

    # Not found returns None
    assert indexed.get_day("NonExistent") is None
    assert indexed.get_month("NonExistent") is None
    assert indexed.get_total("NonExistent") is None


def test_indexed_workers_lookups(sample_workers: list[HordeWorker]) -> None:
    """Test IndexedHordeWorkers provides O(1) lookups."""
    indexed = IndexedHordeWorkers(sample_workers)

    # Case-sensitive exact match
    workers = indexed.get("TestModel")
    assert len(workers) == 2
    assert workers[0].id == "worker-1"
    assert workers[1].id == "worker-2"

    # Case-insensitive lowercase
    workers = indexed.get("testmodel")
    assert len(workers) == 2

    # Case-insensitive uppercase
    workers = indexed.get("TESTMODEL")
    assert len(workers) == 2

    # Model with single worker
    workers = indexed.get("AnotherModel")
    assert len(workers) == 1
    assert workers[0].id == "worker-1"

    # Not found returns empty list
    workers = indexed.get("NonExistent")
    assert workers == []

    # Get all unique workers
    all_workers = indexed.get_all()
    assert len(all_workers) == 2
    worker_ids = {w.id for w in all_workers}
    assert worker_ids == {"worker-1", "worker-2"}


def test_indexed_types_preserve_original_data(
    sample_horde_status: list[HordeModelStatus],
    sample_horde_stats: HordeModelStatsResponse,
    sample_workers: list[HordeWorker],
) -> None:
    """Test that indexed types preserve all original data."""
    indexed_status = IndexedHordeModelStatus(sample_horde_status)
    indexed_stats = IndexedHordeModelStats(sample_horde_stats)
    indexed_workers = IndexedHordeWorkers(sample_workers)

    # Status data preserved
    status = indexed_status.get("TestModel")
    assert status is not None
    assert status.performance == 100.5
    assert status.eta == 30
    assert status.queued == 500
    assert status.type == "image"

    # Stats data preserved
    assert indexed_stats.get_day("TestModel") == 100
    assert indexed_stats.get_month("AnotherModel") == 1500
    assert indexed_stats.get_total("AnotherModel") == 25000

    # Worker data preserved
    workers = indexed_workers.get("TestModel")
    assert len(workers) == 2
    assert workers[0].name == "Test Worker 1"
    assert workers[0].trusted is True
    assert workers[0].online is True
    assert workers[1].name == "Test Worker 2"
    assert workers[1].trusted is False
