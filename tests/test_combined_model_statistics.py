"""Tests for CombinedModelStatistics worker_count computed field.

Tests the worker_count field's fallback behavior when worker_summaries is None.
"""

from __future__ import annotations

from horde_model_reference.integrations.data_merger import CombinedModelStatistics, WorkerSummary


def test_worker_count_from_worker_summaries() -> None:
    """Test that worker_count returns len(worker_summaries) when available."""
    stats = CombinedModelStatistics(
        worker_count_from_status=10,  # Should be ignored
        worker_summaries={
            "w1": WorkerSummary(id="w1", name="worker1", performance="1.0", online=True, trusted=True, uptime=100),
            "w2": WorkerSummary(id="w2", name="worker2", performance="2.0", online=True, trusted=True, uptime=200),
        },
    )

    assert stats.worker_count == 2, "worker_count should use len(worker_summaries)"


def test_worker_count_from_status_fallback() -> None:
    """Test that worker_count falls back to worker_count_from_status when worker_summaries is None."""
    stats = CombinedModelStatistics(
        worker_count_from_status=5,
        worker_summaries=None,  # Explicit None
    )

    assert stats.worker_count == 5, "worker_count should fall back to worker_count_from_status"


def test_worker_count_zero_when_no_data() -> None:
    """Test that worker_count returns 0 when neither source is available."""
    stats = CombinedModelStatistics(
        worker_count_from_status=None,
        worker_summaries=None,
    )

    assert stats.worker_count == 0, "worker_count should return 0 when no data available"


def test_worker_count_empty_worker_summaries() -> None:
    """Test that worker_count returns 0 when worker_summaries is empty dict."""
    stats = CombinedModelStatistics(
        worker_count_from_status=5,  # Should be ignored
        worker_summaries={},  # Empty dict
    )

    assert stats.worker_count == 0, "worker_count should return 0 for empty worker_summaries"


def test_worker_count_zero_status_with_summaries() -> None:
    """Test that worker_count prefers worker_summaries even when status shows different count."""
    stats = CombinedModelStatistics(
        worker_count_from_status=0,
        worker_summaries={
            "w1": WorkerSummary(id="w1", name="worker1", performance="1.0", online=True, trusted=True, uptime=100),
        },
    )

    assert stats.worker_count == 1, "worker_count should use worker_summaries even when status differs"
