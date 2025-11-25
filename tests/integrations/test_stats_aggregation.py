"""Tests for stats aggregation across quantization variants."""

from __future__ import annotations

from horde_model_reference.integrations.horde_api_models import (
    HordeModelStatsResponse,
    IndexedHordeModelStats,
)


class TestIndexedHordeModelStatsAggregation:
    """Test suite for stats aggregation with quantization variants."""

    def test_aggregate_stats_with_quantization_variants(self) -> None:
        """Test that stats aggregate correctly across quantization variants."""
        stats = HordeModelStatsResponse(
            day={
                "koboldcpp/Lumimaid-v0.2-8B": 4080,
                "koboldcpp/Lumimaid-v0.2-8B-Q8_0": 1500,
                "koboldcpp/Lumimaid-v0.2-8B-Q4_K_M": 800,
            },
            month={
                "koboldcpp/Lumimaid-v0.2-8B": 40000,
                "koboldcpp/Lumimaid-v0.2-8B-Q8_0": 15000,
                "koboldcpp/Lumimaid-v0.2-8B-Q4_K_M": 8000,
            },
            total={
                "koboldcpp/Lumimaid-v0.2-8B": 400000,
                "koboldcpp/Lumimaid-v0.2-8B-Q8_0": 150000,
                "koboldcpp/Lumimaid-v0.2-8B-Q4_K_M": 80000,
            },
        )

        indexed = IndexedHordeModelStats(stats)
        day, month, total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        # Should aggregate all quantization variants
        assert day == 4080 + 1500 + 800
        assert month == 40000 + 15000 + 8000
        assert total == 400000 + 150000 + 80000

    def test_aggregate_stats_with_org_prefix_variants(self) -> None:
        """Test that stats aggregate across org-prefixed variants."""
        stats = HordeModelStatsResponse(
            day={
                "Lumimaid-v0.2-8B": 1000,
                "aphrodite/NeverSleep/Lumimaid-v0.2-8B": 2000,
                "koboldcpp/Lumimaid-v0.2-8B": 3000,
            },
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        day, _month, _total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        # Should aggregate across all prefixes
        assert day == 1000 + 2000 + 3000

    def test_aggregate_stats_with_all_variants(self) -> None:
        """Test that stats aggregate across backend, org, and quantization variants."""
        stats = HordeModelStatsResponse(
            day={
                "Lumimaid-v0.2-8B": 100,
                "aphrodite/Lumimaid-v0.2-8B": 200,
                "aphrodite/NeverSleep/Lumimaid-v0.2-8B": 300,
                "koboldcpp/Lumimaid-v0.2-8B": 400,
                "koboldcpp/Lumimaid-v0.2-8B-Q8_0": 500,
                "koboldcpp/Lumimaid-v0.2-8B-Q4_K_M": 600,
            },
            month={
                "Lumimaid-v0.2-8B": 1000,
                "aphrodite/NeverSleep/Lumimaid-v0.2-8B": 3000,
            },
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        day, month, _total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        assert day == 100 + 200 + 300 + 400 + 500 + 600
        assert month == 1000 + 3000

    def test_get_stats_with_variations_backend_breakdown(self) -> None:
        """Test that variations are correctly grouped by backend."""
        stats = HordeModelStatsResponse(
            day={
                "Lumimaid-v0.2-8B": 100,
                "aphrodite/NeverSleep/Lumimaid-v0.2-8B": 200,
                "koboldcpp/Lumimaid-v0.2-8B": 300,
                "koboldcpp/Lumimaid-v0.2-8B-Q8_0": 400,
            },
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        (day_total, _month, _total), variations = indexed.get_stats_with_variations("Lumimaid-v0.2-8B")

        assert day_total == 100 + 200 + 300 + 400
        assert "canonical" in variations
        assert "aphrodite" in variations
        assert "koboldcpp" in variations
        assert variations["canonical"][0] == 100
        assert variations["aphrodite"][0] == 200
        assert variations["koboldcpp"][0] == 300 + 400  # Aggregates quant variants

    def test_aggregate_stats_different_base_models_not_mixed(self) -> None:
        """Test that different base models are not incorrectly aggregated."""
        stats = HordeModelStatsResponse(
            day={
                "Lumimaid-v0.2-8B": 100,
                "Lumimaid-v0.2-8B-Q8_0": 200,
                "Llama-3-8B": 1000,
                "Llama-3-8B-Q4_K_M": 2000,
            },
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)

        lumimaid_day, _m, _t = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")
        llama_day, _m, _t = indexed.get_aggregated_stats("Llama-3-8B")

        # Each base model should only aggregate its own variants
        assert lumimaid_day == 100 + 200
        assert llama_day == 1000 + 2000

    def test_aggregate_stats_case_insensitive(self) -> None:
        """Test that aggregation is case-insensitive."""
        stats = HordeModelStatsResponse(
            day={
                "KOBOLDCPP/Lumimaid-v0.2-8B": 100,
                "koboldcpp/lumimaid-v0.2-8b-q8_0": 200,
            },
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        day, _m, _t = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        assert day == 100 + 200

    def test_aggregate_stats_no_duplicates(self) -> None:
        """Test that stats are not double-counted when exact match exists."""
        stats = HordeModelStatsResponse(
            day={
                "Lumimaid-v0.2-8B": 100,  # Exact match and base name match
            },
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        day, _m, _t = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        # Should only count once, not twice
        assert day == 100

    def test_aggregate_stats_empty_response(self) -> None:
        """Test aggregation with empty stats response."""
        stats = HordeModelStatsResponse(day={}, month={}, total={})

        indexed = IndexedHordeModelStats(stats)
        day, month, total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        assert day == 0
        assert month == 0
        assert total == 0

    def test_aggregate_stats_model_not_found(self) -> None:
        """Test aggregation when model doesn't exist in stats."""
        stats = HordeModelStatsResponse(
            day={"SomeOtherModel": 100},
            month={},
            total={},
        )

        indexed = IndexedHordeModelStats(stats)
        day, month, total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")

        assert day == 0
        assert month == 0
        assert total == 0
