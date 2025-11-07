"""Integration tests for audit analysis with live Horde API data.

Contains integration tests hitting live API (slower, network-dependent, marked with @pytest.mark.integration)

Golden models (e.g., stable_diffusion, stable_diffusion_xl) should never be flagged as critical
or marked for deletion regardless of current usage statistics.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Golden models that should never be flagged for deletion
GOLDEN_IMAGE_MODELS: set[str] = {
    "stable_diffusion",
    "stable_diffusion_2.1",
    "stable_diffusion_xl",
    "stable_diffusion_3",
}

GOLDEN_TEXT_MODELS: set[str] = set()
# Add golden text models here when identified

# Performance threshold for full category audit (seconds)
MAX_AUDIT_TIME_SECONDS = 15.0


def verify_audit_response_structure(response_data: dict[str, Any]) -> None:
    """Verify the structure of an audit response.

    Args:
        response_data: The parsed JSON response from audit endpoint
    """
    assert "summary" in response_data
    assert "models" in response_data

    summary = response_data["summary"]
    assert isinstance(summary, dict)
    assert "total_models" in summary
    assert "models_at_risk" in summary
    assert "models_critical" in summary
    assert "average_risk_score" in summary
    assert "category_total_month_usage" in summary

    models = response_data["models"]
    assert isinstance(models, list)


def verify_golden_models_not_critical(
    response_data: dict[str, Any],
    golden_models: set[str],
) -> None:
    """Verify that golden models are never marked as critical.

    Args:
        response_data: The parsed JSON response from audit endpoint
        golden_models: Set of model names that should never be critical
    """
    models = response_data["models"]
    assert isinstance(models, list)

    for model in models:
        if model["name"] in golden_models or model.get("baseline") in golden_models:
            assert not model.get(
                "is_critical", False
            ), f"Golden model {model['name']} (baseline: {model.get('baseline')}) should never be marked as critical"


def verify_summary_consistency(response_data: dict[str, Any]) -> None:
    """Verify that summary statistics are consistent with total model count.

    Note: Summary reflects ALL models in category, not just filtered/paginated results.
    The models list may be filtered by preset or paginated, so we verify against total_count.

    Args:
        response_data: The parsed JSON response from audit endpoint
    """
    summary = response_data["summary"]
    total_count = response_data["total_count"]
    assert isinstance(summary, dict)

    # Summary total_models should match total_count (not returned_count which may be filtered)
    assert (
        summary["total_models"] == total_count
    ), f"Summary total_models ({summary['total_models']}) does not match total_count ({total_count})"

    # Verify summary counts are non-negative and don't exceed total
    assert 0 <= summary["models_at_risk"] <= summary["total_models"]
    assert 0 <= summary["models_critical"] <= summary["total_models"]
    assert summary["average_risk_score"] >= 0.0


class TestAuditCacheBehavior:
    """Test audit caching behavior."""

    def test_audit_cache_consistency(self, api_client: TestClient) -> None:
        """Test that repeated audit calls within cache window return consistent results."""
        # First call
        response1 = api_client.get("/model_references/statistics/image_generation/audit")
        assert response1.status_code == 200
        data1 = response1.json()

        # Second call (should hit cache)
        response2 = api_client.get("/model_references/statistics/image_generation/audit")
        assert response2.status_code == 200
        data2 = response2.json()

        # Results should be identical
        assert data1["summary"]["total_models"] == data2["summary"]["total_models"]
        assert data1["summary"]["models_at_risk"] == data2["summary"]["models_at_risk"]
        assert len(data1["models"]) == len(data2["models"])


class TestAuditPerformance:
    """Test audit performance characteristics."""

    def test_audit_completes_within_threshold(self, api_client: TestClient) -> None:
        """Test that full category audit completes within performance threshold."""
        start_time = time.time()

        response = api_client.get("/model_references/statistics/image_generation/audit")

        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert (
            elapsed < MAX_AUDIT_TIME_SECONDS
        ), f"Audit took {elapsed:.2f}s, exceeding threshold of {MAX_AUDIT_TIME_SECONDS}s"

    def test_audit_sorting_by_usage(self, api_client: TestClient) -> None:
        """Test that models are sorted by usage (descending)."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        models = data["models"]
        if len(models) > 1:
            # Check that usage_month is in descending order
            usage_values = [m.get("usage_month", 0) for m in models]
            assert usage_values == sorted(
                usage_values, reverse=True
            ), "Models should be sorted by usage_month in descending order"


class TestAuditEdgeCases:
    """Test edge cases in audit analysis."""

    def test_models_with_zero_workers(self, api_client: TestClient) -> None:
        """Test handling of models with zero active workers."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        # Find models with zero workers
        zero_worker_models = [m for m in data["models"] if m.get("worker_count", 0) == 0]

        # They should be flagged with no_active_workers
        for model in zero_worker_models:
            flags = model.get("deletion_risk_flags", {})
            assert flags.get(
                "no_active_workers", False
            ), f"Model {model['name']} with zero workers should have no_active_workers flag"

    def test_models_with_multiple_hosts(self, api_client: TestClient) -> None:
        """Test identification of models with multiple download hosts."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        # Find models with multiple hosts
        for model in data["models"]:
            download_hosts = model.get("download_hosts", [])
            if len(download_hosts) > 1:
                flags = model.get("deletion_risk_flags", {})
                assert flags.get(
                    "has_multiple_hosts", False
                ), f"Model {model['name']} with {len(download_hosts)} hosts should have has_multiple_hosts flag"

    def test_usage_percentage_calculations(self, api_client: TestClient) -> None:
        """Test that usage percentage calculations are accurate."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        category_total = data["summary"]["category_total_month_usage"]

        if category_total > 0:
            for model in data["models"]:
                usage_month = model.get("usage_month", 0)
                usage_percentage = model.get("usage_percentage_of_category", 0)

                expected_percentage = (usage_month / category_total) * 100
                assert abs(usage_percentage - expected_percentage) < 0.01, (
                    f"Model {model['name']} usage percentage {usage_percentage}% "
                    f"does not match expected {expected_percentage}%"
                )

    def test_cost_benefit_score_calculations(self, api_client: TestClient) -> None:
        """Test that cost-benefit scores are calculated correctly."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        for model in data["models"]:
            cost_benefit = model.get("cost_benefit_score")
            usage_month = model.get("usage_month", 0)
            size_gb = model.get("size_gb")

            if size_gb and size_gb > 0:
                expected_cost_benefit = usage_month / size_gb
                assert cost_benefit is not None
                # Use wider tolerance for floating-point precision (cost-benefit can be large numbers)
                tolerance = max(abs(expected_cost_benefit) * 0.01, 50.0)  # 1% or 50, whichever is larger
                assert abs(cost_benefit - expected_cost_benefit) < tolerance, (
                    f"Model {model['name']} cost-benefit score {cost_benefit} "
                    f"does not match expected {expected_cost_benefit} (tolerance: {tolerance})"
                )
            else:
                assert cost_benefit is None, f"Model {model['name']} without size should have None cost_benefit_score"


# Integration tests that hit the live API
@pytest.mark.integration
class TestAuditLiveIntegration:
    """Integration tests against live Horde API (marked with @pytest.mark.integration)."""

    def test_live_image_generation_audit(self, api_client: TestClient) -> None:
        """Test image generation audit against live Horde API."""
        response = api_client.get("/model_references/statistics/image_generation/audit")

        assert response.status_code == 200
        data = response.json()

        verify_audit_response_structure(data)
        verify_summary_consistency(data)
        verify_golden_models_not_critical(data, GOLDEN_IMAGE_MODELS)

    def test_live_text_generation_audit(self, api_client: TestClient) -> None:
        """Test text generation audit against live Horde API."""
        response = api_client.get("/model_references/statistics/text_generation/audit")

        assert response.status_code == 200
        data = response.json()

        verify_audit_response_structure(data)
        verify_summary_consistency(data)

    def test_live_audit_performance(self, api_client: TestClient) -> None:
        """Test audit performance against live API."""
        start_time = time.time()

        response = api_client.get("/model_references/statistics/image_generation/audit")

        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert (
            elapsed < MAX_AUDIT_TIME_SECONDS
        ), f"Live audit took {elapsed:.2f}s, exceeding threshold of {MAX_AUDIT_TIME_SECONDS}s"

    def test_live_golden_models_consistency(self, api_client: TestClient) -> None:
        """Test that golden models maintain consistent non-critical status."""
        # Run audit twice
        response1 = api_client.get("/model_references/statistics/image_generation/audit")
        assert response1.status_code == 200
        data1 = response1.json()

        time.sleep(2)  # Brief pause

        response2 = api_client.get("/model_references/statistics/image_generation/audit")
        assert response2.status_code == 200
        data2 = response2.json()

        # Both should not flag golden models as critical
        verify_golden_models_not_critical(data1, GOLDEN_IMAGE_MODELS)
        verify_golden_models_not_critical(data2, GOLDEN_IMAGE_MODELS)
