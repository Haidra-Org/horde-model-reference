"""Test that audit endpoint correctly reports worker counts.

This test reproduces the bug where all models show worker_count=0 in the audit endpoint
even though the Horde API status data contains valid worker counts.

Issue: The audit endpoint passes workers=None to merge_category_with_horde_data, which
causes the worker_count computed field to return 0 instead of using the count field
from HordeModelStatus.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_audit_endpoint_includes_worker_counts(api_client: TestClient) -> None:
    """Test that audit endpoint correctly reports worker counts from Horde API status data.

    This test verifies that the audit endpoint doesn't show all models as having 0 workers
    when the Horde API status data actually contains worker counts.

    Expected behavior:
    - Models with workers in Horde API status should have worker_count > 0
    - Models without workers should have worker_count = 0
    - At least some popular models should have workers (not all zero)
    - no_active_workers flag should be False for models with workers
    """
    response = api_client.get("/model_references/statistics/image_generation/audit")

    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "summary" in data
    models = data["models"]

    # Should have models in the response
    assert len(models) > 0, "Expected at least some models in audit response"

    # Track statistics about worker counts
    models_with_workers = []
    models_without_workers = []
    total_models = len(models)

    for model in models:
        assert "name" in model
        assert "worker_count" in model
        assert "deletion_risk_flags" in model
        assert "no_active_workers" in model["deletion_risk_flags"]

        model_name = model["name"]
        worker_count = model["worker_count"]
        no_active_workers_flag = model["deletion_risk_flags"]["no_active_workers"]

        if worker_count > 0:
            models_with_workers.append(model_name)
            # If worker_count > 0, the no_active_workers flag should be False
            assert no_active_workers_flag is False, (
                f"Model '{model_name}' has {worker_count} workers but no_active_workers flag is True"
            )
        else:
            models_without_workers.append(model_name)
            # If worker_count == 0, the no_active_workers flag should be True
            assert no_active_workers_flag is True, (
                f"Model '{model_name}' has 0 workers but no_active_workers flag is False"
            )

    # The bug we're testing for: ALL models showing worker_count=0
    # This assertion will FAIL with the current buggy implementation
    assert len(models_with_workers) > 0, (
        f"BUG: All {total_models} models show worker_count=0, but some should have workers from Horde API status data"
    )

    # At least some models should have workers
    # In a realistic dataset, not all models are served by workers
    # but popular models definitely should be
    percentage_with_workers = (len(models_with_workers) / total_models) * 100

    # Log the results for debugging
    print("\n=== Worker Count Statistics ===")
    print(f"Total models: {total_models}")
    print(f"Models with workers: {len(models_with_workers)} ({percentage_with_workers:.1f}%)")
    print(f"Models without workers: {len(models_without_workers)}")
    print("\nFirst 10 models with workers:")
    for name in models_with_workers[:10]:
        model = next(m for m in models if m["name"] == name)
        print(f"  - {name}: {model['worker_count']} workers")
    print("\nFirst 10 models without workers:")
    for name in models_without_workers[:10]:
        print(f"  - {name}")

    # Verify summary statistics match
    summary = data["summary"]
    assert "models_with_no_active_workers" in summary
    expected_count = len(models_without_workers)
    actual_count = summary["models_with_no_active_workers"]
    assert actual_count == expected_count, f"Summary count {actual_count} doesn't match actual count {expected_count}"


@pytest.mark.integration
def test_audit_endpoint_worker_count_consistency(api_client: TestClient) -> None:
    """Test that worker_count and no_active_workers flag are always consistent.

    This is a stricter test that verifies the relationship between:
    - worker_count field
    - no_active_workers deletion risk flag
    """
    response = api_client.get("/model_references/statistics/image_generation/audit")

    assert response.status_code == 200
    data = response.json()

    models = data["models"]
    inconsistent_models = []

    for model in models:
        worker_count = model["worker_count"]
        no_active_workers = model["deletion_risk_flags"]["no_active_workers"]

        # These must always be consistent
        expected_flag = worker_count == 0

        if no_active_workers != expected_flag:
            inconsistent_models.append(
                {
                    "name": model["name"],
                    "worker_count": worker_count,
                    "no_active_workers_flag": no_active_workers,
                    "expected_flag": expected_flag,
                }
            )

    assert len(inconsistent_models) == 0, (
        f"Found {len(inconsistent_models)} models with inconsistent worker_count "
        f"and no_active_workers flag: {inconsistent_models}"
    )
