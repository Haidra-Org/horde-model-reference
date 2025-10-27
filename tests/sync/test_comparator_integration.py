"""Integration tests for ModelReferenceComparator using live PRIMARY instance.

These tests require a running PRIMARY instance (default: http://localhost:19800).
Set HORDE_TEST_PRIMARY_API_URL environment variable to override the default.

To skip these tests in CI:
    pytest -m "not integration"

To run only integration tests:
    pytest -m integration
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
from loguru import logger

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.sync import ModelReferenceComparator

# Pytest marks for integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def primary_api_url() -> str:
    """Get PRIMARY API URL from environment or use default.

    Returns:
        The PRIMARY API base URL (including /api prefix).
    """
    return os.getenv("HORDE_TEST_PRIMARY_API_URL", "http://localhost:19800/api")


@pytest.fixture(scope="session")
def primary_instance_available(primary_api_url: str) -> bool:
    """Check if PRIMARY instance is available.

    Args:
        primary_api_url: The PRIMARY API base URL.

    Returns:
        True if PRIMARY instance is responding, False otherwise.
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{primary_api_url}/heartbeat")
            available = response.status_code == 200
            if available:
                logger.info(f"PRIMARY instance available at {primary_api_url}")
            else:
                logger.warning(f"PRIMARY instance returned status {response.status_code}")
            return available
    except Exception as e:
        logger.warning(f"PRIMARY instance not available at {primary_api_url}: {e}")
        return False


@pytest.fixture
def skip_if_primary_unavailable(primary_instance_available: bool) -> None:
    """Skip test if PRIMARY instance is not available.

    Args:
        primary_instance_available: Whether PRIMARY instance is responding.
    """
    if not primary_instance_available:
        pytest.skip("PRIMARY instance not available")


def fetch_primary_category_data(
    *,
    api_url: str,
    category: MODEL_REFERENCE_CATEGORY,
    timeout: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """Fetch model reference data from PRIMARY v1 API.

    Args:
        api_url: Base URL of PRIMARY API.
        category (MODEL_REFERENCE_CATEGORY): The category to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary of model records in legacy format.

    Raises:
        httpx.HTTPError: If the request fails.
    """
    endpoint = f"{api_url.rstrip('/')}/model_references/v1/{category}"
    logger.debug(f"Fetching from {endpoint}")

    with httpx.Client(timeout=timeout) as client:
        response = client.get(endpoint)
        response.raise_for_status()
        data: dict[str, dict[str, Any]] = response.json()

    logger.info(f"Fetched {len(data)} models for {category}")
    return data


def normalize_for_comparison(data: dict[str, Any]) -> str:
    """Normalize data for comparison by converting to sorted JSON.

    This ensures consistent comparison regardless of field ordering or
    serialization artifacts.

    Args:
        data: The data to normalize.

    Returns:
        JSON string with sorted keys.
    """
    return json.dumps(data, sort_keys=True, indent=2)


def deep_compare_models(
    model1: dict[str, Any],
    model2: dict[str, Any],
    path: str = "",
) -> list[str]:
    """Recursively compare two model dictionaries and report differences.

    Args:
        model1: First model dictionary.
        model2: Second model dictionary.
        path: Current path in the data structure (for error reporting).

    Returns:
        List of difference descriptions.
    """
    differences: list[str] = []

    all_keys = set(model1.keys()) | set(model2.keys())

    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key

        if key not in model1:
            differences.append(f"{current_path}: missing in first model")
            continue

        if key not in model2:
            differences.append(f"{current_path}: missing in second model")
            continue

        val1 = model1[key]
        val2 = model2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            differences.extend(deep_compare_models(val1, val2, current_path))
        elif isinstance(val1, list) and isinstance(val2, list):
            if len(val1) != len(val2):
                differences.append(f"{current_path}: list length differs ({len(val1)} vs {len(val2)})")
            else:
                for i, (item1, item2) in enumerate(zip(val1, val2, strict=False)):
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        differences.extend(deep_compare_models(item1, item2, f"{current_path}[{i}]"))
                    elif item1 != item2:
                        differences.append(
                            f"{current_path}[{i}]: {item1!r} ({type(item1).__name__}) != "
                            f"{item2!r} ({type(item2).__name__})"
                        )
        elif val1 != val2:
            differences.append(f"{current_path}: {val1!r} ({type(val1).__name__}) != {val2!r} ({type(val2).__name__})")

    return differences


class TestModelReferenceComparatorIntegration:
    """Integration tests for ModelReferenceComparator using live PRIMARY instance."""

    @pytest.fixture
    def comparator(self) -> ModelReferenceComparator:
        """Create a comparator instance for testing.

        Returns:
            A fresh ModelReferenceComparator instance.
        """
        return ModelReferenceComparator()

    def test_identical_data_no_changes(
        self,
        comparator: ModelReferenceComparator,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
    ) -> None:
        """Test that comparing identical data from PRIMARY shows no changes.

        This is the core test - if we fetch the same data twice and compare it,
        there should be zero differences detected.

        Args:
            comparator: The comparator instance.
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        data1 = fetch_primary_category_data(api_url=primary_api_url, category=category)
        data2 = fetch_primary_category_data(api_url=primary_api_url, category=category)

        logger.info(f"Comparing {len(data1)} models against {len(data2)} models")

        diff = comparator.compare_categories(
            category=category,
            primary_data=data1,
            github_data=data2,
        )

        if diff.has_changes():
            logger.error("UNEXPECTED CHANGES DETECTED!")
            logger.error(f"Summary:\n{diff.summary()}")

            if diff.added_models:
                logger.error(f"Added models: {list(diff.added_models.keys())}")

            if diff.removed_models:
                logger.error(f"Removed models: {list(diff.removed_models.keys())}")

            if diff.modified_models:
                logger.error(f"Modified models: {list(diff.modified_models.keys())}")

                for model_name in list(diff.modified_models.keys())[:3]:
                    logger.error(f"\nDetailed comparison for '{model_name}':")
                    differences = deep_compare_models(data1[model_name], data2[model_name], model_name)
                    for difference in differences:
                        logger.error(f"  {difference}")

        assert not diff.has_changes(), f"Expected no changes, but found {diff.total_changes()} differences"
        assert len(diff.added_models) == 0, f"Found {len(diff.added_models)} added models (expected 0)"
        assert len(diff.removed_models) == 0, f"Found {len(diff.removed_models)} removed models (expected 0)"
        assert len(diff.modified_models) == 0, f"Found {len(diff.modified_models)} modified models (expected 0)"

    @pytest.mark.parametrize(
        "category",
        [
            MODEL_REFERENCE_CATEGORY.image_generation,
            MODEL_REFERENCE_CATEGORY.text_generation,
            MODEL_REFERENCE_CATEGORY.controlnet,
            MODEL_REFERENCE_CATEGORY.clip,
        ],
    )
    def test_per_category_no_spurious_changes(
        self,
        comparator: ModelReferenceComparator,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
        category: MODEL_REFERENCE_CATEGORY,
    ) -> None:
        """Test each category individually to ensure no spurious changes.

        Args:
            comparator: The comparator instance.
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
            category (MODEL_REFERENCE_CATEGORY): The category to test.
        """
        try:
            data1 = fetch_primary_category_data(api_url=primary_api_url, category=category)
            data2 = fetch_primary_category_data(api_url=primary_api_url, category=category)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                pytest.skip(f"Category {category} not available in PRIMARY instance")
            raise

        if not data1:
            pytest.skip(f"Category {category} has no data")

        diff = comparator.compare_categories(
            category=category,
            primary_data=data1,
            github_data=data2,
        )

        if diff.has_changes():
            logger.error(f"Category {category} has unexpected changes:")
            logger.error(diff.summary())

        assert not diff.has_changes(), f"Category {category} should have no changes"

    def test_model_by_model_comparison(
        self,
        comparator: ModelReferenceComparator,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
    ) -> None:
        """Test individual model comparisons to identify problematic models.

        Args:
            comparator: The comparator instance.
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        data1 = fetch_primary_category_data(api_url=primary_api_url, category=category)
        data2 = fetch_primary_category_data(api_url=primary_api_url, category=category)

        assert set(data1.keys()) == set(data2.keys()), "Model name sets should be identical"

        models_with_differences = []

        for model_name in data1:
            model1 = data1[model_name]
            model2 = data2[model_name]

            if model1 != model2:
                models_with_differences.append(model_name)
                logger.error(f"Model '{model_name}' differs between fetches:")

                differences = deep_compare_models(model1, model2, model_name)
                for difference in differences:
                    logger.error(f"  {difference}")

                norm1 = normalize_for_comparison(model1)
                norm2 = normalize_for_comparison(model2)

                if norm1 == norm2:
                    logger.warning("  -> Models are identical after normalization (field ordering issue)")
                else:
                    logger.error("  -> Models differ even after normalization (actual content difference)")

        if models_with_differences:
            logger.error(f"Total models with differences: {len(models_with_differences)}")
            logger.error(f"Models: {models_with_differences[:10]}")

        assert (
            len(models_with_differences) == 0
        ), f"Found {len(models_with_differences)} models with differences when comparing identical data"

    def test_normalized_comparison_matches(
        self,
        comparator: ModelReferenceComparator,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
    ) -> None:
        """Test that JSON normalization produces identical results.

        This tests whether the differences (if any) are due to serialization
        ordering rather than actual data differences.

        Args:
            comparator: The comparator instance.
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        data1 = fetch_primary_category_data(api_url=primary_api_url, category=category)
        data2 = fetch_primary_category_data(api_url=primary_api_url, category=category)

        norm1 = normalize_for_comparison(data1)
        norm2 = normalize_for_comparison(data2)

        if norm1 != norm2:
            logger.error("Normalized data differs!")
            logger.error(f"Length: {len(norm1)} vs {len(norm2)}")

            import difflib

            diff_lines = list(difflib.unified_diff(norm1.splitlines(), norm2.splitlines(), lineterm=""))
            logger.error("Diff (first 50 lines):\n" + "\n".join(diff_lines[:50]))

        assert norm1 == norm2, "Normalized JSON representations should be identical"

    def test_fetch_consistency_multiple_requests(
        self,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
    ) -> None:
        """Test that multiple fetches of the same category return identical data.

        This verifies that the PRIMARY API is stable and not randomly
        changing data between requests.

        Args:
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        fetches = [fetch_primary_category_data(api_url=primary_api_url, category=category) for _ in range(3)]

        for i in range(1, len(fetches)):
            norm_first = normalize_for_comparison(fetches[0])
            norm_current = normalize_for_comparison(fetches[i])

            assert (
                norm_first == norm_current
            ), f"Fetch {i} differs from first fetch - API may be unstable or data is changing"

    def test_v1_v2_api_comparison(
        self,
        comparator: ModelReferenceComparator,
        primary_api_url: str,
        skip_if_primary_unavailable: None,
    ) -> None:
        """Test comparing v1 API data against v2 API data (if available).

        This helps identify if differences are due to API version differences.
        The v2 API might include additional metadata fields that v1 doesn't have.

        Args:
            comparator: The comparator instance.
            primary_api_url: PRIMARY API URL.
            skip_if_primary_unavailable: Fixture that skips if PRIMARY is down.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        v1_data = fetch_primary_category_data(api_url=primary_api_url, category=category)

        try:
            endpoint_v2 = f"{primary_api_url.rstrip('/')}/model_references/v2/{category}"
            with httpx.Client(timeout=10.0) as client:
                response = client.get(endpoint_v2)
                response.raise_for_status()
                v2_data: dict[str, dict[str, Any]] = response.json()

            logger.info(f"Comparing v1 ({len(v1_data)} models) vs v2 ({len(v2_data)} models)")

            diff = comparator.compare_categories(
                category=category,
                primary_data=v1_data,
                github_data=v2_data,
            )

            if diff.has_changes():
                logger.warning(f"v1 vs v2 API differences detected: {diff.total_changes()} changes")
                logger.info(f"Summary:\n{diff.summary()}")

                if diff.modified_models:
                    sample_model = next(iter(diff.modified_models.keys()))
                    logger.info(f"\nSample model '{sample_model}' differences:")
                    differences = deep_compare_models(v1_data[sample_model], v2_data[sample_model], sample_model)
                    for difference in differences[:10]:
                        logger.info(f"  {difference}")

                logger.warning(
                    "Note: v1 and v2 API formats may legitimately differ in metadata fields. "
                    "This test helps identify such differences."
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                pytest.skip("v2 API not available (may be in legacy canonical format mode)")
            pytest.skip(f"v2 API not available: {e}")
        except httpx.HTTPError as e:
            pytest.skip(f"v2 API not available: {e}")

    def test_empty_vs_populated_category(
        self,
        comparator: ModelReferenceComparator,
    ) -> None:
        """Test comparing empty category against populated one.

        This helps verify that the comparator correctly handles edge cases
        where one data source has models and the other doesn't.

        Args:
            comparator: The comparator instance.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        populated_data = {
            "test_model": {
                "name": "test_model",
                "description": "Test",
            }
        }
        empty_data: dict[str, dict[str, Any]] = {}

        diff = comparator.compare_categories(
            category=category,
            primary_data=populated_data,
            github_data=empty_data,
        )

        assert diff.has_changes()
        assert len(diff.added_models) == 1
        assert "test_model" in diff.added_models
        assert len(diff.removed_models) == 0

        diff_reversed = comparator.compare_categories(
            category=category,
            primary_data=empty_data,
            github_data=populated_data,
        )

        assert diff_reversed.has_changes()
        assert len(diff_reversed.added_models) == 0
        assert len(diff_reversed.removed_models) == 1
        assert "test_model" in diff_reversed.removed_models

    def test_field_ordering_differences(
        self,
        comparator: ModelReferenceComparator,
    ) -> None:
        """Test that field ordering doesn't cause spurious differences.

        This is a common source of false positives - if two dictionaries
        have the same content but different key ordering, they should
        still be considered equal.

        Args:
            comparator: The comparator instance.
        """
        category = MODEL_REFERENCE_CATEGORY.image_generation

        data1 = {
            "model1": {
                "name": "model1",
                "description": "Test",
                "baseline": "stable_diffusion_1",
                "nsfw": False,
            }
        }

        data2 = {
            "model1": {
                "nsfw": False,
                "baseline": "stable_diffusion_1",
                "description": "Test",
                "name": "model1",
            }
        }

        diff = comparator.compare_categories(
            category=category,
            primary_data=data1,
            github_data=data2,
        )

        if diff.has_changes():
            logger.error("Field ordering caused false positive!")
            logger.error(f"Summary:\n{diff.summary()}")
            if diff.modified_models:
                logger.error("Modified models:")
                for model_name in diff.modified_models:
                    differences = deep_compare_models(data1[model_name], data2[model_name], model_name)
                    for difference in differences:
                        logger.error(f"  {difference}")

        assert not diff.has_changes(), "Field ordering should not cause differences"
