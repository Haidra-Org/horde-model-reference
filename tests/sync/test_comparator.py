"""Tests for the ModelReferenceComparator."""

from __future__ import annotations

from typing import Any

import pytest

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.sync import ModelReferenceComparator


class TestModelReferenceComparator:
    """Test suite for ModelReferenceComparator."""

    @pytest.fixture
    def comparator(self) -> ModelReferenceComparator:
        """Create a comparator instance for testing."""
        return ModelReferenceComparator()

    @pytest.fixture
    def sample_primary_data(self) -> dict[str, dict[str, Any]]:
        """Sample PRIMARY data for testing."""
        return {
            "model1": {
                "name": "model1",
                "description": "Test model 1",
                "baseline": "stable_diffusion_1",
                "config": {
                    "download": [
                        {
                            "file_name": "model1.ckpt",
                            "file_url": "https://example.com/model1.ckpt",
                            "sha256sum": "abc123",
                        }
                    ]
                },
            },
            "model2": {
                "name": "model2",
                "description": "Test model 2",
                "baseline": "stable_diffusion_xl",
                "config": {
                    "download": [
                        {
                            "file_name": "model2.safetensors",
                            "file_url": "https://example.com/model2.safetensors",
                            "sha256sum": "def456",
                        }
                    ]
                },
            },
            "model3": {
                "name": "model3",
                "description": "New model in PRIMARY",
                "baseline": "flux_1",
                "config": {"download": []},
            },
        }

    @pytest.fixture
    def sample_github_data(self) -> dict[str, dict[str, Any]]:
        """Sample GitHub data for testing."""
        return {
            "model1": {
                "name": "model1",
                "description": "Test model 1",
                "baseline": "stable_diffusion_1",
                "config": {
                    "download": [
                        {
                            "file_name": "model1.ckpt",
                            "file_url": "https://example.com/model1.ckpt",
                            "sha256sum": "abc123",
                        }
                    ]
                },
            },
            "model2": {
                "name": "model2",
                "description": "Old description",
                "baseline": "stable_diffusion_xl",
                "config": {
                    "download": [
                        {
                            "file_name": "model2.safetensors",
                            "file_url": "https://old-url.com/model2.safetensors",
                            "sha256sum": "def456",
                        }
                    ]
                },
            },
            "model_removed": {
                "name": "model_removed",
                "description": "This model was removed from PRIMARY",
                "baseline": "stable_diffusion_1",
                "config": {"download": []},
            },
        }

    def test_no_changes(
        self,
        comparator: ModelReferenceComparator,
    ) -> None:
        """Test comparison when data is identical."""
        data = {
            "model1": {
                "name": "model1",
                "description": "Test",
            }
        }

        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=data,
            github_data=data,
        )

        assert not diff.has_changes()
        assert len(diff.added_models) == 0
        assert len(diff.removed_models) == 0
        assert len(diff.modified_models) == 0
        assert diff.total_changes() == 0

    def test_added_models(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test detection of added models."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data=sample_github_data,
        )

        assert diff.has_changes()
        assert "model3" in diff.added_models
        assert len(diff.added_models) == 1

    def test_removed_models(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test detection of removed models."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data=sample_github_data,
        )

        assert diff.has_changes()
        assert "model_removed" in diff.removed_models
        assert len(diff.removed_models) == 1

    def test_modified_models(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test detection of modified models."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data=sample_github_data,
        )

        assert diff.has_changes()
        assert "model2" in diff.modified_models
        assert diff.modified_models["model2"]["description"] == "Test model 2"
        assert (
            diff.modified_models["model2"]["config"]["download"][0]["file_url"]
            == "https://example.com/model2.safetensors"
        )

    def test_diff_summary(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test diff summary generation."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data=sample_github_data,
        )

        summary = diff.summary()

        assert "image_generation" in summary
        assert "Added:" in summary
        assert "Removed:" in summary
        assert "Modified:" in summary
        assert "model3" in summary
        assert "model_removed" in summary
        assert "model2" in summary

    def test_total_changes(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test total changes calculation."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data=sample_github_data,
        )

        assert diff.total_changes() == 3

    def test_empty_data(
        self,
        comparator: ModelReferenceComparator,
    ) -> None:
        """Test comparison with empty datasets."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data={},
            github_data={},
        )

        assert not diff.has_changes()
        assert diff.total_changes() == 0

    def test_all_new_models(
        self,
        comparator: ModelReferenceComparator,
        sample_primary_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test comparison when GitHub is empty (all models are new)."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data=sample_primary_data,
            github_data={},
        )

        assert diff.has_changes()
        assert len(diff.added_models) == len(sample_primary_data)
        assert len(diff.removed_models) == 0
        assert len(diff.modified_models) == 0

    def test_all_removed_models(
        self,
        comparator: ModelReferenceComparator,
        sample_github_data: dict[str, dict[str, Any]],
    ) -> None:
        """Test comparison when PRIMARY is empty (all models were removed)."""
        diff = comparator.compare_categories(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            primary_data={},
            github_data=sample_github_data,
        )

        assert diff.has_changes()
        assert len(diff.added_models) == 0
        assert len(diff.removed_models) == len(sample_github_data)
        assert len(diff.modified_models) == 0
