"""Unit tests for the statistics module."""

from __future__ import annotations

from typing import Any

import pytest

from horde_model_reference.analytics.statistics import calculate_category_statistics
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)


def _create_image_generation_model(
    name: str,
    baseline: str = "stable_diffusion_xl",
    nsfw: bool = False,
    tags: list[str] | None = None,
    style: str | None = None,
    trigger: list[str] | None = None,
    inpainting: bool = False,
    showcases: list[str] | None = None,
    downloads: list[dict[str, str]] | None = None,
    size_on_disk_bytes: int | None = None,
) -> ImageGenerationModelRecord:
    """Create an ImageGenerationModelRecord for testing."""
    download_records = []
    if downloads:
        download_records = [
            DownloadRecord(
                file_name=d.get("file_name", ""),
                file_url=d.get("file_url", ""),
                sha256sum=d.get("sha256sum", "abc123"),
            )
            for d in downloads
        ]

    return ImageGenerationModelRecord(
        name=name,
        baseline=baseline,
        nsfw=nsfw,
        tags=tags or [],
        style=style,
        trigger=trigger or [],
        inpainting=inpainting,
        showcases=showcases or [],
        config=GenericModelRecordConfig(download=download_records),
        size_on_disk_bytes=size_on_disk_bytes,
    )


def _create_text_generation_model(
    name: str,
    baseline: str | None = None,
    nsfw: bool = False,
    tags: list[str] | None = None,
    parameters: int = 7000000000,
) -> TextGenerationModelRecord:
    """Create a TextGenerationModelRecord for testing."""
    return TextGenerationModelRecord(
        name=name,
        baseline=baseline,
        nsfw=nsfw,
        tags=tags or [],
        parameters=parameters,
        config=GenericModelRecordConfig(download=[]),
    )


class TestCategoryStatistics:
    """Tests for category statistics calculation."""

    def test_calculate_statistics_empty_dict(self) -> None:
        """Test statistics calculation with an empty model dictionary."""
        stats = calculate_category_statistics({}, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.category == MODEL_REFERENCE_CATEGORY.image_generation
        assert stats.total_models == 0
        assert stats.nsfw_count == 0
        assert stats.sfw_count == 0
        assert len(stats.baseline_distribution) == 0
        assert stats.download_stats is not None
        assert stats.download_stats.total_models_with_downloads == 0
        assert stats.download_stats.total_download_entries == 0
        assert stats.download_stats.total_size_bytes == 0
        assert stats.download_stats.average_size_bytes == 0.0
        assert len(stats.top_tags) == 0
        assert len(stats.top_styles) == 0
        assert stats.models_with_trigger_words == 0
        assert stats.models_with_inpainting == 0
        assert stats.models_with_showcases == 0

    def test_calculate_statistics_single_model(self) -> None:
        """Test statistics calculation with a single image generation model."""
        model = _create_image_generation_model(
            name="test_model",
            baseline="stable_diffusion_xl",
            nsfw=False,
            tags=["anime", "character"],
            style="anime",
            trigger=["test_trigger"],
            inpainting=False,
            showcases=["https://example.com/showcase1"],
            downloads=[
                {
                    "file_name": "test.safetensors",
                    "file_url": "https://huggingface.co/test/model.safetensors",
                    "sha256sum": "abc123",
                }
            ],
            size_on_disk_bytes=7000000000,
        )

        models: dict[str, Any] = {"test_model": model.model_dump()}

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.category == MODEL_REFERENCE_CATEGORY.image_generation
        assert stats.total_models == 1
        assert stats.nsfw_count == 0
        assert stats.sfw_count == 1

        assert len(stats.baseline_distribution) == 1
        baseline_stats = stats.baseline_distribution.get("stable_diffusion_xl")
        assert baseline_stats is not None
        assert baseline_stats.count == 1
        assert baseline_stats.percentage == 100.0

        assert stats.download_stats is not None
        assert stats.download_stats.total_models_with_downloads == 1
        assert stats.download_stats.total_download_entries == 1
        assert stats.download_stats.total_size_bytes == 7000000000
        assert stats.download_stats.models_with_size_info == 1
        assert stats.download_stats.average_size_bytes == 7000000000.0
        assert "huggingface.co" in stats.download_stats.hosts
        assert stats.download_stats.hosts["huggingface.co"] == 1

        assert len(stats.top_tags) == 2
        tag_names = [tag.tag for tag in stats.top_tags]
        assert "anime" in tag_names
        assert "character" in tag_names

        assert len(stats.top_styles) == 1
        assert stats.top_styles[0].tag == "anime"
        assert stats.top_styles[0].count == 1
        assert stats.top_styles[0].percentage == 100.0

        assert stats.models_with_trigger_words == 1
        assert stats.models_with_inpainting == 0
        assert stats.models_with_showcases == 1

    def test_calculate_statistics_multiple_models(self) -> None:
        """Test statistics calculation with multiple models."""
        models = {
            "model1": {
                "name": "model1",
                "baseline": "stable_diffusion_xl",
                "nsfw": True,
                "tags": ["anime"],
                "style": "anime",
                "trigger": [],
                "inpainting": False,
                "showcases": [],
                "config": {"download": []},
                "size_on_disk_bytes": 5000000000,
            },
            "model2": {
                "name": "model2",
                "baseline": "stable_diffusion_1",
                "nsfw": False,
                "tags": ["realistic", "portrait"],
                "style": "realistic",
                "trigger": ["portrait"],
                "inpainting": True,
                "showcases": ["https://example.com/showcase"],
                "config": {
                    "download": [
                        {
                            "file_name": "model2.safetensors",
                            "file_url": "https://civitai.com/model2.safetensors",
                            "sha256sum": "def456",
                        }
                    ]
                },
                "size_on_disk_bytes": 3000000000,
            },
            "model3": {
                "name": "model3",
                "baseline": "stable_diffusion_xl",
                "nsfw": False,
                "tags": ["anime", "realistic"],
                "style": "hybrid",
                "trigger": [],
                "inpainting": False,
                "showcases": [],
                "config": {"download": []},
            },
        }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.total_models == 3
        assert stats.nsfw_count == 1
        assert stats.sfw_count == 2

        assert len(stats.baseline_distribution) == 2
        baseline_stats_xl = stats.baseline_distribution.get("stable_diffusion_xl")
        assert baseline_stats_xl is not None
        assert baseline_stats_xl.count == 2
        assert baseline_stats_xl.percentage == pytest.approx(66.67, abs=0.01)
        baseline_stats_1 = stats.baseline_distribution.get("stable_diffusion_1")
        assert baseline_stats_1 is not None
        assert baseline_stats_1.count == 1
        assert baseline_stats_1.percentage == pytest.approx(33.33, abs=0.01)

        assert stats.download_stats is not None
        assert stats.download_stats.total_models_with_downloads == 1
        assert stats.download_stats.total_download_entries == 1
        assert stats.download_stats.models_with_size_info == 2
        assert stats.download_stats.total_size_bytes == 8000000000
        assert stats.download_stats.average_size_bytes == 4000000000.0

        assert len(stats.top_tags) >= 2
        tag_counts = {tag.tag: tag.count for tag in stats.top_tags}
        assert tag_counts["anime"] == 2
        assert tag_counts["realistic"] == 2

        assert stats.models_with_trigger_words == 1
        assert stats.models_with_inpainting == 1
        assert stats.models_with_showcases == 1

    def test_calculate_statistics_text_generation(self) -> None:
        """Test statistics calculation for text generation models."""
        models = {
            "llama": {
                "name": "llama",
                "baseline": "llama-2",
                "nsfw": False,
                "tags": ["instruction", "chat"],
                "parameters": 7000000000,
                "config": {"download": []},
            },
            "mistral": {
                "name": "mistral",
                "baseline": "mistral",
                "nsfw": False,
                "tags": ["chat"],
                "parameters": 7000000000,
                "config": {"download": []},
            },
        }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.text_generation)

        assert stats.total_models == 2
        assert stats.nsfw_count == 0
        assert stats.sfw_count == 2
        assert len(stats.baseline_distribution) == 2

        tag_counts = {tag.tag: tag.count for tag in stats.top_tags}
        assert tag_counts["chat"] == 2
        assert tag_counts["instruction"] == 1

        assert stats.models_with_trigger_words == 0
        assert stats.models_with_inpainting == 0

    def test_calculate_statistics_handles_missing_fields(self) -> None:
        """Test statistics calculation handles models with missing fields gracefully."""
        models = {
            "minimal_model": {
                "name": "minimal_model",
            }
        }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.total_models == 1
        assert stats.nsfw_count == 0
        # Since nsfw field is missing, nsfw_count=0, so sfw_count = total_models - nsfw_count = 1
        assert stats.sfw_count == 1
        assert len(stats.baseline_distribution) == 0
        assert stats.download_stats is not None
        assert stats.download_stats.total_models_with_downloads == 0

    def test_calculate_statistics_multiple_download_hosts(self) -> None:
        """Test statistics calculation tracks multiple download hosts."""
        models = {
            "model1": {
                "name": "model1",
                "baseline": "stable_diffusion_1",
                "config": {
                    "download": [
                        {
                            "file_name": "model1.safetensors",
                            "file_url": "https://huggingface.co/model1.safetensors",
                            "sha256sum": "abc",
                        },
                        {
                            "file_name": "model1_vae.safetensors",
                            "file_url": "https://huggingface.co/vae.safetensors",
                            "sha256sum": "def",
                        },
                    ]
                },
            },
            "model2": {
                "name": "model2",
                "baseline": "stable_diffusion_1",
                "config": {
                    "download": [
                        {
                            "file_name": "model2.safetensors",
                            "file_url": "https://civitai.com/model2.safetensors",
                            "sha256sum": "ghi",
                        }
                    ]
                },
            },
        }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.download_stats is not None
        assert stats.download_stats.total_models_with_downloads == 2
        assert stats.download_stats.total_download_entries == 3
        assert stats.download_stats.hosts["huggingface.co"] == 2
        assert stats.download_stats.hosts["civitai.com"] == 1

    def test_calculate_statistics_top_tags_limit(self) -> None:
        """Test that top_tags is limited to 20 entries."""
        models = {}
        for i in range(30):
            models[f"model_{i}"] = {
                "name": f"model_{i}",
                "baseline": "stable_diffusion_1",
                "tags": [f"tag_{i}"],
            }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.total_models == 30
        assert len(stats.top_tags) == 20

    def test_calculate_statistics_percentage_calculations(self) -> None:
        """Test that percentage calculations are correct."""
        models = {
            f"model_{i}": {
                "name": f"model_{i}",
                "baseline": "stable_diffusion_xl" if i % 2 == 0 else "stable_diffusion_1",
                "nsfw": i % 3 == 0,
                "tags": ["test_tag"] if i % 5 == 0 else [],
            }
            for i in range(100)
        }

        stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)

        assert stats.total_models == 100

        baseline_stats_xl = stats.baseline_distribution.get("stable_diffusion_xl")
        assert baseline_stats_xl is not None
        assert baseline_stats_xl.percentage == 50.0
        baseline_stats_1 = stats.baseline_distribution.get("stable_diffusion_1")
        assert baseline_stats_1 is not None
        assert baseline_stats_1.percentage == 50.0
        assert stats.nsfw_count == 34
        assert stats.sfw_count == 66

        if len(stats.top_tags) > 0:
            test_tag = next((tag for tag in stats.top_tags if tag.tag == "test_tag"), None)
            assert test_tag is not None
            assert test_tag.count == 20
            assert test_tag.percentage == 20.0
