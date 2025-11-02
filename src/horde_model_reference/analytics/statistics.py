"""Category-level statistics calculation for model reference data.

Provides functions to compute aggregate statistics over collections of model records.
Statistics include model counts, baseline distributions, download information, and tag/style distributions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference.analytics.constants import PARAMETER_BUCKETS, TOP_STYLES_LIMIT, TOP_TAGS_LIMIT
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class ParameterBucketStats(BaseModel):
    """Statistics about parameter count buckets for text generation models.

    Used to group models by parameter size ranges for analysis.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    bucket_label: str
    """Human-readable bucket label (e.g., '< 3B', '3B-6B', '70B-100B', '100B+')."""
    min_params: int | None = None
    """Minimum parameter count for this bucket (None for open-ended ranges)."""
    max_params: int | None = None
    """Maximum parameter count for this bucket (None for open-ended ranges like '100B+')."""
    count: int = Field(ge=0)
    """Number of models in this parameter bucket."""
    percentage: float = Field(ge=0.0, le=100.0)
    """Percentage of total models in this bucket."""


class BaselineStats(BaseModel):
    """Statistics about model baselines in a category."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    baseline: str
    """The baseline name (e.g., 'stable_diffusion_xl', 'stable_diffusion_1')."""
    count: int = Field(ge=0)
    """Number of models using this baseline."""
    percentage: float = Field(ge=0.0, le=100.0)
    """Percentage of total models using this baseline."""


class DownloadStats(BaseModel):
    """Statistics about model downloads."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    total_models_with_downloads: int = Field(ge=0)
    """Number of models that have at least one download entry."""
    total_download_entries: int = Field(ge=0)
    """Total number of download entries across all models."""
    total_size_bytes: int = Field(ge=0)
    """Total size in bytes of all downloads (for models with size information)."""
    models_with_size_info: int = Field(ge=0)
    """Number of models that have size_on_disk_bytes information."""
    average_size_bytes: float = Field(ge=0.0)
    """Average size in bytes for models with size information."""
    hosts: dict[str, int] = Field(default_factory=dict)
    """Distribution of download hosts (domain -> count)."""


class TagStats(BaseModel):
    """Statistics about model tags or styles."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tag: str
    """The tag or style name."""
    count: int = Field(ge=0)
    """Number of models with this tag/style."""
    percentage: float = Field(ge=0.0, le=100.0)
    """Percentage of total models with this tag/style."""


class CategoryStatistics(BaseModel):
    """Comprehensive statistics for a model reference category.

    Contains aggregate metrics, distributions, and metadata about all models in a category.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    category: MODEL_REFERENCE_CATEGORY
    """The category these statistics describe."""
    total_models: int = Field(ge=0)
    """Total number of models in the category (before pagination)."""
    returned_models: int = Field(ge=0, default=0)
    """Number of models returned in this response (after pagination)."""
    offset: int = Field(ge=0, default=0)
    """Starting index for pagination (0 if not paginated)."""
    limit: int | None = None
    """Maximum number of models per page (None if not paginated)."""
    nsfw_count: int = Field(ge=0, default=0)
    """Number of NSFW models (if applicable to category)."""

    # sfw_count: int = Field(ge=0, default=0)
    @property
    def sfw_count(self) -> int:
        """Number of SFW models (if applicable to category)."""
        return self.total_models - self.nsfw_count

    baseline_distribution: dict[str, BaselineStats] = Field(default_factory=dict)
    """Distribution of models across different baselines."""

    download_stats: DownloadStats | None = None
    """Aggregate download statistics."""

    top_tags: list[TagStats] = Field(default_factory=list)
    """Top tags by frequency (for categories with tags)."""
    top_styles: list[TagStats] = Field(default_factory=list)
    """Top styles by frequency (for categories with styles)."""

    parameter_buckets: list[ParameterBucketStats] = Field(default_factory=list)
    """Distribution of text models by parameter count buckets (text generation only)."""
    models_without_param_info: int = Field(ge=0, default=0)
    """Number of models without parameter count information (text generation only)."""

    models_with_trigger_words: int = Field(ge=0, default=0)
    """Number of models with trigger words (image generation only)."""
    models_with_inpainting: int = Field(ge=0, default=0)
    """Number of inpainting models (image generation only)."""
    models_with_requirements: int = Field(ge=0, default=0)
    """Number of models with special requirements (image generation only)."""
    models_with_showcases: int = Field(ge=0, default=0)
    """Number of models with showcase links."""

    computed_at: int
    """Unix timestamp when these statistics were computed."""


def calculate_category_statistics(
    models: dict[str, Any],
    category: MODEL_REFERENCE_CATEGORY,
) -> CategoryStatistics:
    """Calculate comprehensive statistics for a category of models.

    Args:
        models: Dictionary mapping model names to model data dictionaries (raw JSON format).
        category: The category these models belong to.

    Returns:
        CategoryStatistics containing all computed metrics.

    Example:
        >>> models = manager.get_raw_model_reference_json(MODEL_REFERENCE_CATEGORY.image_generation)
        >>> stats = calculate_category_statistics(models, MODEL_REFERENCE_CATEGORY.image_generation)
        >>> print(f"Total models: {stats.total_models}")
        >>> print(f"NSFW: {stats.nsfw_count}, SFW: {stats.sfw_count}")
    """
    import time
    from urllib.parse import urlparse

    logger.debug(f"Calculating statistics for category {category} with {len(models)} models")

    total_models = len(models)
    nsfw_count = 0

    baseline_counts: dict[str, int] = defaultdict(int)

    download_count = 0
    download_entries_count = 0
    total_size_bytes = 0
    models_with_size = 0
    host_counts: dict[str, int] = defaultdict(int)

    tag_counts: dict[str, int] = defaultdict(int)
    style_counts: dict[str, int] = defaultdict(int)

    parameter_counts: list[int] = []
    models_without_params = 0

    models_with_triggers = 0
    models_with_inpainting = 0
    models_with_requirements = 0
    models_with_showcases = 0

    for model_name, model_data in models.items():
        if not isinstance(model_data, dict):
            logger.warning(f"Skipping model {model_name}: invalid data type {type(model_data)}")
            continue

        # NSFW statistics (applicable to image_generation and text_generation)
        if model_data.get("nsfw"):
            nsfw_count += 1

        # Baseline distribution
        baseline = model_data.get("baseline")
        if baseline:
            baseline_counts[str(baseline)] += 1

        # Download statistics
        config = model_data.get("config", {})
        if isinstance(config, dict):
            downloads = config.get("download", [])
            if isinstance(downloads, list) and len(downloads) > 0:
                download_count += 1
                download_entries_count += len(downloads)

                for download in downloads:
                    if isinstance(download, dict):
                        url = download.get("file_url", "")
                        if url:
                            parsed = urlparse(url)
                            if parsed.netloc:
                                host_counts[parsed.netloc] += 1

        # Size information
        size_on_disk = model_data.get("size_on_disk_bytes")
        if size_on_disk and isinstance(size_on_disk, (int, float)) and size_on_disk > 0:
            models_with_size += 1
            total_size_bytes += int(size_on_disk)

        # Tag distribution (image_generation, text_generation)
        tags = model_data.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if tag:
                    tag_counts[str(tag)] += 1

        # Style distribution (image_generation)
        style = model_data.get("style")
        if style:
            style_counts[str(style)] += 1

        # Text-generation specific metrics
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            params_count = model_data.get("parameters_count")
            if params_count and isinstance(params_count, (int, float)) and params_count > 0:
                parameter_counts.append(int(params_count))
            else:
                models_without_params += 1

        # Image-generation specific metrics
        if category == MODEL_REFERENCE_CATEGORY.image_generation:
            trigger = model_data.get("trigger")
            if isinstance(trigger, list) and len(trigger) > 0:
                models_with_triggers += 1

            if model_data.get("inpainting"):
                models_with_inpainting += 1

            requirements = model_data.get("requirements")
            if requirements and isinstance(requirements, dict) and len(requirements) > 0:
                models_with_requirements += 1

        # Showcase links
        showcases = model_data.get("showcases")
        if isinstance(showcases, list) and len(showcases) > 0:
            models_with_showcases += 1

    # Calculate baseline percentages
    baseline_stats: dict[str, BaselineStats] = {}
    if total_models > 0:
        for baseline, count in sorted(baseline_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_models) * 100.0
            baseline_stats[baseline] = BaselineStats(
                baseline=baseline,
                count=count,
                percentage=round(percentage, 2),
            )

    # Calculate download statistics
    download_statistics = DownloadStats(
        total_models_with_downloads=download_count,
        total_download_entries=download_entries_count,
        total_size_bytes=total_size_bytes,
        models_with_size_info=models_with_size,
        average_size_bytes=round(total_size_bytes / models_with_size, 2) if models_with_size > 0 else 0.0,
        hosts=dict(sorted(host_counts.items(), key=lambda x: x[1], reverse=True)),
    )

    # Top tags (limit to top 20)
    top_tags_list = []
    if total_models > 0:
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            percentage = (count / total_models) * 100.0
            top_tags_list.append(
                TagStats(
                    tag=tag,
                    count=count,
                    percentage=round(percentage, 2),
                )
            )

    # Top styles (limit to top 20)
    top_styles_list = []
    if total_models > 0:
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            percentage = (count / total_models) * 100.0
            top_styles_list.append(
                TagStats(
                    tag=style,
                    count=count,
                    percentage=round(percentage, 2),
                )
            )

    # Parameter bucket calculation (text generation only)
    parameter_bucket_stats: list[ParameterBucketStats] = []
    if category == MODEL_REFERENCE_CATEGORY.text_generation and parameter_counts:
        # Use configured parameter buckets from constants
        bucket_counts: dict[str, int] = defaultdict(int)

        for param_count in parameter_counts:
            for min_val, max_val, label in PARAMETER_BUCKETS:
                if max_val == float("inf"):
                    if param_count >= min_val:
                        bucket_counts[label] += 1
                        break
                elif min_val <= param_count < max_val:
                    bucket_counts[label] += 1
                    break

        models_with_params = len(parameter_counts)
        if total_models > 0:
            for min_val, max_val, label in PARAMETER_BUCKETS:
                count = bucket_counts.get(label, 0)
                if count > 0:
                    percentage = (count / total_models) * 100.0
                    # Convert float('inf') to None for the Pydantic model
                    max_params_value = None if max_val == float("inf") else int(max_val)
                    parameter_bucket_stats.append(
                        ParameterBucketStats(
                            bucket_label=label,
                            min_params=int(min_val),
                            max_params=max_params_value,
                            count=count,
                            percentage=round(percentage, 2),
                        )
                    )

    logger.debug(f"Statistics calculated: {total_models} models, {len(baseline_stats)} baselines")

    return CategoryStatistics(
        category=category,
        total_models=total_models,
        returned_models=total_models,  # No pagination by default
        offset=0,
        limit=None,
        nsfw_count=nsfw_count,
        baseline_distribution=baseline_stats,
        download_stats=download_statistics,
        top_tags=top_tags_list,
        top_styles=top_styles_list,
        parameter_buckets=parameter_bucket_stats,
        models_without_param_info=models_without_params,
        models_with_trigger_words=models_with_triggers,
        models_with_inpainting=models_with_inpainting,
        models_with_requirements=models_with_requirements,
        models_with_showcases=models_with_showcases,
        computed_at=int(time.time()),
    )
