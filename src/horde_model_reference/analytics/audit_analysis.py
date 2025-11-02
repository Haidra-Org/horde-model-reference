"""Audit analysis for model references.

Provides functions to analyze models for deletion risk and audit-worthiness.
Identifies issues like missing downloads, non-preferred hosts, low usage, etc.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from horde_model_reference import horde_model_reference_settings
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY


class UsageTrend(BaseModel):
    """Usage trend metrics comparing different time periods.

    Ratios help identify model momentum and activity patterns.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    day_to_month_ratio: float | None = None
    """Ratio of day usage to month usage (day/month). None if month usage is zero."""
    month_to_total_ratio: float | None = None
    """Ratio of month usage to total usage (month/total). None if total usage is zero."""


class DeletionRiskFlags(BaseModel):
    """Flags indicating potential reasons for model deletion.

    Each flag represents a specific issue that might warrant model removal.
    Aligned with frontend expectations for consistent presentation.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    zero_usage_day: bool = False
    """Model has no usage in the past day."""
    zero_usage_month: bool = False
    """Model has no usage in the past month."""
    zero_usage_total: bool = False
    """Model has no usage in total (all time)."""
    no_active_workers: bool = False
    """Model has zero active workers (from Horde API data)."""
    has_multiple_hosts: bool = False
    """Model downloads are distributed across multiple file hosts."""
    has_non_preferred_host: bool = False
    """Model is hosted on non-preferred file hosts (non-HuggingFace)."""
    has_unknown_host: bool = False
    """Model has download URLs with unknown or unparseable hosts."""
    no_download_urls: bool = False
    """Model has no download URLs, empty download list, or invalid URLs."""
    missing_description: bool = False
    """Model lacks a description field."""
    missing_baseline: bool = False
    """Model lacks baseline information (for applicable categories)."""
    low_usage: bool = False
    """Model has very low usage (threshold defined by LOW_USAGE_THRESHOLD constant)."""

    def any_flags(self) -> bool:
        """Check if any deletion risk flags are set.

        Returns:
            True if at least one flag is set, False otherwise.
        """
        return any(
            [
                self.zero_usage_day,
                self.zero_usage_month,
                self.zero_usage_total,
                self.no_active_workers,
                self.has_multiple_hosts,
                self.has_non_preferred_host,
                self.has_unknown_host,
                self.no_download_urls,
                self.missing_description,
                self.missing_baseline,
                self.low_usage,
            ]
        )

    def flag_count(self) -> int:
        """Count the number of flags set.

        Returns:
            Number of deletion risk flags that are True.
        """
        return sum(
            [
                self.zero_usage_day,
                self.zero_usage_month,
                self.zero_usage_total,
                self.no_active_workers,
                self.has_multiple_hosts,
                self.has_non_preferred_host,
                self.has_unknown_host,
                self.no_download_urls,
                self.missing_description,
                self.missing_baseline,
                self.low_usage,
            ]
        )


class ModelAuditInfo(BaseModel):
    """Audit information for a single model.

    Contains model metadata along with deletion risk assessment and usage statistics.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """The model name."""
    category: MODEL_REFERENCE_CATEGORY
    """The category this model belongs to."""

    deletion_risk_flags: DeletionRiskFlags
    """Flags indicating potential deletion risks."""
    at_risk: bool
    """True if model has any deletion risk flags."""
    risk_score: int = Field(ge=0)
    """Total number of deletion risk flags (0 = no risk)."""

    worker_count: int = Field(ge=0, default=0)
    """Number of active workers serving this model."""
    usage_day: int = Field(ge=0, default=0)
    """Usage count for the past day."""
    usage_month: int = Field(ge=0, default=0)
    """Usage count for the past month."""
    usage_total: int = Field(ge=0, default=0)
    """Total usage count (all time)."""
    usage_hour: int | None = None
    """Usage count for the past hour (if available)."""
    usage_minute: int | None = None
    """Usage count for the past minute (if available)."""
    usage_percentage_of_category: float = Field(ge=0.0, le=100.0, default=0.0)
    """Percentage of category's total monthly usage."""
    usage_trend: UsageTrend
    """Usage trend comparing different time periods."""

    cost_benefit_score: float | None = None
    """Cost-benefit metric: usage per GB (usage_month / size_gb). Only for models with size info."""
    size_gb: float | None = None
    """Model size in gigabytes (if available)."""

    baseline: str | None = None
    """Model baseline (if applicable)."""
    nsfw: bool | None = None
    """Whether model is NSFW (if applicable)."""
    has_description: bool = False
    """Whether model has a description."""
    download_count: int = Field(ge=0, default=0)
    """Number of download entries."""
    download_hosts: list[str] = Field(default_factory=list)
    """List of download host domains."""

    @property
    def flag_count(self) -> int:
        """Count of deletion risk flags set.

        Returns:
            Number of flags that are True.
        """
        return self.deletion_risk_flags.flag_count()

    @property
    def is_critical(self) -> bool:
        """Determine if model is in critical state.

        Critical = zero month usage AND no active workers.

        Returns:
            True if model meets critical criteria.
        """
        return self.deletion_risk_flags.zero_usage_month and self.deletion_risk_flags.no_active_workers

    @property
    def has_warning(self) -> bool:
        """Determine if model has warning-level issues.

        Warnings include host-related issues or download problems.

        Returns:
            True if model has warning-level flags.
        """
        return (
            self.deletion_risk_flags.has_multiple_hosts
            or self.deletion_risk_flags.has_non_preferred_host
            or self.deletion_risk_flags.has_unknown_host
            or self.deletion_risk_flags.no_download_urls
        )


class CategoryAuditSummary(BaseModel):
    """Summary statistics for a category audit.

    Aggregates audit information across all models in a category.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    total_models: int = Field(ge=0)
    """Total number of models in the category."""
    models_at_risk: int = Field(ge=0)
    """Number of models with at least one deletion risk flag."""
    models_critical: int = Field(ge=0)
    """Number of models in critical state (zero month usage AND no workers)."""
    models_with_warnings: int = Field(ge=0)
    """Number of models with warning-level issues (host/download problems)."""

    models_with_zero_day_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage in the past day."""
    models_with_zero_month_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage in the past month."""
    models_with_zero_total_usage: int = Field(ge=0, default=0)
    """Number of models with zero usage all-time."""
    models_with_no_active_workers: int = Field(ge=0, default=0)
    """Number of models with no active workers."""
    models_with_no_downloads: int = Field(ge=0, default=0)
    """Number of models without download URLs."""
    models_with_non_preferred_hosts: int = Field(ge=0, default=0)
    """Number of models on non-preferred hosts."""
    models_with_multiple_hosts: int = Field(ge=0, default=0)
    """Number of models with downloads across multiple hosts."""
    models_with_low_usage: int = Field(ge=0, default=0)
    """Number of models with very low usage (< 0.1% of category)."""

    average_risk_score: float = Field(ge=0.0)
    """Average risk score across all models."""
    category_total_month_usage: int = Field(ge=0, default=0)
    """Total monthly usage for the entire category."""


class CategoryAuditResponse(BaseModel):
    """Complete audit response for a category.

    Contains both per-model audit information and aggregate summary.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    category: MODEL_REFERENCE_CATEGORY
    """The category being audited."""
    category_total_month_usage: int = Field(ge=0)
    """Total monthly usage for the entire category."""

    total_count: int = Field(ge=0, default=0)
    """Total number of models in the category (before pagination/filtering)."""
    returned_count: int = Field(ge=0, default=0)
    """Number of models returned in this response (after pagination/filtering)."""
    offset: int = Field(ge=0, default=0)
    """Starting index for pagination (0 if not paginated)."""
    limit: int | None = None
    """Maximum number of models per page (None if not paginated)."""

    models: list[ModelAuditInfo]
    """List of audit information for each model."""
    summary: CategoryAuditSummary
    """Aggregate summary statistics."""


def get_deletion_flags(
    model_data: dict[str, Any],
    category: MODEL_REFERENCE_CATEGORY,
    enriched_data: dict[str, Any] | None = None,
    category_total_usage: int = 0,
) -> DeletionRiskFlags:
    """Analyze a model and determine deletion risk flags.

    Args:
        model_data: Raw model reference data dictionary.
        category: The category this model belongs to.
        enriched_data: Optional enriched data with Horde API stats (worker_count, usage_stats).
        category_total_usage: Total monthly usage for the category (for percentage calculations).

    Returns:
        DeletionRiskFlags with appropriate flags set.
    """
    flags = DeletionRiskFlags()

    # Check for missing or empty downloads
    config = model_data.get("config", {})
    downloads = config.get("download", []) if isinstance(config, dict) else []

    unique_hosts: set[str] = set()
    has_valid_url = False
    has_preferred_host = False

    if not downloads or (isinstance(downloads, list) and len(downloads) == 0):
        flags.no_download_urls = True
    else:
        # Check download URLs and analyze hosts
        if isinstance(downloads, list):
            preferred_hosts = horde_model_reference_settings.preferred_file_hosts

            for download in downloads:
                if isinstance(download, dict):
                    url = download.get("file_url", "")
                    if url:
                        try:
                            parsed = urlparse(url)
                            if parsed.scheme and parsed.netloc:
                                has_valid_url = True
                                unique_hosts.add(parsed.netloc)

                                # Check if this host is preferred
                                if preferred_hosts and any(host in parsed.netloc for host in preferred_hosts):
                                    has_preferred_host = True
                        except Exception:
                            # Failed to parse URL - unknown host
                            flags.has_unknown_host = True

            # Set flags based on analysis
            if not has_valid_url:
                flags.no_download_urls = True

            if len(unique_hosts) > 1:
                flags.has_multiple_hosts = True

            if has_valid_url and not has_preferred_host:
                flags.has_non_preferred_host = True

    # Check for missing description
    description = model_data.get("description")
    if not description or (isinstance(description, str) and len(description.strip()) == 0):
        flags.missing_description = True

    # Check for missing baseline (applicable to image_generation and text_generation)
    if category in [MODEL_REFERENCE_CATEGORY.image_generation, MODEL_REFERENCE_CATEGORY.text_generation]:
        baseline = model_data.get("baseline")
        if not baseline:
            flags.missing_baseline = True

    # Check Horde API data if provided
    if enriched_data:
        # Check for zero workers
        worker_count = enriched_data.get("worker_count", 0)
        if isinstance(worker_count, int) and worker_count == 0:
            flags.no_active_workers = True

        # Check usage statistics
        usage_stats = enriched_data.get("usage_stats", {})
        if isinstance(usage_stats, dict):
            usage_day = usage_stats.get("day", 0)
            usage_month = usage_stats.get("month", 0)
            usage_total = usage_stats.get("total", 0)

            # Check for zero usage across different time periods
            if isinstance(usage_day, int) and usage_day == 0:
                flags.zero_usage_day = True

            if isinstance(usage_month, int) and usage_month == 0:
                flags.zero_usage_month = True

            if isinstance(usage_total, int) and usage_total == 0:
                flags.zero_usage_total = True

            # Low usage (< 0.1% of category total monthly usage)
            if category_total_usage > 0 and isinstance(usage_month, int):
                usage_percentage = (usage_month / category_total_usage) * 100.0
                if usage_percentage < 0.1:
                    flags.low_usage = True

    return flags


def analyze_models_for_audit(
    enriched_models: dict[str, Any],
    category_total_usage: int,
    category: MODEL_REFERENCE_CATEGORY,
) -> list[ModelAuditInfo]:
    """Analyze enriched model data to create audit information.

    Args:
        enriched_models: Dictionary of model names to enriched model data
            (model reference + Horde API data merged).
        category_total_usage: Total monthly usage for the entire category.
        category: The model reference category.

    Returns:
        List of ModelAuditInfo sorted by usage (descending).
    """
    audit_models: list[ModelAuditInfo] = []

    for model_name, model_data in enriched_models.items():
        # Get deletion risk flags
        flags = get_deletion_flags(model_data, category, model_data, category_total_usage)

        # Extract Horde API data
        worker_count = model_data.get("worker_count", 0)
        usage_stats = model_data.get("usage_stats", {})
        usage_day = usage_stats.get("day", 0) if isinstance(usage_stats, dict) else 0
        usage_month = usage_stats.get("month", 0) if isinstance(usage_stats, dict) else 0
        usage_total = usage_stats.get("total", 0) if isinstance(usage_stats, dict) else 0
        usage_hour = usage_stats.get("hour") if isinstance(usage_stats, dict) else None
        usage_minute = usage_stats.get("minute") if isinstance(usage_stats, dict) else None

        # Calculate usage percentage
        usage_percentage = 0.0
        if category_total_usage > 0 and isinstance(usage_month, int):
            usage_percentage = (usage_month / category_total_usage) * 100.0

        # Calculate usage trend ratios
        day_to_month_ratio: float | None = None
        if isinstance(usage_month, int) and usage_month > 0 and isinstance(usage_day, int):
            day_to_month_ratio = usage_day / usage_month

        month_to_total_ratio: float | None = None
        if isinstance(usage_total, int) and usage_total > 0 and isinstance(usage_month, int):
            month_to_total_ratio = usage_month / usage_total

        usage_trend = UsageTrend(
            day_to_month_ratio=day_to_month_ratio,
            month_to_total_ratio=month_to_total_ratio,
        )

        # Extract model metadata
        baseline = model_data.get("baseline")
        nsfw = model_data.get("nsfw")
        description = model_data.get("description")
        has_description = bool(description and isinstance(description, str) and len(description.strip()) > 0)

        # Calculate size in GB and cost-benefit score
        size_bytes = model_data.get("size_on_disk_bytes")
        size_gb: float | None = None
        cost_benefit_score: float | None = None

        if size_bytes and isinstance(size_bytes, (int, float)) and size_bytes > 0:
            size_gb = size_bytes / (1024**3)

            if isinstance(usage_month, int) and size_gb > 0:
                cost_benefit_score = usage_month / size_gb

        # Extract download information
        config = model_data.get("config", {})
        downloads = config.get("download", []) if isinstance(config, dict) else []
        download_count = len(downloads) if isinstance(downloads, list) else 0

        download_hosts = []
        if isinstance(downloads, list):
            for download in downloads:
                if isinstance(download, dict):
                    url = download.get("file_url", "")
                    if url:
                        try:
                            parsed = urlparse(url)
                            if parsed.netloc and parsed.netloc not in download_hosts:
                                download_hosts.append(parsed.netloc)
                        except Exception:
                            pass

        # Create audit info
        audit_info = ModelAuditInfo(
            name=model_name,
            category=category,
            deletion_risk_flags=flags,
            at_risk=flags.any_flags(),
            risk_score=flags.flag_count(),
            worker_count=worker_count if isinstance(worker_count, int) else 0,
            usage_day=usage_day if isinstance(usage_day, int) else 0,
            usage_month=usage_month if isinstance(usage_month, int) else 0,
            usage_total=usage_total if isinstance(usage_total, int) else 0,
            usage_hour=usage_hour if isinstance(usage_hour, int) else None,
            usage_minute=usage_minute if isinstance(usage_minute, int) else None,
            usage_percentage_of_category=round(usage_percentage, 4),
            usage_trend=usage_trend,
            cost_benefit_score=round(cost_benefit_score, 2) if cost_benefit_score else None,
            size_gb=round(size_gb, 2) if size_gb else None,
            baseline=str(baseline) if baseline else None,
            nsfw=nsfw if isinstance(nsfw, bool) else None,
            has_description=has_description,
            download_count=download_count,
            download_hosts=download_hosts,
        )

        audit_models.append(audit_info)

    # Sort by usage (descending) for easier review
    audit_models.sort(key=lambda x: x.usage_month, reverse=True)

    logger.info(f"Analyzed {len(audit_models)} models for audit: {sum(1 for m in audit_models if m.at_risk)} at risk")

    return audit_models


def calculate_audit_summary(audit_models: list[ModelAuditInfo]) -> CategoryAuditSummary:
    """Calculate summary statistics from audit models.

    Args:
        audit_models: List of ModelAuditInfo objects.

    Returns:
        CategoryAuditSummary with aggregate statistics.
    """
    total_models = len(audit_models)
    models_at_risk = sum(1 for m in audit_models if m.at_risk)
    models_critical = sum(1 for m in audit_models if m.is_critical)
    models_with_warnings = sum(1 for m in audit_models if m.has_warning)

    models_with_zero_day_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_day)
    models_with_zero_month_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_month)
    models_with_zero_total_usage = sum(1 for m in audit_models if m.deletion_risk_flags.zero_usage_total)
    models_with_no_active_workers = sum(1 for m in audit_models if m.deletion_risk_flags.no_active_workers)
    models_with_no_downloads = sum(1 for m in audit_models if m.deletion_risk_flags.no_download_urls)
    models_with_non_preferred_hosts = sum(1 for m in audit_models if m.deletion_risk_flags.has_non_preferred_host)
    models_with_multiple_hosts = sum(1 for m in audit_models if m.deletion_risk_flags.has_multiple_hosts)
    models_with_low_usage = sum(1 for m in audit_models if m.deletion_risk_flags.low_usage)

    category_total_month_usage = sum(m.usage_month for m in audit_models)

    total_risk_score = sum(m.risk_score for m in audit_models)
    average_risk_score = total_risk_score / total_models if total_models > 0 else 0.0

    return CategoryAuditSummary(
        total_models=total_models,
        models_at_risk=models_at_risk,
        models_critical=models_critical,
        models_with_warnings=models_with_warnings,
        models_with_zero_day_usage=models_with_zero_day_usage,
        models_with_zero_month_usage=models_with_zero_month_usage,
        models_with_zero_total_usage=models_with_zero_total_usage,
        models_with_no_active_workers=models_with_no_active_workers,
        models_with_no_downloads=models_with_no_downloads,
        models_with_non_preferred_hosts=models_with_non_preferred_hosts,
        models_with_multiple_hosts=models_with_multiple_hosts,
        models_with_low_usage=models_with_low_usage,
        average_risk_score=round(average_risk_score, 2),
        category_total_month_usage=category_total_month_usage,
    )
