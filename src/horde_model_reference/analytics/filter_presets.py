"""Filter presets for audit analysis.

Provides predefined filter presets to quickly identify models of interest
(e.g., deletion candidates, zero usage models, models with missing data).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference.analytics.constants import LOW_USAGE_THRESHOLD

if TYPE_CHECKING:
    from horde_model_reference.analytics.audit_analysis import ModelAuditInfo


class AuditFilterPreset(str, Enum):
    """Predefined filter presets for audit analysis."""

    DELETION_CANDIDATES = "deletion_candidates"
    """Models that are candidates for deletion (any flags, very low usage, or no workers)."""

    ZERO_USAGE = "zero_usage"
    """Models with zero monthly usage."""

    NO_WORKERS = "no_workers"
    """Models with no active workers."""

    MISSING_DATA = "missing_data"
    """Models missing critical data (description or baseline)."""

    HOST_ISSUES = "host_issues"
    """Models with file hosting issues (non-preferred hosts, multiple hosts, or unknown hosts)."""

    CRITICAL = "critical"
    """Models in critical state (zero month usage AND no workers)."""

    LOW_USAGE = "low_usage"
    """Models with very low usage (< 0.1% of category total)."""


def filter_deletion_candidates(model: ModelAuditInfo) -> bool:
    """Check if model is a deletion candidate.

    A model is a deletion candidate if it has:
    - Any deletion risk flags set, OR
    - Very low usage (< 0.1% of category), OR
    - No active workers

    Args:
        model: The model to check.

    Returns:
        True if model matches deletion candidate criteria.
    """
    return model.at_risk or model.usage_percentage_of_category < LOW_USAGE_THRESHOLD or model.worker_count == 0


def filter_zero_usage(model: ModelAuditInfo) -> bool:
    """Check if model has zero monthly usage.

    Args:
        model: The model to check.

    Returns:
        True if model has zero usage in the past month.
    """
    return model.usage_month == 0


def filter_no_workers(model: ModelAuditInfo) -> bool:
    """Check if model has no active workers.

    Args:
        model: The model to check.

    Returns:
        True if model has zero active workers.
    """
    return model.worker_count == 0


def filter_missing_data(model: ModelAuditInfo) -> bool:
    """Check if model is missing critical data.

    A model is missing data if it lacks:
    - Description, OR
    - Baseline (for categories where baseline is applicable)

    Args:
        model: The model to check.

    Returns:
        True if model is missing description or baseline.
    """
    return model.deletion_risk_flags.missing_description or model.deletion_risk_flags.missing_baseline


def filter_host_issues(model: ModelAuditInfo) -> bool:
    """Check if model has file hosting issues.

    Issues include:
    - Downloads on non-preferred hosts
    - Downloads across multiple hosts
    - Unknown or unparseable hosts
    - No download URLs

    Args:
        model: The model to check.

    Returns:
        True if model has any hosting-related issues.
    """
    return (
        model.deletion_risk_flags.has_non_preferred_host
        or model.deletion_risk_flags.has_multiple_hosts
        or model.deletion_risk_flags.has_unknown_host
        or model.deletion_risk_flags.no_download_urls
    )


def filter_critical(model: ModelAuditInfo) -> bool:
    """Check if model is in critical state.

    Critical state = zero month usage AND no active workers.

    Args:
        model: The model to check.

    Returns:
        True if model is in critical state.
    """
    return model.is_critical


def filter_low_usage(model: ModelAuditInfo) -> bool:
    """Check if model has very low usage.

    Low usage = less than 0.1% of category's total monthly usage.

    Args:
        model: The model to check.

    Returns:
        True if model has low usage.
    """
    return model.deletion_risk_flags.low_usage


PRESET_FILTERS: dict[AuditFilterPreset, Callable[[ModelAuditInfo], bool]] = {
    AuditFilterPreset.DELETION_CANDIDATES: filter_deletion_candidates,
    AuditFilterPreset.ZERO_USAGE: filter_zero_usage,
    AuditFilterPreset.NO_WORKERS: filter_no_workers,
    AuditFilterPreset.MISSING_DATA: filter_missing_data,
    AuditFilterPreset.HOST_ISSUES: filter_host_issues,
    AuditFilterPreset.CRITICAL: filter_critical,
    AuditFilterPreset.LOW_USAGE: filter_low_usage,
}


def apply_preset_filter(models: list[ModelAuditInfo], preset: str | AuditFilterPreset) -> list[ModelAuditInfo]:
    """Apply a preset filter to a list of models.

    Args:
        models: List of models to filter.
        preset: The preset to apply (string or enum value).

    Returns:
        Filtered list of models matching the preset criteria.

    Raises:
        ValueError: If preset is not recognized.
    """
    if isinstance(preset, str):
        try:
            preset_enum = AuditFilterPreset(preset)
        except ValueError as e:
            valid_presets = ", ".join(p.value for p in AuditFilterPreset)
            raise ValueError(f"Unknown preset: '{preset}'. Valid presets: {valid_presets}") from e
    else:
        preset_enum = preset

    filter_func = PRESET_FILTERS.get(preset_enum)
    if not filter_func:
        raise ValueError(f"No filter function defined for preset: {preset_enum}")

    filtered = [model for model in models if filter_func(model)]

    logger.debug(f"Applied preset '{preset_enum.value}': {len(filtered)}/{len(models)} models matched")

    return filtered


def get_available_presets() -> list[dict[str, str]]:
    """Get list of available presets with descriptions.

    Returns:
        List of dicts with 'name' and 'description' keys.
    """
    return [
        {
            "name": AuditFilterPreset.DELETION_CANDIDATES.value,
            "description": "Models that are candidates for deletion (any flags, very low usage, or no workers)",
        },
        {
            "name": AuditFilterPreset.ZERO_USAGE.value,
            "description": "Models with zero monthly usage",
        },
        {
            "name": AuditFilterPreset.NO_WORKERS.value,
            "description": "Models with no active workers",
        },
        {
            "name": AuditFilterPreset.MISSING_DATA.value,
            "description": "Models missing critical data (description or baseline)",
        },
        {
            "name": AuditFilterPreset.HOST_ISSUES.value,
            "description": "Models with file hosting issues (non-preferred hosts, multiple hosts, or unknown hosts)",
        },
        {
            "name": AuditFilterPreset.CRITICAL.value,
            "description": "Models in critical state (zero month usage AND no workers)",
        },
        {
            "name": AuditFilterPreset.LOW_USAGE.value,
            "description": "Models with very low usage (< 0.1% of category total)",
        },
    ]
