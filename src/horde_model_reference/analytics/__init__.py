"""Analytics module for model reference data.

Provides statistics calculation, audit analysis, and text model parsing functionality.
"""

from horde_model_reference.analytics.audit_analysis import (
    CategoryAuditResponse,
    CategoryAuditSummary,
    DeletionRiskFlags,
    ModelAuditInfo,
    analyze_models_for_audit,
    calculate_audit_summary,
    get_deletion_flags,
)
from horde_model_reference.analytics.statistics import (
    BaselineStats,
    CategoryStatistics,
    DownloadStats,
    TagStats,
    calculate_category_statistics,
)
from horde_model_reference.analytics.statistics_cache import StatisticsCache
from horde_model_reference.analytics.text_model_parser import (
    ParsedTextModelName,
    get_base_model_name,
    get_model_size,
    get_model_variant,
    group_text_models_by_base,
    is_quantized_variant,
    normalize_model_name,
    parse_text_model_name,
)

__all__ = [
    "BaselineStats",
    "CategoryAuditResponse",
    "CategoryAuditSummary",
    "CategoryStatistics",
    "DeletionRiskFlags",
    "DownloadStats",
    "ModelAuditInfo",
    "ParsedTextModelName",
    "StatisticsCache",
    "TagStats",
    "analyze_models_for_audit",
    "calculate_audit_summary",
    "calculate_category_statistics",
    "get_base_model_name",
    "get_deletion_flags",
    "get_model_size",
    "get_model_variant",
    "group_text_models_by_base",
    "is_quantized_variant",
    "normalize_model_name",
    "parse_text_model_name",
]
