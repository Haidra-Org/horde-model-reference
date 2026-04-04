"""Analytics module for model reference data.

Provides statistics calculation, deletion risk analysis, and text model parsing functionality.
"""

from horde_model_reference.analytics.deletion_risk_analysis import (
    CategoryDeletionRiskResponse,
    CategoryDeletionRiskSummary,
    DeletionRiskFlags,
    ModelDeletionRiskInfo,
    ModelDeletionRiskInfoFactory,
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
    NameFormatSchema,
    ParsedTextModelName,
    TextModelGroupSummary,
    compute_group_summaries,
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
    "CategoryDeletionRiskResponse",
    "CategoryDeletionRiskSummary",
    "CategoryStatistics",
    "DeletionRiskFlags",
    "DownloadStats",
    "ModelDeletionRiskInfo",
    "ModelDeletionRiskInfoFactory",
    "NameFormatSchema",
    "ParsedTextModelName",
    "StatisticsCache",
    "TagStats",
    "TextModelGroupSummary",
    "calculate_category_statistics",
    "compute_group_summaries",
    "get_base_model_name",
    "get_model_size",
    "get_model_variant",
    "group_text_models_by_base",
    "is_quantized_variant",
    "normalize_model_name",
    "parse_text_model_name",
]
