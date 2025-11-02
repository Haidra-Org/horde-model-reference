"""Constants for analytics calculations.

Centralized configuration values used across statistics and audit analysis modules.
"""

from __future__ import annotations

# Statistics calculation constants
TOP_TAGS_LIMIT = 20
"""Maximum number of top tags to include in statistics."""

TOP_STYLES_LIMIT = 20
"""Maximum number of top styles to include in statistics."""

# Audit analysis constants
LOW_USAGE_THRESHOLD = 0.1
"""Threshold (as percentage) for flagging models with low usage.

Models with monthly usage below this percentage of category total are flagged.
Example: 0.1 means < 0.1% of category usage is considered low.
"""

# Parameter bucket ranges for text models (in billions)
PARAMETER_BUCKETS = [
    (0, 3_000_000_000, "< 3B"),
    (3_000_000_000, 6_000_000_000, "3B-6B"),
    (6_000_000_000, 9_000_000_000, "6B-9B"),
    (9_000_000_000, 13_000_000_000, "9B-13B"),
    (13_000_000_000, 20_000_000_000, "13B-20B"),
    (20_000_000_000, 35_000_000_000, "20B-35B"),
    (35_000_000_000, 70_000_000_000, "35B-70B"),
    (70_000_000_000, float("inf"), "> 70B"),
]
"""Parameter count buckets as (min, max, label) tuples.

Each tuple defines a parameter range and its display label.
The last bucket uses infinity to capture all larger models.
"""
