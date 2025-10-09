"""Backend providers for model reference data sources.

This module defines the abstract backend interface and concrete implementations
for fetching model reference data from different sources (GitHub, database, etc.).
"""

from .base import ModelReferenceBackend
from .legacy_github import LegacyGitHubBackend

__all__ = [
    "LegacyGitHubBackend",
    "ModelReferenceBackend",
]
