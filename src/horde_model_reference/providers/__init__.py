"""Third-party model provider system.

This package lets consumers of horde-model-reference register external sources of
model records ("providers") for any category, and query canonical data, a specific
provider, or all sources together.

See :mod:`horde_model_reference.source_consts` for the source-selection constants
(``HORDE_SOURCE_ID``, ``ANY_SOURCE``) and the ``SourceSelector`` type.
"""

from __future__ import annotations

from horde_model_reference.providers.base import ModelProvider
from horde_model_reference.providers.pending_provider import PENDING_SOURCE_ID, PendingModelProvider
from horde_model_reference.providers.registry import ModelProviderRegistry
from horde_model_reference.providers.static_provider import StaticModelProvider
from horde_model_reference.source_consts import (
    ANY_SOURCE,
    HORDE_SOURCE_ID,
    RESERVED_SOURCE_IDS,
    SourceSelector,
    normalize_source_selector,
)

__all__ = [
    "ANY_SOURCE",
    "HORDE_SOURCE_ID",
    "PENDING_SOURCE_ID",
    "RESERVED_SOURCE_IDS",
    "ModelProvider",
    "ModelProviderRegistry",
    "PendingModelProvider",
    "SourceSelector",
    "StaticModelProvider",
    "normalize_source_selector",
]
