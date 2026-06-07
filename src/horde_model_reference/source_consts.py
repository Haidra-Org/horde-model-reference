"""Shared constants and types for model record provenance ("sources").

A *source* identifies where a model record originated. The canonical
horde-model-reference data is served under the reserved id ``"horde"``; third
parties register :class:`~horde_model_reference.providers.base.ModelProvider`
instances under their own ids to contribute additional records for any category.

These constants live in their own dependency-light module so that both the query
layer (:mod:`horde_model_reference.query`) and the provider layer
(:mod:`horde_model_reference.providers`) can import them without creating an
import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence

HORDE_SOURCE_ID = "horde"
"""Reserved source id for canonical horde-model-reference data.

This is the default source for every read/query API, so existing callers that do
not pass a ``source`` argument keep their current behavior unchanged.
"""

ANY_SOURCE = "any"
"""Sentinel selector meaning "the canonical source plus every registered provider".

When used as a query selector, records from all sources are merged. By default the
canonical source wins name collisions; see
:meth:`horde_model_reference.query.ModelQuery.duplicate_names` to detect when a
collision occurred.
"""

RESERVED_SOURCE_IDS: frozenset[str] = frozenset({HORDE_SOURCE_ID, ANY_SOURCE})
"""Source ids that third-party providers may not register under."""

type SourceSelector = str | Sequence[str]
"""A source selection accepted by the read/query APIs.

May be:

* ``"horde"`` (:data:`HORDE_SOURCE_ID`) - canonical data only (the default).
* ``"any"`` (:data:`ANY_SOURCE`) - canonical data merged with all providers.
* a single provider id (e.g. ``"civitai"``).
* a sequence of ids (e.g. ``["horde", "civitai"]``) - an explicit, ordered set.
  Ordering controls collision precedence: earlier sources win.
"""


def normalize_source_selector(source: SourceSelector) -> list[str]:
    """Return *source* as a de-duplicated, order-preserving list of source ids.

    A bare string becomes a single-element list. ``ANY_SOURCE`` is preserved as a
    sentinel and resolved later by the manager (which knows the live provider set).

    Args:
        source: The source selector to normalize.

    Returns:
        list[str]: The normalized list of source ids (or ``[ANY_SOURCE]``).

    """
    if isinstance(source, str):
        return [source]

    seen: set[str] = set()
    normalized: list[str] = []
    for item in source:
        if item not in seen:
            seen.add(item)
            normalized.append(item)

    if not normalized:
        return [HORDE_SOURCE_ID]

    return normalized
