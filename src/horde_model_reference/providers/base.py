"""Abstract interface for third-party model providers.

A :class:`ModelProvider` lets a third party contribute model records for one or
more :class:`~horde_model_reference.meta_consts.MODEL_REFERENCE_CATEGORY` values
without going through the canonical write loop. Providers are **read-only** from
the library's perspective: any persistence, validation-on-write, or moderation is
the third party's own responsibility.

Providers return already-validated Pydantic records (subclasses of
:class:`~horde_model_reference.model_reference_records.GenericModelRecord`). A
provider may reuse the built-in record type for a category
(:data:`~horde_model_reference.model_reference_records.MODEL_RECORD_TYPE_LOOKUP`)
or register and return its own subclass via
:func:`~horde_model_reference.model_reference_records.register_record_type`.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord
from horde_model_reference.source_consts import RESERVED_SOURCE_IDS


class ModelProvider(ABC):
    """Read-only third-party source of model records for one or more categories.

    Subclasses must implement :attr:`source_id`, :meth:`provided_categories`, and
    :meth:`fetch_category`. The async variant has a thread-pool default and only
    needs overriding when a native async implementation is available.

    Implementations should:

    * Return a mapping of ``model_name -> record`` from :meth:`fetch_category`,
      or ``None`` when the category cannot be served right now.
    * Validate/construct records themselves (the library does not re-validate
      provider output against any schema).
    * Be resilient: raising from :meth:`fetch_category` is tolerated by the
      manager (the error is logged and that provider is skipped), but returning
      ``None`` is the preferred way to signal "no data".
    """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Return this provider's stable, unique source id.

        Must not be one of :data:`~horde_model_reference.source_consts.RESERVED_SOURCE_IDS`
        (``"horde"`` / ``"any"``). The id is how consumers select this provider in
        queries, so it should be stable across versions (e.g. ``"civitai"``).
        """

    @abstractmethod
    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the set of categories this provider can serve."""

    @abstractmethod
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return ``model_name -> record`` for *category*, or ``None`` if unavailable.

        Args:
            category: The category to fetch.
            force_refresh: If ``True``, bypass any provider-side cache.

        Returns:
            dict[str, GenericModelRecord] | None: Validated records keyed by model
                name, or ``None`` when the provider cannot serve this category.

        """

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Asynchronously fetch records for *category*.

        The default implementation runs :meth:`fetch_category` in a worker thread.
        Override when a native async data path is available.

        Args:
            category: The category to fetch.
            force_refresh: If ``True``, bypass any provider-side cache.

        Returns:
            dict[str, GenericModelRecord] | None: Validated records keyed by model
                name, or ``None`` when the provider cannot serve this category.

        """
        return await asyncio.to_thread(
            self.fetch_category,
            category,
            force_refresh=force_refresh,
        )

    def serves_category(self, category: MODEL_REFERENCE_CATEGORY | str) -> bool:
        """Return whether this provider advertises support for *category*."""
        return category in self.provided_categories()

    def supports_writes(self) -> bool:
        """Return whether this provider supports write operations.

        Always ``False``: third-party write semantics are explicitly out of scope.
        The hook exists so consumers can branch on capability uniformly.
        """
        return False

    def cache_ttl_seconds(self) -> int | None:
        """Return a staleness hint in seconds, or ``None`` for no hint.

        This is advisory metadata for consumers; the library does not currently
        cache provider output itself.
        """
        return None

    def validate_source_id(self) -> None:
        """Raise ``ValueError`` if :attr:`source_id` is empty or reserved.

        Called by the registry at registration time.
        """
        if not self.source_id:
            raise ValueError("Provider source_id must be a non-empty string.")
        if self.source_id in RESERVED_SOURCE_IDS:
            raise ValueError(
                f"Provider source_id {self.source_id!r} is reserved. Reserved ids: {sorted(RESERVED_SOURCE_IDS)}.",
            )
