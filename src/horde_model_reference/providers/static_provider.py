"""A ready-to-use, in-memory :class:`ModelProvider`.

Most third parties already have their records in hand (or as plain dicts) and only
need to expose them under a source id. :class:`StaticModelProvider` covers that case
without requiring a custom subclass: construct it with already-built records, or use
:meth:`StaticModelProvider.from_raw` to validate plain dicts against each category's
record type. Subclass :class:`~horde_model_reference.providers.base.ModelProvider`
directly only when records must be fetched lazily from a live/remote source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import MODEL_RECORD_TYPE_LOOKUP, GenericModelRecord
from horde_model_reference.providers.base import ModelProvider


def _record_type_for(category: MODEL_REFERENCE_CATEGORY | str) -> type[GenericModelRecord]:
    """Return the record class registered for *category*, or ``GenericModelRecord``.

    Unknown category strings (a provider may define its own) fall back to
    :class:`~horde_model_reference.model_reference_records.GenericModelRecord`.
    """
    try:
        enum_category = MODEL_REFERENCE_CATEGORY(category)
    except ValueError:
        return GenericModelRecord
    return MODEL_RECORD_TYPE_LOOKUP.get(enum_category, GenericModelRecord)


class StaticModelProvider(ModelProvider):
    """A :class:`ModelProvider` backed by a fixed, in-memory set of records.

    This is the lowest-friction way to contribute model records: hand it the records
    you already have and register it. The provider is read-only and never refreshes
    (``force_refresh`` is a no-op), which is exactly what you want for a static
    snapshot.

    Example:
        ```python
        from horde_model_reference import (
            ModelReferenceManager,
            MODEL_REFERENCE_CATEGORY,
            StaticModelProvider,
        )

        provider = StaticModelProvider.from_raw(
            "civitai",
            {
                MODEL_REFERENCE_CATEGORY.image_generation: {
                    "my_model": {"baseline": "stable_diffusion_xl", "nsfw": False},
                },
            },
        )
        ModelReferenceManager().register_provider(provider)
        ```
    """

    def __init__(
        self,
        source_id: str,
        records_by_category: Mapping[MODEL_REFERENCE_CATEGORY | str, Mapping[str, GenericModelRecord]],
        *,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        """Store already-built records to serve under *source_id*.

        Args:
            source_id: This provider's stable, unique id (e.g. ``"civitai"``). Must not
                be a reserved id (``"horde"`` / ``"any"``); validated on construction.
            records_by_category: Mapping of category to ``model_name -> record``. The
                records must already be validated instances of the appropriate
                :class:`~horde_model_reference.model_reference_records.GenericModelRecord`
                subclass (use :meth:`from_raw` to build them from plain dicts).
            cache_ttl_seconds: Optional advisory staleness hint reported via
                :meth:`cache_ttl_seconds`.

        Raises:
            ValueError: If *source_id* is empty or reserved.

        """
        self._source_id = source_id
        self._records: dict[MODEL_REFERENCE_CATEGORY | str, dict[str, GenericModelRecord]] = {
            category: dict(records) for category, records in records_by_category.items()
        }
        self._cache_ttl_seconds = cache_ttl_seconds
        self.validate_source_id()

    @classmethod
    def from_raw(
        cls,
        source_id: str,
        raw_by_category: Mapping[MODEL_REFERENCE_CATEGORY | str, Mapping[str, Mapping[str, object]]],
        *,
        cache_ttl_seconds: int | None = None,
    ) -> StaticModelProvider:
        """Build a provider by validating plain dicts against each category's record type.

        Each raw record is validated with the record class registered for its category
        (:data:`~horde_model_reference.model_reference_records.MODEL_RECORD_TYPE_LOOKUP`,
        falling back to ``GenericModelRecord``). The model name from the mapping key is
        injected as the record's ``name`` so callers need not repeat it.

        Args:
            source_id: This provider's stable, unique id (e.g. ``"civitai"``).
            raw_by_category: Mapping of category to ``model_name -> raw field dict``.
            cache_ttl_seconds: Optional advisory staleness hint.

        Returns:
            StaticModelProvider: A provider serving the validated records.

        Raises:
            pydantic.ValidationError: If any raw record fails validation.
            ValueError: If *source_id* is empty or reserved.

        """
        records_by_category: dict[MODEL_REFERENCE_CATEGORY | str, dict[str, GenericModelRecord]] = {}
        for category, raw_records in raw_by_category.items():
            record_cls = _record_type_for(category)
            records_by_category[category] = {
                name: record_cls.model_validate({**raw, "name": name}) for name, raw in raw_records.items()
            }
        return cls(source_id, records_by_category, cache_ttl_seconds=cache_ttl_seconds)

    @property
    @override
    def source_id(self) -> str:
        """Return this provider's source id."""
        return self._source_id

    @override
    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the categories this provider holds records for."""
        return set(self._records)

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return a copy of the stored records for *category*, or ``None`` if absent."""
        records = self._records.get(category)
        if records is None:
            return None
        return dict(records)

    @override
    def cache_ttl_seconds(self) -> int | None:
        """Return the advisory staleness hint set at construction, if any."""
        return self._cache_ttl_seconds
