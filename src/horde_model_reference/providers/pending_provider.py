"""A :class:`ModelProvider` that surfaces a PRIMARY's pending-queue models as "beta".

As the ecosystem moves to a service-only source of truth, the old "run a beta by pointing
the GitHub backend at a branch" trick no longer applies. Instead, models sitting in the
PRIMARY's pending queue (``PENDING``/``APPROVED``) are treated as beta models: this provider
reads them from the PRIMARY's ``/{category}/pending`` endpoint and exposes them under the
``"pending"`` source, so a REPLICA can opt specific categories into beta via the normal
multi-source query surface (e.g. ``manager.query(category, source=["pending", "horde"])``).

The provider is read-only and authenticates with a reader-level Horde API key (the pending
read surfaces require any valid key). Responses are cached briefly so the library — which
does not cache provider output itself — is not hammered on every reference load.
"""

from __future__ import annotations

import threading
import time
from typing import Any, override

import httpx
from loguru import logger

from horde_model_reference.http_retry import (
    RetryableHTTPStatusError,
    http_retry_async,
    http_retry_sync,
    is_retryable_status_code,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import MODEL_RECORD_TYPE_LOOKUP, GenericModelRecord
from horde_model_reference.providers.base import ModelProvider

PENDING_SOURCE_ID = "pending"
"""The stable provider source id under which a PRIMARY's pending (beta) models are served."""


def _record_type_for(category: MODEL_REFERENCE_CATEGORY) -> type[GenericModelRecord]:
    """Return the record class registered for *category*, or ``GenericModelRecord``."""
    return MODEL_RECORD_TYPE_LOOKUP.get(category, GenericModelRecord)


class PendingModelProvider(ModelProvider):
    """Serves a PRIMARY's pending-queue (beta) models under the ``"pending"`` source.

    The provider fetches ``GET {primary_api_url}/model_references/v2/{category}/pending`` —
    which the PRIMARY materializes into v2 records keyed by model name — validates each
    record against the category's record type, and caches the result per category for
    ``cache_ttl_seconds``.
    """

    def __init__(
        self,
        *,
        primary_api_url: str,
        apikey: str,
        categories: set[MODEL_REFERENCE_CATEGORY],
        cache_ttl_seconds: int = 60,
        timeout_seconds: float = 10.0,
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        """Configure the provider.

        Args:
            primary_api_url: Base URL of the PRIMARY API including the ``/api`` root path
                (e.g. ``"https://models.aihorde.net/api"``).
            apikey: A reader-level Horde API key sent as the ``apikey`` header.
            categories: The categories to advertise and serve.
            cache_ttl_seconds: How long a fetched category is reused before refetching.
            timeout_seconds: Per-request HTTP timeout.
            retry_max_attempts: Max attempts for transient HTTP failures.
            retry_backoff_seconds: Minimum backoff between retries.
        """
        self._primary_api_url = primary_api_url.rstrip("/")
        self._apikey = apikey
        self._categories = set(categories)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

        self._lock = threading.RLock()
        self._cache: dict[MODEL_REFERENCE_CATEGORY, tuple[float, dict[str, GenericModelRecord]]] = {}

    @property
    @override
    def source_id(self) -> str:
        """Return the stable ``"pending"`` source id."""
        return PENDING_SOURCE_ID

    @override
    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the configured beta categories."""
        return set(self._categories)

    @override
    def cache_ttl_seconds(self) -> int | None:
        """Return the advisory staleness hint for this provider's data."""
        return self._cache_ttl_seconds

    def _category_url(self, category: MODEL_REFERENCE_CATEGORY) -> str:
        return f"{self._primary_api_url}/model_references/v2/{category.value}/pending"

    def _as_category(self, category: MODEL_REFERENCE_CATEGORY | str) -> MODEL_REFERENCE_CATEGORY | None:
        if isinstance(category, MODEL_REFERENCE_CATEGORY):
            return category
        try:
            return MODEL_REFERENCE_CATEGORY(category)
        except ValueError:
            return None

    def _cached(self, category: MODEL_REFERENCE_CATEGORY) -> dict[str, GenericModelRecord] | None:
        with self._lock:
            entry = self._cache.get(category)
            if entry is None:
                return None
            cached_at, records = entry
            if (time.monotonic() - cached_at) >= self._cache_ttl_seconds:
                return None
            return dict(records)

    def _store(self, category: MODEL_REFERENCE_CATEGORY, records: dict[str, GenericModelRecord]) -> None:
        with self._lock:
            self._cache[category] = (time.monotonic(), records)

    def _parse(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        raw: dict[str, Any],
    ) -> dict[str, GenericModelRecord]:
        """Validate the raw ``{name: record_dict}`` payload into records, skipping bad entries."""
        record_cls = _record_type_for(category)
        records: dict[str, GenericModelRecord] = {}
        for name, fields in raw.items():
            try:
                records[name] = record_cls.model_validate({**fields, "name": name})
            except Exception as exc:
                logger.warning(f"Skipping invalid pending {category.value} model {name!r}: {exc}")
        return records

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return pending (beta) records for *category*, or ``None`` if unavailable."""
        resolved = self._as_category(category)
        if resolved is None or resolved not in self._categories:
            return None

        if not force_refresh:
            cached = self._cached(resolved)
            if cached is not None:
                return cached

        url = self._category_url(resolved)
        try:
            for attempt in http_retry_sync(
                max_attempts=self._retry_max_attempts,
                min_wait=self._retry_backoff_seconds,
            ):
                with attempt:
                    response = httpx.get(
                        url,
                        headers={"apikey": self._apikey},
                        timeout=self._timeout_seconds,
                    )
                    if is_retryable_status_code(response.status_code):
                        raise RetryableHTTPStatusError(response)
                    if response.status_code != 200:
                        logger.warning(f"PRIMARY pending endpoint returned {response.status_code} for {resolved}")
                        return None
                    raw: dict[str, Any] = response.json()
                    records = self._parse(resolved, raw)
                    self._store(resolved, records)
                    return dict(records)
        except Exception as exc:
            logger.warning(f"Failed to fetch pending {resolved.value} models from PRIMARY: {exc}")
        return None

    @override
    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Async counterpart to :meth:`fetch_category`."""
        resolved = self._as_category(category)
        if resolved is None or resolved not in self._categories:
            return None

        if not force_refresh:
            cached = self._cached(resolved)
            if cached is not None:
                return cached

        url = self._category_url(resolved)
        try:
            async for attempt in http_retry_async(
                max_attempts=self._retry_max_attempts,
                min_wait=self._retry_backoff_seconds,
            ):
                with attempt:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            url,
                            headers={"apikey": self._apikey},
                            timeout=self._timeout_seconds,
                        )
                    if is_retryable_status_code(response.status_code):
                        raise RetryableHTTPStatusError(response)
                    if response.status_code != 200:
                        logger.warning(f"PRIMARY pending endpoint returned {response.status_code} for {resolved}")
                        return None
                    raw: dict[str, Any] = response.json()
                    records = self._parse(resolved, raw)
                    self._store(resolved, records)
                    return dict(records)
        except Exception as exc:
            logger.warning(f"Failed to fetch pending {resolved.value} models from PRIMARY (async): {exc}")
        return None
