"""Thread-safe registry of third-party model providers.

The registry is owned by :class:`~horde_model_reference.model_reference_manager.ModelReferenceManager`
(one instance per manager singleton). Consumers usually interact with it through
the manager's ``register_provider`` / ``unregister_provider`` / ``list_providers``
methods rather than constructing it directly.
"""

from __future__ import annotations

import threading

from loguru import logger

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.providers.base import ModelProvider


class ModelProviderRegistry:
    """In-memory, thread-safe registry mapping source ids to providers."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.RLock()
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider, *, replace: bool = False) -> None:
        """Register *provider* under its :attr:`~ModelProvider.source_id`.

        Args:
            provider: The provider to register.
            replace: If ``True``, silently replace an existing provider with the
                same source id. If ``False`` (default), a duplicate id raises.

        Raises:
            ValueError: If the source id is empty/reserved, or already registered
                and *replace* is ``False``.

        """
        provider.validate_source_id()
        source_id = provider.source_id
        with self._lock:
            if not replace and source_id in self._providers:
                raise ValueError(
                    f"A provider is already registered under source id {source_id!r}. Pass replace=True to override.",
                )
            self._providers[source_id] = provider
            logger.info(f"Registered model provider {source_id!r} (replace={replace}).")

    def unregister(self, source_id: str) -> bool:
        """Remove the provider registered under *source_id*.

        Args:
            source_id: The source id to remove.

        Returns:
            bool: ``True`` if a provider was removed, ``False`` if none existed.

        """
        with self._lock:
            removed = self._providers.pop(source_id, None)
        if removed is not None:
            logger.info(f"Unregistered model provider {source_id!r}.")
            return True
        return False

    def get(self, source_id: str) -> ModelProvider | None:
        """Return the provider registered under *source_id*, or ``None``."""
        with self._lock:
            return self._providers.get(source_id)

    def has(self, source_id: str) -> bool:
        """Return whether a provider is registered under *source_id*."""
        with self._lock:
            return source_id in self._providers

    def source_ids(self) -> list[str]:
        """Return the ids of all registered providers (registration order)."""
        with self._lock:
            return list(self._providers)

    def all(self) -> dict[str, ModelProvider]:
        """Return a shallow copy of the ``source_id -> provider`` mapping."""
        with self._lock:
            return dict(self._providers)

    def providers_for(self, category: MODEL_REFERENCE_CATEGORY | str) -> list[ModelProvider]:
        """Return registered providers that advertise support for *category*.

        Args:
            category: The category to match.

        Returns:
            list[ModelProvider]: Matching providers in registration order.

        """
        with self._lock:
            providers = list(self._providers.values())
        return [p for p in providers if p.serves_category(category)]

    def clear(self) -> None:
        """Remove all registered providers (primarily for tests)."""
        with self._lock:
            self._providers.clear()
