"""Generic EnumRegistry and DescriptorRegistry base classes for runtime-extensible registries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import TypeVar

T = TypeVar("T", bound=str)
K = TypeVar("K", bound=str | Enum)
V = TypeVar("V")


class EnumRegistry[T]:
    """Minimal registry helper for enum-like string values.

    meta_consts.py exposes many ``register/is_known`` pairs. Centralizing the
    semantics here keeps runtime-extensible enums consistent while still
    allowing static enums for IDE help.
    """

    def __init__(self, initial: Iterable[T]) -> None:
        """Initialize the enum registry with an optional iterable of initial values.

        Args:
            initial (Iterable[T]): An iterable of initial string values to populate the registry with.

        """
        self._known: set[str] = {str(item) for item in initial}

    def is_known(self, value: str | Enum) -> bool:
        """Check if a value is known (registered) in the enum registry.

        Args:
            value (str | Enum): The value to check for membership in the registry.

        Returns:
            bool: True if the value is known (registered), False otherwise.

        """
        return str(value) in self._known

    def register(self, value: str | Enum) -> None:
        """Register a new value in the enum registry.

        Attempting to register a value that is already known is a no-op.

        Args:
            value (str | Enum): The value to register.

        """
        normalized = str(value)
        if normalized in self._known:
            return
        self._known.add(normalized)

    def values(self) -> set[str]:
        """Get a set of all known (registered) values in the enum registry.

        Returns:
            set[str]: A set of all known (registered) string values in the registry.

        """
        return set(self._known)

    def mutable_values(self) -> set[str]:
        """Expose a live set for backwards-compatible global aliases.

        Returns:
            set[str]: A live set of known values.

        """
        return self._known


class DescriptorRegistry[K, V]:
    """Registry for keyed descriptors with a rebuild hook for derived data.

    Category/baseline registries need to invalidate and rebuild derived
    lookups after mutation, but only once initialization is complete. This
    helper centralizes that lifecycle so new registries get the same safety
    guards by default.
    """

    def __init__(self, rebuild: Callable[[dict[K, V]], None]) -> None:
        """Initialize the descriptor registry.

        Args:
            rebuild (Callable[[dict[K, V]], None]): Function to call to rebuild derived data when the registry is
                updated.

        """
        self._data: dict[K, V] = {}
        self._rebuild = rebuild
        self._init_complete = False

    def register(self, key: K, value: V) -> None:
        """Register a key-value pair, raising if the key is already registered.

        Args:
            key (K): The key to register.
            value (V): The value to associate with the key.

        Raises:
            ValueError: If the key is already registered.

        """
        if key in self._data:
            raise ValueError(f"{key!r} is already registered")
        self._data[key] = value
        if self._init_complete:
            self._rebuild(self._data)

    def update_value(self, key: K, value: V) -> None:
        """Update the value for an already registered key, raising if the key is not registered.

        Args:
            key (K): The key to update.
            value (V): The new value to associate with the key.

        Raises:
            ValueError: If the key is not already registered.

        """
        if key not in self._data:
            raise ValueError(f"{key!r} is not registered and cannot be updated")
        self._data[key] = value
        if self._init_complete:
            self._rebuild(self._data)

    def finalize(self) -> None:
        """Mark initialization complete and trigger a final rebuild with all registered data.

        Subsequent calls to register() will trigger immediate rebuilds, so this should only be called once after the
        initial batch of registrations is done.
        """
        if self._init_complete:
            return
        self._init_complete = True
        self._rebuild(self._data)

    def get(self, key: K) -> V:
        """Get the value for a registered key, or raise if the key is unknown.

        Args:
            key (K): The key to look up.

        Returns:
            V: The value associated with the key.

        """
        return self._data[key]

    def all(self) -> dict[K, V]:
        """Get a copy of all registered key-value pairs.

        Returns:
            dict[K, V]: A copy of all registered key-value pairs.

        """
        return dict(self._data)

    def contains(self, key: K) -> bool:
        """Check if a key is registered.

        Args:
            key (K): The key to check.

        Returns:
            bool: True if the key is registered, False otherwise.

        """
        return key in self._data
