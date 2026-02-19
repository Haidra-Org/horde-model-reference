from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T", bound=str)
K = TypeVar("K", bound=str | Enum)
V = TypeVar("V")


class EnumRegistry(Generic[T]):
    """Minimal registry helper for enum-like string values.

    meta_consts.py exposes many ``register/is_known`` pairs. Centralizing the
    semantics here keeps runtime-extensible enums consistent while still
    allowing static enums for IDE help.
    """

    def __init__(self, initial: Iterable[T]):
        self._known: set[str] = set(str(item) for item in initial)

    def is_known(self, value: str | Enum) -> bool:
        return str(value) in self._known

    def register(self, value: str | Enum) -> None:
        normalized = str(value)
        if normalized in self._known:
            return
        self._known.add(normalized)

    def values(self) -> set[str]:
        return set(self._known)

    def mutable_values(self) -> set[str]:
        """Expose a live set for backwards-compatible global aliases."""
        return self._known


class DescriptorRegistry(Generic[K, V]):
    """Registry for keyed descriptors with a rebuild hook for derived data.

    Category/baseline registries need to invalidate and rebuild derived
    lookups after mutation, but only once initialization is complete. This
    helper centralizes that lifecycle so new registries get the same safety
    guards by default.
    """

    def __init__(self, rebuild: Callable[[dict[K, V]], None]):
        self._data: dict[K, V] = {}
        self._rebuild = rebuild
        self._init_complete = False

    def register(self, key: K, value: V) -> None:
        if key in self._data:
            raise ValueError(f"{key!r} is already registered")
        self._data[key] = value
        if self._init_complete:
            self._rebuild(self._data)

    def finalize(self) -> None:
        if self._init_complete:
            return
        self._init_complete = True
        self._rebuild(self._data)

    def get(self, key: K) -> V:
        return self._data[key]

    def all(self) -> dict[K, V]:
        return dict(self._data)

    def contains(self, key: K) -> bool:
        return key in self._data
