from enum import auto

from strenum import StrEnum

from horde_model_reference.registries import EnumRegistry


class TEXT_BACKENDS(StrEnum):
    """An enum of all the text backends."""

    aphrodite = auto()
    koboldcpp = auto()


_TEXT_BACKEND_REGISTRY = EnumRegistry(item.value for item in TEXT_BACKENDS)
KNOWN_TEXT_BACKENDS = _TEXT_BACKEND_REGISTRY.mutable_values()


def register_text_backend(backend: str) -> None:
    """Register a new text backend.

    Args:
        backend: The text backend to register.
    """
    _TEXT_BACKEND_REGISTRY.register(backend)


def is_known_text_backend(backend: str) -> bool:
    """Check if a text backend is known.

    Args:
        backend: The text backend to check.

    Returns:
        True if the text backend is known, False otherwise.
    """
    return _TEXT_BACKEND_REGISTRY.is_known(backend)
