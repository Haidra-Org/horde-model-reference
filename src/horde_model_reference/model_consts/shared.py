"""Shared model constants and enums used across multiple model categories."""

from enum import auto

from strenum import StrEnum

from horde_model_reference.registries import EnumRegistry


class MODEL_STYLE(StrEnum):
    """An enum of all the model styles."""

    generalist = auto()
    anime = auto()
    furry = auto()
    artistic = auto()
    other = auto()
    realistic = auto()


_MODEL_STYLE_REGISTRY = EnumRegistry(item.value for item in MODEL_STYLE)


def register_model_style(style: MODEL_STYLE | str) -> None:
    """Register a new model style."""
    _MODEL_STYLE_REGISTRY.register(style)


def is_known_model_style(style: MODEL_STYLE | str) -> bool:
    """Check if a model style is known."""
    return _MODEL_STYLE_REGISTRY.is_known(style)


_KNOWN_TAGS_INITIAL = (
    "anime",
    "manga",
    "cyberpunk",
    "tv show",
    "booru",
    "retro",
    "character",
    "hentai",
    "scenes",
    "low poly",
    "cg",
    "sketch",
    "high resolution",
    "landscapes",
    "comic",
    "cartoon",
    "painting",
    "game",
)

_TAG_REGISTRY = EnumRegistry(_KNOWN_TAGS_INITIAL)
KNOWN_TAGS = _TAG_REGISTRY.mutable_values()


def get_known_tags() -> list[str]:
    """Return a snapshot of all known tags as a list."""
    return sorted(_TAG_REGISTRY.values())


def register_tag(tag: str | StrEnum) -> None:
    """Register a new known tag."""
    _TAG_REGISTRY.register(tag)


def is_known_tag(tag: str | StrEnum) -> bool:
    """Check if a tag is known."""
    return _TAG_REGISTRY.is_known(tag)
