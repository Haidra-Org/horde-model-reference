"""Resolve deprecated model identifiers to their canonical storage keys."""

from __future__ import annotations

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

__all__ = ["model_name_candidates", "resolve_model_alias"]


_MODEL_ALIASES: dict[MODEL_REFERENCE_CATEGORY, dict[str, str]] = {
    MODEL_REFERENCE_CATEGORY.controlnet: {
        "control_qr_sdxl": "control_qr_xl",
    },
}
"""Deprecated external identifiers mapped to canonical model-record keys."""


def resolve_model_alias(category: MODEL_REFERENCE_CATEGORY, model_name: str) -> str:
    """Return the canonical identifier for a model name."""
    return _MODEL_ALIASES.get(category, {}).get(model_name, model_name)


def model_name_candidates(category: MODEL_REFERENCE_CATEGORY, model_name: str) -> tuple[str, ...]:
    """Return storage lookup candidates, with the canonical name first.

    The deprecated key remains a read fallback while a PRIMARY deployment is
    being migrated. It is never a canonical write target.
    """
    canonical_name = resolve_model_alias(category, model_name)
    aliases = tuple(
        alias
        for alias, target in _MODEL_ALIASES.get(category, {}).items()
        if target == canonical_name and alias != model_name
    )
    return tuple(dict.fromkeys((canonical_name, *aliases, model_name)))
