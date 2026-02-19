"""Text backend name manipulation utilities.

Functions for detecting, stripping, and generating text-backend-prefixed
model name variants used in the legacy Horde API format.
"""

from __future__ import annotations

from horde_model_reference.meta_consts import TEXT_BACKENDS

TEXT_LEGACY_BACKEND_PREFIXES: dict[TEXT_BACKENDS, str] = {
    TEXT_BACKENDS.aphrodite: "aphrodite/",
    TEXT_BACKENDS.koboldcpp: "koboldcpp/",
}
"""Backend prefixes on duplicate entries for backwards compatibility in the legacy format."""


def has_legacy_text_backend_prefix(model_name: str) -> bool:
    """Check if a model name has a legacy text backend prefix.

    Args:
        model_name: The model name to check.

    Returns:
        True if the model name has a legacy text backend prefix, False otherwise.
    """
    return any(model_name.startswith(prefix) for prefix in TEXT_LEGACY_BACKEND_PREFIXES.values())


def strip_backend_prefix(model_name: str) -> str:
    """Strip backend prefix from a model name if present.

    Args:
        model_name: The model name to strip.

    Returns:
        The model name without the backend prefix.

    Example:
        >>> strip_backend_prefix("koboldcpp/Broken-Tutu-24B")
        'Broken-Tutu-24B'
        >>> strip_backend_prefix("aphrodite/ReadyArt/Broken-Tutu-24B")
        'ReadyArt/Broken-Tutu-24B'
        >>> strip_backend_prefix("ReadyArt/Broken-Tutu-24B")
        'ReadyArt/Broken-Tutu-24B'
    """
    for prefix in TEXT_LEGACY_BACKEND_PREFIXES.values():
        if model_name.startswith(prefix):
            return model_name[len(prefix) :]
    return model_name


def get_model_name_variants(canonical_name: str) -> list[str]:
    """Get all possible name variants for a canonical model name.

    Given a canonical name like "ReadyArt/Broken-Tutu-24B", returns all possible
    variants that might appear in the Horde API stats:
    - Canonical: ReadyArt/Broken-Tutu-24B
    - Aphrodite: aphrodite/ReadyArt/Broken-Tutu-24B
    - KoboldCPP: koboldcpp/Broken-Tutu-24B (uses model name only, not org prefix)
    - Legacy KoboldCPP: koboldcpp/ReadyArt_Broken-Tutu-24B (slashes flattened)

    Args:
        canonical_name: The canonical model name from the model reference.

    Returns:
        List of all possible name variants, including the canonical name.

    Example:
        >>> get_model_name_variants("ReadyArt/Broken-Tutu-24B")
        ['ReadyArt/Broken-Tutu-24B', 'aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B', 'koboldcpp/ReadyArt_Broken-Tutu-24B']
    """
    variants = [canonical_name]

    model_name_only = canonical_name.split("/", 1)[1] if "/" in canonical_name else canonical_name
    sanitized_name = canonical_name.replace("/", "_")

    def _append_variant(value: str) -> None:
        if value not in variants:
            variants.append(value)

    for prefix in TEXT_LEGACY_BACKEND_PREFIXES.values():
        if prefix == "aphrodite/":
            _append_variant(f"{prefix}{canonical_name}")
        elif prefix == "koboldcpp/":
            _append_variant(f"{prefix}{model_name_only}")
            if sanitized_name not in {canonical_name, model_name_only}:
                _append_variant(f"{prefix}{sanitized_name}")

    return variants
