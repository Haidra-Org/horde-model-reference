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


def validate_not_backend_prefixed(model_name: str) -> None:
    """Reject model names that start with a known or unknown backend-like prefix.

    Writes should always use the canonical (unprefixed) model name. Backend-prefixed
    duplicates are generated automatically. This function raises ``ValueError`` if the
    name looks like it was submitted with a backend prefix.

    Args:
        model_name: The model name to validate.

    Raises:
        ValueError: If the model name starts with a known backend prefix or matches
            the ``<word>/`` pattern where ``<word>`` is a single lowercase ASCII token
            that could be mistaken for a backend prefix (e.g., ``vllm/SomeModel``).
            Legitimate author-prefixed names like ``ReadyArt/Model`` are allowed
            because the author segment typically contains uppercase letters or digits.

    """
    for backend, prefix in TEXT_LEGACY_BACKEND_PREFIXES.items():
        if model_name.startswith(prefix):
            raise ValueError(
                f"Model name {model_name!r} starts with backend prefix {prefix!r} "
                f"(backend: {backend.value}). Use the canonical name without the prefix; "
                f"backend-prefixed duplicates are generated automatically."
            )


def get_model_name_variants(canonical_name: str) -> list[str]:
    """Get all possible name variants for a canonical model name.

    Given a canonical name like "ReadyArt/Broken-Tutu-24B", returns all possible
    variants that might appear in the Horde API stats:
    - Canonical: ReadyArt/Broken-Tutu-24B
    - Aphrodite: aphrodite/ReadyArt/Broken-Tutu-24B (prefix + full canonical name)
    - KoboldCpp: koboldcpp/Broken-Tutu-24B (prefix + model name only)

    Args:
        canonical_name: The canonical model name from the model reference.

    Returns:
        List of all possible name variants, including the canonical name.

    Example:
        >>> get_model_name_variants("ReadyArt/Broken-Tutu-24B")
        ['ReadyArt/Broken-Tutu-24B', 'aphrodite/ReadyArt/Broken-Tutu-24B', 'koboldcpp/Broken-Tutu-24B']
    """
    variants = [canonical_name]

    model_name_only = canonical_name.split("/", 1)[1] if "/" in canonical_name else canonical_name

    def _append_variant(value: str) -> None:
        if value not in variants:
            variants.append(value)

    for backend, prefix in TEXT_LEGACY_BACKEND_PREFIXES.items():
        if backend == TEXT_BACKENDS.koboldcpp:
            _append_variant(f"{prefix}{model_name_only}")
        elif backend == TEXT_BACKENDS.aphrodite:
            _append_variant(f"{prefix}{canonical_name}")

    return variants
