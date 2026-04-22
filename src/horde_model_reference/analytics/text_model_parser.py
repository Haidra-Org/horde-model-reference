"""Text model name parser for grouping and analysis.

Provides functions to parse text generation model names into structured components
like base name, size, variant, and quantization. Useful for grouping model variants
(e.g., different quant versions of the same base model).

The parser recognises five *primary* parts (base, size, variant, version, quant) and
an open-ended list of *extra* parts — name segments that do not fit any primary
category (date suffixes, descriptive variant words, sub-model identifiers, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import auto
from functools import lru_cache
from typing import TYPE_CHECKING

from loguru import logger
from strenum import StrEnum

if TYPE_CHECKING:
    from horde_model_reference.group_aliases import GroupAliasStore


class ExtraPartType(StrEnum):
    """Classification of a name segment that does not fit any primary part category."""

    DATE = auto()
    """Calendar-style suffix such as ``08-2024`` or ``2024-03``."""

    DESCRIPTOR = auto()
    """A descriptive word that is not in the standard variant list (e.g., ``Transgression``, ``Unslop``)."""

    LEADING_VERSION = auto()
    """A dotted version string that appears *before* the base name (e.g., ``4.2.0-``)."""

    UNKNOWN = auto()
    """Could not be automatically classified."""


@dataclass(frozen=True)
class ExtraNamePart:
    """A name segment that was not classified as a primary part.

    Attributes:
        value: The raw text of the segment.
        position: Zero-based ordinal position within the separator-split name.
        inferred_type: Best-guess classification, or ``UNKNOWN``.

    """

    value: str
    position: int
    inferred_type: ExtraPartType = ExtraPartType.UNKNOWN


@dataclass
class ParsedTextModelName:
    """Structured representation of a parsed text model name.

    Attributes:
        original_name: The original model name as provided.
        base_name: The base model name without size/variant/quant/version info.
        size: Model size if detected (e.g., "7B", "13B", "70B", "7B1").
        variant: Model variant if detected (e.g., "Instruct", "Chat", "Code").
        quant: Quantization type if detected (e.g., "Q4", "Q8", "GGUF").
        version: Model version if detected (e.g., "v0.1", "v2.1").
        normalized_name: A normalized version of the name for comparison.
        extras: Name segments that did not match any primary part category.

    """

    original_name: str
    base_name: str
    size: str | None = None
    variant: str | None = None
    quant: str | None = None
    version: str | None = None
    normalized_name: str | None = None
    extras: list[ExtraNamePart] = field(default_factory=list)


# Common text model size patterns
# Uses lookahead/lookbehind instead of \b because underscore is a word character
# in regex, so \b won't fire at boundaries like `Eclipse_12B`.
SIZE_PATTERNS = [
    r"(?<![a-zA-Z])(\d+\.?\d*[BMK]\d*)(?![a-zA-Z])",  # 7B, 13B, 1.5B, 7B1 (trailing digits), etc.
    r"(?<![a-zA-Z])(\d+x\d+[BMK])(?![a-zA-Z])",  # MoE models: 8x7B, 8x22B
]

# Version patterns (v-prefixed version strings like v0.1, v2.1, V0.420)
VERSION_PATTERNS = [
    r"(?<![a-zA-Z0-9])([vV]\d+(?:\.\d+)+)(?![a-zA-Z0-9])",  # v2.1, V0.420 (with dots)
    r"(?<![a-zA-Z0-9])([vV]\d+)(?![a-zA-Z0-9.])",  # v2, V1 (standalone, not followed by dot)
]

# Common variant indicators
VARIANT_PATTERNS = [
    r"\b(Instruct|Chat|Code|Base|Uncensored|Finetune|FT)\b",
    r"\b(Thinking|Reasoning)\b",
    r"\b(turbo|preview|latest)\b",
]

# Quantization patterns
QUANT_PATTERNS = [
    r"\b(Q[2-8]_K(?:_[SMLH])?)\b",  # Q4_K_M, Q5_K_S, Q6_K (K-quants with optional size)
    r"\b(Q[2-8]_[01])\b",  # Q4_0, Q5_0, Q5_1, Q8_0 (legacy/standard quants)
    r"\b(Q[2-8])\b",  # Q4, Q8 (bare quant indicators)
    r"\b(GGUF|GGML|GPTQ|AWQ|EXL2)\b",
    r"\b(fp16|fp32|int8|int4)\b",
]

# Date-like suffixes that model authors append (e.g., "08-2024", "03-2025", "2024-03")
DATE_PATTERNS = [
    r"(?<![a-zA-Z0-9])(\d{2}-\d{4})(?![a-zA-Z0-9])",  # MM-YYYY (e.g., 08-2024)
    r"(?<![a-zA-Z0-9])(\d{4}-\d{2})(?![a-zA-Z0-9])",  # YYYY-MM (e.g., 2024-08)
    r"(?<![a-zA-Z0-9])(\d{4})(?![a-zA-Z0-9])",  # Standalone 4-digit date code (YYMM/MMDD: 2407, 0414)
]

# Dotted version string that appears *before* the base name (e.g., "4.2.0-Broken-Tutu")
LEADING_VERSION_PATTERN = r"^(\d+\.\d+(?:\.\d+)?)[_\-]"

# Separators to normalize
SEPARATORS = ["-", "_", " ", "."]


@lru_cache(maxsize=2048)
def parse_text_model_name(model_name: str) -> ParsedTextModelName:
    """Parse a text model name into structured components.

    Attempts to extract base name, size, variant, quantization, version,
    and *extra* segments from a model name using regex patterns.

    Extraction order: leading version → size → version → quant → variant → date.
    Segments that remain after all primary extractions are classified as extras.

    Args:
        model_name: The full model name to parse.

    Returns:
        ParsedTextModelName with extracted components and extras.

    Example:
        >>> parsed = parse_text_model_name("Llama-3-8B-Instruct-Q4_K_M")
        >>> parsed.base_name
        'Llama-3'
        >>> parsed.size
        '8B'
        >>> parsed = parse_text_model_name("c4ai-command-r-08-2024")
        >>> parsed.base_name
        'c4ai-command-r'
        >>> parsed.extras[0].inferred_type
        <ExtraPartType.DATE: 'date'>

    """
    logger.trace(f"Parsing text model name: {model_name}")

    name_parts = model_name
    size = None
    variant = None
    quant = None
    version = None
    extras: list[ExtraNamePart] = []

    # --- Extract leading dotted-version (e.g., "4.2.0-Broken-Tutu") ---
    leading_match = re.match(LEADING_VERSION_PATTERN, name_parts)
    if leading_match:
        leading_ver = leading_match.group(1)
        extras.append(
            ExtraNamePart(
                value=leading_ver,
                position=0,
                inferred_type=ExtraPartType.LEADING_VERSION,
            ),
        )
        name_parts = name_parts[leading_match.end() :]
        logger.trace(f"Extracted leading version extra: {leading_ver}")

    # --- Extract size ---
    for pattern in SIZE_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            size = match.group(1).upper()
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.trace(f"Extracted size: {size}")
            break

    # --- Extract version (after size so v-prefixed versions aren't confused with sizes) ---
    for pattern in VERSION_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            version = match.group(1)
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.trace(f"Extracted version: {version}")
            break

    # --- Extract quantization ---
    for pattern in QUANT_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            quant = match.group(1).upper()
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.trace(f"Extracted quant: {quant}")
            break

    # --- Extract variant ---
    for pattern in VARIANT_PATTERNS:
        match = re.search(pattern, name_parts, re.IGNORECASE)
        if match:
            variant = match.group(1)
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.trace(f"Extracted variant: {variant}")
            break

    # --- Extract date suffixes as extras ---
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, name_parts)
        if match:
            date_value = match.group(1)
            extras.append(
                ExtraNamePart(
                    value=date_value,
                    position=_count_separators_before(model_name, match.start()),
                    inferred_type=ExtraPartType.DATE,
                ),
            )
            name_parts = name_parts[: match.start()] + name_parts[match.end() :]
            logger.trace(f"Extracted date extra: {date_value}")
            break

    # Clean up base name — collapse repeated separators and strip edges
    base_name = name_parts
    for sep in SEPARATORS:
        while sep + sep in base_name:
            base_name = base_name.replace(sep + sep, sep)

    base_name = base_name.strip("-_ .")

    if not base_name:
        base_name = model_name
        logger.debug(f"Could not extract base name, using original: {base_name}")
    else:
        logger.trace(f"Extracted base name: {base_name}")

    normalized = normalize_model_name(model_name)

    return ParsedTextModelName(
        original_name=model_name,
        base_name=base_name,
        size=size,
        variant=variant,
        quant=quant,
        version=version,
        normalized_name=normalized,
        extras=extras,
    )


def _count_separators_before(text: str, position: int) -> int:
    """Count how many separator-delimited segments occur before *position* in *text*."""
    prefix = text[:position]
    count = 0
    for char in prefix:
        if char in "-_. ":
            count += 1
    return count


@lru_cache(maxsize=2048)
def get_base_model_name(model_name: str) -> str:
    """Get the base model name for grouping purposes.

    Extracts just the base name without backend prefix, author prefix,
    size, variant, or quantization info. Useful for grouping different
    variants of the same model together.

    Args:
        model_name: The full model name (may include backend and author prefixes).

    Returns:
        The base model name without prefixes.

    Example:
        >>> get_base_model_name("Llama-3-8B-Instruct-Q4_K_M")
        "Llama-3"
        >>> get_base_model_name("Mistral-7B-v0.1")
        "Mistral"
        >>> get_base_model_name("koboldcpp/sophosympatheia/StrawberryLemonade-L3-70B-v1.2")
        "StrawberryLemonade-L3-v1"
        >>> get_base_model_name("aphrodite/ReadyArt/Broken-Tutu-24B")
        "Broken-Tutu"

    """
    from horde_model_reference.text_backend_names import strip_backend_prefix

    # First strip backend prefix (e.g., "koboldcpp/", "aphrodite/")
    name_without_backend = strip_backend_prefix(model_name)

    # Then strip author prefix if present (e.g., "sophosympatheia/", "ReadyArt/")
    # Author prefix is the first part before "/" if there's one remaining
    if "/" in name_without_backend:
        name_without_author = name_without_backend.split("/", 1)[1]
    else:
        name_without_author = name_without_backend

    parsed = parse_text_model_name(name_without_author)
    return parsed.base_name


@lru_cache(maxsize=2048)
def normalize_model_name(model_name: str) -> str:
    """Normalize a model name for case-insensitive comparison.

    Converts to lowercase and normalizes separators.

    Args:
        model_name: The model name to normalize.

    Returns:
        Normalized model name.

    Example:
        >>> normalize_model_name("Llama-3-8B-Instruct")
        "llama_3_8b_instruct"

    """
    normalized = model_name.lower()

    for sep in ["-", " ", "."]:
        normalized = normalized.replace(sep, "_")

    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


@dataclass
class TextModelGroup:
    """Represents a group of text model variants sharing the same base model.

    Attributes:
        base_name: The base model name.
        variants: List of full model names that are variants of the base model.

    """

    base_name: str
    variants: list[str]


def group_text_models_by_base(
    model_names: list[str],
    *,
    alias_store: GroupAliasStore | None = None,
) -> dict[str, TextModelGroup]:
    """Group text model names by their base model.

    Groups variants of the same model together based on extracted base names.
    When *alias_store* is provided, alias-resolved canonical names are used
    as group keys.

    Args:
        model_names: List of model names to group.
        alias_store: Optional alias store for resolving group names.

    Returns:
        Dictionary mapping base names to lists of full model names.

    Example:
        >>> models = [
        ...     "Llama-3-8B-Instruct",
        ...     "Llama-3-8B-Instruct-Q4",
        ...     "Llama-3-70B-Instruct",
        ...     "Mistral-7B-v0.1",
        ... ]
        >>> grouped = group_text_models_by_base(models)
        >>> print(grouped)
        {
            "Llama-3": ["Llama-3-8B-Instruct", "Llama-3-8B-Instruct-Q4", "Llama-3-70B-Instruct"],
            "Mistral": ["Mistral-7B-v0.1"]
        }

    """
    grouped: dict[str, list[str]] = {}

    for model_name in model_names:
        base_name = get_base_model_name(model_name)
        if alias_store is not None:
            base_name = alias_store.resolve(base_name)

        if base_name not in grouped:
            grouped[base_name] = []

        grouped[base_name].append(model_name)

    logger.debug(f"Grouped {len(model_names)} models into {len(grouped)} base models")

    return {
        base_name: TextModelGroup(
            base_name=base_name,
            variants=variants,
        )
        for base_name, variants in grouped.items()
    }


@lru_cache(maxsize=2048)
def is_quantized_variant(model_name: str) -> bool:
    """Check if a model name indicates a quantized variant.

    Args:
        model_name: The model name to check.

    Returns:
        True if the model appears to be a quantized variant.

    Example:
        >>> is_quantized_variant("Llama-3-8B-Instruct-Q4_K_M")
        True
        >>> is_quantized_variant("Llama-3-8B-Instruct")
        False

    """
    parsed = parse_text_model_name(model_name)
    return parsed.quant is not None


@lru_cache(maxsize=2048)
def get_model_size(model_name: str) -> str | None:
    """Extract the model size from a model name.

    Args:
        model_name: The model name to parse.

    Returns:
        The model size (e.g., "7B", "13B") or None if not found.

    Example:
        >>> get_model_size("Llama-3-8B-Instruct")
        "8B"
        >>> get_model_size("GPT-4")
        None

    """
    parsed = parse_text_model_name(model_name)
    return parsed.size


@lru_cache(maxsize=2048)
def get_model_variant(model_name: str) -> str | None:
    """Extract the model variant from a model name.

    Args:
        model_name: The model name to parse.

    Returns:
        The model variant (e.g., "Instruct", "Chat") or None if not found.

    Example:
        >>> get_model_variant("Llama-3-8B-Instruct")
        "Instruct"
        >>> get_model_variant("Llama-3-8B")
        None

    """
    parsed = parse_text_model_name(model_name)
    return parsed.variant


@dataclass
class NameFormatSchema:
    """Describes the naming convention inferred from a group of models.

    Used by compose_name to produce names consistent with existing group members.

    ``extra_parts`` lists labels for name segments that do not fit any primary
    category (e.g., ``["date"]``).  The labels are free-form strings that help
    admins understand what the segment represents.
    """

    separator: str = "-"
    part_order: list[str] = field(default_factory=lambda: ["base", "size", "variant", "version", "quant"])
    author_included: bool = False
    common_author: str | None = None
    template: str = "{base}-{size}"
    extra_parts: list[str] = field(default_factory=list)


def _detect_separator(names: list[str]) -> str:
    """Detect the dominant separator in model names (ignoring separators within quant tokens)."""
    hyphen_count = 0
    underscore_count = 0

    for name in names:
        cleaned = name
        for pattern in QUANT_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        hyphen_count += cleaned.count("-")
        underscore_count += cleaned.count("_")

    return "_" if underscore_count > hyphen_count else "-"


def _detect_part_order(original: str, parsed: ParsedTextModelName) -> list[str]:
    """Detect the order of parts in a model name by their position in the original string.

    Includes extra-part labels (prefixed with ``extra:``) so that the inferred
    template reflects the full name structure.
    """
    parts: dict[str, str] = {}
    if parsed.base_name:
        parts["base"] = parsed.base_name
    if parsed.size:
        parts["size"] = parsed.size
    if parsed.variant:
        parts["variant"] = parsed.variant
    if parsed.version:
        parts["version"] = parsed.version
    if parsed.quant:
        parts["quant"] = parsed.quant
    for extra in parsed.extras:
        label = f"extra:{extra.inferred_type.value}"
        parts[label] = extra.value

    positions: dict[str, int] = {}
    original_lower = original.lower()
    for part_name, part_value in parts.items():
        pos = original_lower.find(part_value.lower())
        if pos >= 0:
            positions[part_name] = pos

    return [name for name, _ in sorted(positions.items(), key=lambda x: x[1])]


def infer_name_format(member_names: list[str]) -> NameFormatSchema:
    """Infer the naming convention from existing group members.

    Analyzes separators, part ordering, and author inclusion across
    all member names to produce a schema that can drive consistent
    name composition for new variations.

    Args:
        member_names: List of model names belonging to the same group.

    Returns:
        NameFormatSchema describing the group's naming convention.

    """
    if not member_names:
        return NameFormatSchema()

    # Separate author prefixes
    authors: set[str] = set()
    names_without_author: list[str] = []
    for name in member_names:
        if "/" in name:
            author, _, rest = name.partition("/")
            authors.add(author)
            names_without_author.append(rest)
        else:
            names_without_author.append(name)

    author_included = len(authors) > 0
    common_author = authors.pop() if len(authors) == 1 else None

    separator = _detect_separator(names_without_author)

    # Detect part order from the most-complete member (most extracted parts)
    parsed_members = [parse_text_model_name(n) for n in names_without_author]
    richest = max(
        zip(names_without_author, parsed_members, strict=False),
        key=lambda pair: (
            sum(1 for v in [pair[1].size, pair[1].variant, pair[1].version, pair[1].quant] if v) + len(pair[1].extras)
        ),
    )
    part_order = _detect_part_order(richest[0], richest[1])

    # Collect unique extra-part labels across all members
    seen_extra_labels: list[str] = []
    for pm in parsed_members:
        for extra in pm.extras:
            label = extra.inferred_type.value
            if label not in seen_extra_labels:
                seen_extra_labels.append(label)

    # Build human-readable template
    template_parts: list[str] = []
    if author_included:
        template_parts.append("{author}/")
    for i, part in enumerate(part_order):
        if i == 0:
            template_parts.append(f"{{{part}}}")
        else:
            template_parts.append(f"{separator}{{{part}}}")
    template = "".join(template_parts)

    return NameFormatSchema(
        separator=separator,
        part_order=part_order,
        author_included=author_included,
        common_author=common_author,
        template=template,
        extra_parts=seen_extra_labels,
    )


@dataclass
class TextModelGroupSummary:
    """Aggregated metadata for a group of text model variants."""

    group_name: str
    member_count: int
    available_sizes: list[str]
    available_quants: list[str]
    common_baseline: str | None
    any_nsfw: bool
    any_has_description: bool
    merged_tags: list[str]
    name_format: NameFormatSchema


def compute_group_summaries(
    models_dict: dict[str, dict[str, object]],
) -> dict[str, TextModelGroupSummary]:
    """Compute aggregated summaries for each text model group.

    Expects models_dict entries to already have ``text_model_group`` set.
    Parses each model name to extract sizes, quants, etc. and aggregates
    metadata fields (baseline, nsfw, tags, description) across members.

    Args:
        models_dict: Mapping of model_name → model_data dicts (mutated legacy JSON).

    Returns:
        Mapping of group_name → TextModelGroupSummary.

    """
    # Group model names by their text_model_group value
    groups: dict[str, list[str]] = {}
    for model_name, model_data in models_dict.items():
        group = str(model_data.get("text_model_group", model_name))
        if group not in groups:
            groups[group] = []
        groups[group].append(model_name)

    summaries: dict[str, TextModelGroupSummary] = {}
    for group_name, member_names in groups.items():
        parsed = [parse_text_model_name(name) for name in member_names]

        sizes: set[str] = set()
        quants: set[str] = set()
        baselines: set[str] = set()
        any_nsfw = False
        any_has_description = False
        merged_tags: set[str] = set()

        for p, mname in zip(parsed, member_names, strict=False):
            mdata = models_dict[mname]
            if p.size:
                sizes.add(p.size)
            if p.quant:
                quants.add(p.quant)
            baseline = mdata.get("baseline")
            if baseline:
                baselines.add(str(baseline))
            if mdata.get("nsfw"):
                any_nsfw = True
            if mdata.get("description"):
                any_has_description = True
            tags = mdata.get("tags")
            if isinstance(tags, list):
                merged_tags.update(str(t) for t in tags)

        format_schema = infer_name_format(member_names)

        summaries[group_name] = TextModelGroupSummary(
            group_name=group_name,
            member_count=len(member_names),
            available_sizes=sorted(sizes),
            available_quants=sorted(quants),
            common_baseline=baselines.pop() if len(baselines) == 1 else None,
            any_nsfw=any_nsfw,
            any_has_description=any_has_description,
            merged_tags=sorted(merged_tags),
            name_format=format_schema,
        )

    return summaries
