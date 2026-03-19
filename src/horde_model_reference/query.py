"""Fluent query builder for model reference records.

Provides a read-only, lazy-evaluated query API over cached model records.
All filtering, ordering, and pagination happens in-memory on the already-loaded
Pydantic models — no new storage or network calls are introduced.

Usage::

    from horde_model_reference import ModelReferenceManager

    manager = ModelReferenceManager()
    results = (
        manager.query("image_generation")
        .where(nsfw=False, baseline="stable_diffusion_xl")
        .tags_any(["realistic", "generalist"])
        .order_by("size_on_disk_bytes")
        .limit(10)
        .to_list()
    )
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any, Literal, Protocol, Self, overload, runtime_checkable

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    TEXT_BACKENDS,
)
from horde_model_reference.model_reference_records import (
    ControlNetModelRecord,
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)
from horde_model_reference.query_fields import OrderSpec, Predicate
from horde_model_reference.text_backend_names import (
    TEXT_LEGACY_BACKEND_PREFIXES,
    has_legacy_text_backend_prefix,
)

# ---------------------------------------------------------------------------
# Field name type aliases
# ---------------------------------------------------------------------------

type GenericFieldName = Literal[
    "record_type",
    "name",
    "description",
    "version",
    "finetune_series",
    "metadata",
    "config",
    "model_classification",
]

type ImageGenFieldName = Literal[
    "record_type",
    "name",
    "description",
    "version",
    "finetune_series",
    "metadata",
    "config",
    "model_classification",
    "inpainting",
    "baseline",
    "optimization",
    "tags",
    "showcases",
    "min_bridge_version",
    "trigger",
    "homepage",
    "nsfw",
    "style",
    "requirements",
    "size_on_disk_bytes",
]

type TextGenFieldName = Literal[
    "record_type",
    "name",
    "description",
    "version",
    "finetune_series",
    "metadata",
    "config",
    "model_classification",
    "baseline",
    "parameters_count",
    "nsfw",
    "style",
    "display_name",
    "url",
    "tags",
    "instruct_format",
    "settings",
    "text_model_group",
]

type ControlNetFieldName = Literal[
    "record_type",
    "name",
    "description",
    "version",
    "finetune_series",
    "metadata",
    "config",
    "model_classification",
    "controlnet_style",
]

# ---------------------------------------------------------------------------
# Protocols for record capabilities
# ---------------------------------------------------------------------------


@runtime_checkable
class HasTags(Protocol):
    """Protocol for record types that have a ``tags`` field.

    Satisfied by ``ImageGenerationModelRecord``, ``TextGenerationModelRecord``,
    ``VideoGenerationModelRecord``, and ``AudioGenerationModelRecord``.
    """

    tags: list[str] | None


@runtime_checkable
class HasBaseline(Protocol):
    """Protocol for record types that have a ``baseline`` field.

    Satisfied by ``ImageGenerationModelRecord``, ``TextGenerationModelRecord``,
    ``VideoGenerationModelRecord``, and ``AudioGenerationModelRecord``.
    """

    baseline: str | None


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

_COMPARISON_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
    "ne": operator.ne,
    "in": lambda val, choices: val in choices,
    "contains": lambda val, item: item in val,
}


def _resolve_field_value(record: GenericModelRecord, field_path: str) -> object:
    """Resolve a nested field path like ``finetune_series__name`` or raise on missing segments."""
    obj: object = record
    for part in field_path.split("__"):
        if obj is None:
            raise ValueError(f"Field path '{field_path}' is missing segment '{part}' (encountered None)")

        if isinstance(obj, dict):
            if part not in obj:
                raise ValueError(f"Field path '{field_path}' is missing key '{part}' on intermediate dict segment")
            obj = obj[part]
            continue

        if not hasattr(obj, part):
            raise ValueError(f"Field path '{field_path}' is missing attribute '{part}' on {type(obj).__name__}")

        obj = getattr(obj, part)

    return obj


def _validate_field_exists(record_type: type[GenericModelRecord], field_name: str) -> None:
    """Validate that *field_name* (top-level segment) exists on the Pydantic model."""
    top_level = field_name.split("__")[0]
    all_fields = record_type.model_fields
    if top_level not in all_fields:
        valid = sorted(all_fields.keys())
        raise ValueError(f"Field '{top_level}' does not exist on {record_type.__name__}. Valid fields: {valid}")


def _is_non_string_iterable(value: object) -> bool:
    """Return True when *value* is an iterable but not a string/bytes."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _to_hashable(field: str, value: object) -> Hashable:
    """Convert *value* into a hashable form or raise a helpful error."""
    if isinstance(value, list):
        candidate: object = tuple(value)
    else:
        candidate = value

    try:
        hash(candidate)
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Field '{field}' contains unhashable value of type {type(value).__name__}; "
            "cannot use for distinct/group_by"
        ) from exc

    return candidate


# ---------------------------------------------------------------------------
# Base query builder
# ---------------------------------------------------------------------------


class ModelQuery[T: GenericModelRecord, F: str]:
    """Lazy, immutable query builder over a sequence of model records.

    Every fluent method returns a **new** instance (via ``Self``) so that
    partially-built queries can be safely reused.  Subclasses automatically
    preserve their concrete type through the chain thanks to ``type(self)``
    dispatch in ``_clone``.
    """

    _records: Sequence[T]
    _record_type: type[T]
    _predicates: Sequence[Callable[[T], bool]]
    _sort_key: str | None
    _sort_descending: bool
    _offset_value: int
    _limit_value: int | None

    def __init__(  # noqa: D107
        self,
        records: Sequence[T],
        record_type: type[T],
        *,
        predicates: Sequence[Callable[[T], bool]] | None = None,
        sort_key: str | None = None,
        sort_descending: bool = False,
        offset_value: int = 0,
        limit_value: int | None = None,
    ) -> None:
        self._records = records
        self._record_type = record_type
        self._predicates = list(predicates) if predicates else []
        self._sort_key = sort_key
        self._sort_descending = sort_descending
        self._offset_value = offset_value
        self._limit_value = limit_value

    def _clone(
        self,
        records: Sequence[T] | None = None,
        record_type: type[T] | None = None,
        predicates: Sequence[Callable[[T], bool]] | None = None,
        sort_key: str | None = None,
        sort_descending: bool | None = None,
        offset_value: int | None = None,
        limit_value: int | None = None,
    ) -> Self:
        """Create a shallow copy with optional overrides.

        Uses ``type(self)`` so that subclasses (``TextModelQuery``,
        ``ImageGenerationQuery``, etc.) automatically get back their own
        concrete type without needing to override this method.
        """
        return type(self)(
            records=records if records is not None else self._records,
            record_type=record_type if record_type is not None else self._record_type,
            predicates=predicates if predicates is not None else list(self._predicates),
            sort_key=sort_key if sort_key is not None else self._sort_key,
            sort_descending=sort_descending if sort_descending is not None else self._sort_descending,
            offset_value=offset_value if offset_value is not None else self._offset_value,
            limit_value=limit_value if limit_value is not None else self._limit_value,
        )

    # ------------------------------------------------------------------
    # Equality / comparison filters
    # ------------------------------------------------------------------

    def where(self, *predicates: Predicate, **kwargs: object) -> Self:
        """Filter records by field equality, comparison operators, or ``Predicate`` objects.

        Supports three styles that can be freely mixed in one call:

        1. **Keyword equality/comparison** (Django-style suffixes):
           ``where(nsfw=False, size_on_disk_bytes__gt=1_000_000_000)``
        2. **Field-ref predicates** (typed DSL):
           ``where(ImageF.nsfw == false, ImageF.size_on_disk_bytes > 1_000_000_000)``
        3. **Composed predicates** (boolean algebra):
           ``where((ImageF.nsfw == false) & (ImageF.baseline == "stable_diffusion_xl"))``

        Args:
            *predicates: Zero or more ``Predicate`` objects (from ``FieldRef``
                comparisons or manual construction).
            **kwargs: Field names (with optional operator suffix) mapped to
                the value(s) to compare against.

        Returns:
            A new query with the additional predicates applied.
        """
        new_preds: list[Callable[[T], bool]] = list(self._predicates)
        new_preds.extend(predicates)

        for raw_key, value in kwargs.items():
            field_name, op_name = self._parse_key(raw_key)
            _validate_field_exists(self._record_type, field_name)

            if op_name is None and _is_non_string_iterable(value):
                op_name = "in"

            if op_name is None:
                pred = self._eq_predicate(field_name, value)
            else:
                if op_name not in _COMPARISON_OPS:
                    raise ValueError(
                        f"Unknown operator '{op_name}'. Valid operators: {sorted(_COMPARISON_OPS.keys())}"
                    )
                pred = self._cmp_predicate(field_name, op_name, value)

            new_preds.append(pred)

        return self._clone(predicates=new_preds)

    def where_classification(
        self,
        *,
        domain: MODEL_DOMAIN | None = None,
        purpose: MODEL_PURPOSE | None = None,
    ) -> Self:
        """Filter records by their ``model_classification``."""
        new_preds: list[Callable[[T], bool]] = list(self._predicates)

        def _classification_pred(record: T) -> bool:
            cls = record.model_classification
            if domain is not None and cls.domain != domain:
                return False
            return not (purpose is not None and cls.purpose != purpose)

        new_preds.append(_classification_pred)
        return self._clone(predicates=new_preds)

    # ------------------------------------------------------------------
    # Tag helpers
    # ------------------------------------------------------------------

    def tags_any(self, tags: Iterable[str]) -> Self:
        """Keep records whose ``tags`` field contains **any** of *tags*."""
        tag_set = set(tags)
        _validate_field_exists(self._record_type, "tags")

        def _pred(record: T) -> bool:
            record_tags: list[str] | None = getattr(record, "tags", None)
            if not record_tags:
                return False
            return bool(tag_set & set(record_tags))

        return self._clone(predicates=[*self._predicates, _pred])

    def tags_all(self, tags: Iterable[str]) -> Self:
        """Keep records whose ``tags`` field contains **all** of *tags*."""
        tag_set = set(tags)
        _validate_field_exists(self._record_type, "tags")

        def _pred(record: T) -> bool:
            record_tags: list[str] | None = getattr(record, "tags", None)
            if not record_tags:
                return False
            return tag_set <= set(record_tags)

        return self._clone(predicates=[*self._predicates, _pred])

    def tags_none(self, tags: Iterable[str]) -> Self:
        """Exclude records whose ``tags`` field contains **any** of *tags*."""
        tag_set = set(tags)
        _validate_field_exists(self._record_type, "tags")

        def _pred(record: T) -> bool:
            record_tags: list[str] | None = getattr(record, "tags", None)
            if not record_tags:
                return True
            return not (tag_set & set(record_tags))

        return self._clone(predicates=[*self._predicates, _pred])

    # ------------------------------------------------------------------
    # Arbitrary predicate
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[T], bool]) -> Self:
        """Apply an arbitrary predicate function."""
        return self._clone(predicates=[*self._predicates, predicate])

    # ------------------------------------------------------------------
    # Ordering & pagination
    # ------------------------------------------------------------------

    @overload
    def order_by(self, field: OrderSpec) -> Self: ...

    @overload
    def order_by(self, field: F, *, descending: bool = False) -> Self: ...

    def order_by(self, field: F | OrderSpec, *, descending: bool = False) -> Self:
        """Sort results by *field*; raises ``ValueError`` if values are not comparable.

        Accepts either a field name string or an ``OrderSpec`` from the field
        descriptor DSL (e.g. ``ImageF.size_on_disk_bytes.desc()``).
        """
        if isinstance(field, OrderSpec):
            _validate_field_exists(self._record_type, field.field)
            return self._clone(sort_key=field.field, sort_descending=field.descending)
        _validate_field_exists(self._record_type, field)
        return self._clone(sort_key=field, sort_descending=descending)

    def limit(self, n: int) -> Self:
        """Limit the number of returned results."""
        return self._clone(limit_value=n)

    def offset(self, n: int) -> Self:
        """Skip the first *n* results."""
        return self._clone(offset_value=n)

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def _execute(self) -> list[T]:
        """Apply all predicates, sorting, and pagination."""
        result: list[T] = [r for r in self._records if all(p(r) for p in self._predicates)]

        if self._sort_key is not None:
            key_field = self._sort_key

            def _sort_key(record: T) -> tuple[int, object]:
                val = _resolve_field_value(record, key_field)
                if val is None:
                    return (1, "")
                return (0, val)

            try:
                result.sort(key=_sort_key, reverse=self._sort_descending)
            except TypeError as exc:  # pragma: no cover - exercised via tests
                value_types = {type(_resolve_field_value(r, key_field)).__name__ for r in result}
                raise ValueError(
                    "Cannot order by field "
                    f"'{key_field}' because values are not mutually comparable: {sorted(value_types)}"
                ) from exc

        if self._offset_value:
            result = result[self._offset_value :]
        if self._limit_value is not None:
            result = result[: self._limit_value]

        return result

    def to_list(self) -> list[T]:
        """Execute the query and return results as a list."""
        return self._execute()

    def first(self) -> T | None:
        """Execute the query and return the first result, or ``None``."""
        results = self.limit(1)._execute()
        return results[0] if results else None

    def count(self) -> int:
        """Execute the query and return the number of matching records."""
        return len(self._execute())

    def distinct(self, field: F) -> list[object]:
        """Return unique values of *field* across matching records (raises on unhashable values)."""
        _validate_field_exists(self._record_type, field)
        seen: set[Hashable] = set()
        result: list[object] = []
        for record in self._execute():
            val = _resolve_field_value(record, field)
            hashable_val = _to_hashable(field, val)
            if hashable_val not in seen:
                seen.add(hashable_val)
                result.append(val)
        return result

    def group_by(self, field: F) -> dict[Hashable, list[T]]:
        """Group matching records by *field* value.

        Returns:
            A dict mapping each distinct value to the list of records with that value.
        """
        _validate_field_exists(self._record_type, field)
        groups: dict[Hashable, list[T]] = {}
        for record in self._execute():
            val = _resolve_field_value(record, field)
            key = _to_hashable(field, val)
            groups.setdefault(key, []).append(record)
        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_key(raw_key: str) -> tuple[str, str | None]:
        """Split ``field__op`` into ``(field, op)`` or ``(field, None)``."""
        for op_name in _COMPARISON_OPS:
            suffix = f"__{op_name}"
            if raw_key.endswith(suffix):
                return raw_key[: -len(suffix)], op_name

        if "__" in raw_key:
            parts = raw_key.rsplit("__", maxsplit=1)
            candidate_op = parts[1]
            if candidate_op in _COMPARISON_OPS:
                return parts[0], candidate_op

        return raw_key, None

    @staticmethod
    def _eq_predicate(field_name: str, value: object) -> Callable[[GenericModelRecord], bool]:
        """Build an equality predicate for *field_name*."""

        def _pred(record: GenericModelRecord) -> bool:
            return _resolve_field_value(record, field_name) == value

        return _pred

    @staticmethod
    def _cmp_predicate(field_name: str, op_name: str, value: object) -> Callable[[GenericModelRecord], bool]:
        """Build a comparison predicate for *field_name* using *op_name*."""
        cmp_fn = _COMPARISON_OPS[op_name]

        def _pred(record: GenericModelRecord) -> bool:
            field_val = _resolve_field_value(record, field_name)
            if field_val is None:
                return False
            if op_name == "in":
                if not _is_non_string_iterable(value):
                    raise ValueError("The '__in' operator requires a non-string iterable value.")
                return cmp_fn(field_val, value)
            if op_name == "contains":
                if not _is_non_string_iterable(field_val):
                    return False
                return cmp_fn(field_val, value)
            return cmp_fn(field_val, value)

        return _pred


# ---------------------------------------------------------------------------
# Image generation query builder
# ---------------------------------------------------------------------------


class ImageGenerationQuery(ModelQuery[ImageGenerationModelRecord, ImageGenFieldName]):
    """Query builder with image-generation-specific helpers.

    Adds typed convenience methods for common image model filters
    (baseline, NSFW, inpainting) and overloaded field-name parameters
    that give IDE autocomplete for ``ImageGenerationModelRecord`` fields.
    """

    # ------------------------------------------------------------------
    # Baseline filters
    # ------------------------------------------------------------------

    def for_baseline(self, baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> Self:
        """Keep only models with the given *baseline*."""

        def _pred(record: ImageGenerationModelRecord) -> bool:
            return record.baseline == baseline

        return self._clone(predicates=[*self._predicates, _pred])

    # ------------------------------------------------------------------
    # NSFW / inpainting convenience filters
    # ------------------------------------------------------------------

    def only_nsfw(self) -> Self:
        """Keep only NSFW models."""

        def _pred(record: ImageGenerationModelRecord) -> bool:
            return record.nsfw

        return self._clone(predicates=[*self._predicates, _pred])

    def exclude_nsfw(self) -> Self:
        """Remove NSFW models."""

        def _pred(record: ImageGenerationModelRecord) -> bool:
            return not record.nsfw

        return self._clone(predicates=[*self._predicates, _pred])

    def only_inpainting(self) -> Self:
        """Keep only inpainting models."""

        def _pred(record: ImageGenerationModelRecord) -> bool:
            return bool(record.inpainting)

        return self._clone(predicates=[*self._predicates, _pred])

    def exclude_inpainting(self) -> Self:
        """Remove inpainting models."""

        def _pred(record: ImageGenerationModelRecord) -> bool:
            return not record.inpainting

        return self._clone(predicates=[*self._predicates, _pred])


# ---------------------------------------------------------------------------
# Text generation query builder
# ---------------------------------------------------------------------------


class TextModelQuery(ModelQuery[TextGenerationModelRecord, TextGenFieldName]):
    """Query builder with text-generation-specific helpers.

    Adds filtering by backend prefix, quantization status, and grouping
    by base model name.  Every fluent method returns ``Self`` so the full
    chain stays type-safe.
    """

    # ------------------------------------------------------------------
    # Backend prefix filters
    # ------------------------------------------------------------------

    def for_backend(self, backend: TEXT_BACKENDS) -> Self:
        """Keep only models whose name starts with the legacy prefix for *backend*."""
        prefix = TEXT_LEGACY_BACKEND_PREFIXES[backend]

        def _pred(record: TextGenerationModelRecord) -> bool:
            return record.name.startswith(prefix)

        return self._clone(predicates=[*self._predicates, _pred])

    def exclude_backend_variations(self) -> Self:
        """Remove models that carry any legacy backend prefix."""

        def _pred(record: TextGenerationModelRecord) -> bool:
            return not has_legacy_text_backend_prefix(record.name)

        return self._clone(predicates=[*self._predicates, _pred])

    # ------------------------------------------------------------------
    # Quantization filters
    # ------------------------------------------------------------------

    def only_quantized(self) -> Self:
        """Keep only quantized model variants."""
        from horde_model_reference.analytics.text_model_parser import is_quantized_variant

        def _pred(record: TextGenerationModelRecord) -> bool:
            return is_quantized_variant(record.name)

        return self._clone(predicates=[*self._predicates, _pred])

    def exclude_quantized(self) -> Self:
        """Remove quantized model variants."""
        from horde_model_reference.analytics.text_model_parser import is_quantized_variant

        def _pred(record: TextGenerationModelRecord) -> bool:
            return not is_quantized_variant(record.name)

        return self._clone(predicates=[*self._predicates, _pred])

    # ------------------------------------------------------------------
    # Terminal: group by base model
    # ------------------------------------------------------------------

    def group_by_base_model(self) -> dict[str, list[TextGenerationModelRecord]]:
        """Group matching records by their parsed base model name.

        Returns:
            A dict mapping each base model name to the list of matching records.
        """
        from horde_model_reference.analytics.text_model_parser import get_base_model_name

        groups: dict[str, list[TextGenerationModelRecord]] = {}
        for record in self._execute():
            base = get_base_model_name(record.name)
            groups.setdefault(base, []).append(record)
        return groups


# ---------------------------------------------------------------------------
# ControlNet query builder
# ---------------------------------------------------------------------------


class ControlNetQuery(ModelQuery[GenericModelRecord, ControlNetFieldName]):
    """Query builder for ControlNet models.

    Adds typed convenience methods for filtering by ControlNet style and grouping by
    it.  Every fluent method returns ``Self`` so the full chain stays type-safe.
    """

    def for_style(self, style: str) -> Self:
        """Keep only ControlNet models with the given *style*."""

        def _pred(record: GenericModelRecord) -> bool:
            return getattr(record, "controlnet_style", None) == style

        return self._clone(predicates=[*self._predicates, _pred])

    def group_by_style(self) -> dict[str, list[GenericModelRecord]]:
        """Group matching records by their ControlNet style.

        Returns:
            A dict mapping each style to the list of matching records.
        """
        groups: dict[str, list[GenericModelRecord]] = {}
        for record in self._execute():
            style = getattr(record, "controlnet_style", None)
            if style is not None:
                groups.setdefault(style, []).append(record)
        return groups


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_query[T: GenericModelRecord](
    records: dict[str, T],
    record_type: type[T],
) -> ModelQuery[T, str]:
    """Create a ``ModelQuery`` from a name-to-record mapping.

    Args:
        records: The mapping returned by ``ModelReferenceManager.get_model_reference()``.
        record_type: The Pydantic record type for field validation.

    Returns:
        A fresh ``ModelQuery`` ready for chaining.
    """
    return ModelQuery(
        records=list(records.values()),
        record_type=record_type,
    )


def build_image_query(
    records: dict[str, ImageGenerationModelRecord],
) -> ImageGenerationQuery:
    """Create an ``ImageGenerationQuery`` from a name-to-record mapping.

    Args:
        records: The mapping returned by ``ModelReferenceManager.get_model_reference()``
            for the ``image_generation`` category.

    Returns:
        A fresh ``ImageGenerationQuery`` ready for chaining.
    """
    return ImageGenerationQuery(
        records=list(records.values()),
        record_type=ImageGenerationModelRecord,
    )


def build_text_query(
    records: dict[str, TextGenerationModelRecord],
) -> TextModelQuery:
    """Create a ``TextModelQuery`` from a name-to-record mapping.

    Args:
        records: The mapping returned by ``ModelReferenceManager.get_model_reference()``
            for the ``text_generation`` category.

    Returns:
        A fresh ``TextModelQuery`` ready for chaining.
    """
    return TextModelQuery(
        records=list(records.values()),
        record_type=TextGenerationModelRecord,
    )


def build_controlnet_query(
    records: dict[str, ControlNetModelRecord],
) -> ControlNetQuery:
    """Create a ``ControlNetQuery`` from a name-to-record mapping.

    Args:
        records: The mapping returned by ``ModelReferenceManager.get_model_reference()``
            for the ``controlnet`` category.

    Returns:
        A fresh ``ControlNetQuery`` ready for chaining.
    """
    return ControlNetQuery(
        records=list(records.values()),
        record_type=ControlNetModelRecord,
    )


def build_cross_category_query(
    all_references: dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord]],
) -> ModelQuery[GenericModelRecord, str]:
    """Create a ``ModelQuery`` spanning all categories.

    Args:
        all_references: Mapping returned by ``ModelReferenceManager.get_all_model_references()``.

    Returns:
        A ``ModelQuery[GenericModelRecord]`` over every record in every category.
    """
    all_records: list[GenericModelRecord] = []
    for category_records in all_references.values():
        all_records.extend(category_records.values())
    return ModelQuery(
        records=all_records,
        record_type=GenericModelRecord,
    )
