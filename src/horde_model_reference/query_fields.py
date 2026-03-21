"""Typed field references for the query builder DSL.

Provides per-category field namespaces (``ImageF``, ``TextF``, etc.) that
give IDE autocomplete, static type checking, and a composable predicate
language for ``ModelQuery.where()`` and ``ModelQuery.order_by()``.

Usage::

    from horde_model_reference import ImageF, false
    from horde_model_reference.query import build_image_query

    results = (
        build_image_query(records)
        .where(ImageF.nsfw == false, ImageF.size_on_disk_bytes > 1_000_000_000)
        .order_by(ImageF.size_on_disk_bytes.asc())
        .to_list()
    )
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class Predicate:
    """A composable predicate for use with ``ModelQuery.where()`` and ``filter()``."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[Any], bool]) -> None:  # noqa: D107
        self._fn = fn

    def __call__(self, record: object) -> bool:
        """Evaluate the predicate against *record*."""
        return self._fn(record)

    def __and__(self, other: Predicate) -> Predicate:
        left, right = self._fn, other._fn
        return Predicate(lambda r: left(r) and right(r))

    def __or__(self, other: Predicate) -> Predicate:
        left, right = self._fn, other._fn
        return Predicate(lambda r: left(r) or right(r))

    def __invert__(self) -> Predicate:
        fn = self._fn
        return Predicate(lambda r: not fn(r))

    def __repr__(self) -> str:
        return f"Predicate({self._fn!r})"


class OrderSpec:
    """Specifies a field and sort direction for ``ModelQuery.order_by()``."""

    __slots__ = ("descending", "field")

    def __init__(self, field: str, *, descending: bool = False) -> None:  # noqa: D107
        self.field = field
        self.descending = descending

    def __repr__(self) -> str:
        direction = "DESC" if self.descending else "ASC"
        return f"OrderSpec({self.field!r}, {direction})"


class FieldRef:
    """Reference to a model record field that supports comparison operators.

    Comparison operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``)
    return ``Predicate`` objects that can be passed to ``ModelQuery.where()``
    or combined with ``&``, ``|``, ``~`` for complex expressions.
    """

    __slots__ = ("_field_name",)

    def __init__(self, field_name: str) -> None:  # noqa: D107
        self._field_name = field_name

    @property
    def field_name(self) -> str:
        """The underlying field name string."""
        return self._field_name

    def __eq__(self, other: object) -> Predicate:  # type: ignore[override]
        field = self._field_name
        if isinstance(other, FieldRef):
            other_field = other._field_name
            return Predicate(lambda r: getattr(r, field, None) == getattr(r, other_field, None))
        return Predicate(lambda r: getattr(r, field, None) == other)

    def __ne__(self, other: object) -> Predicate:  # type: ignore[override]
        field = self._field_name
        if isinstance(other, FieldRef):
            other_field = other._field_name
            return Predicate(lambda r: getattr(r, field, None) != getattr(r, other_field, None))
        return Predicate(lambda r: getattr(r, field, None) != other)

    def __lt__(self, other: object) -> Predicate:
        field = self._field_name
        return Predicate(lambda r: (v := getattr(r, field, None)) is not None and v < other)

    def __le__(self, other: object) -> Predicate:
        field = self._field_name
        return Predicate(lambda r: (v := getattr(r, field, None)) is not None and v <= other)

    def __gt__(self, other: object) -> Predicate:
        field = self._field_name
        return Predicate(lambda r: (v := getattr(r, field, None)) is not None and v > other)

    def __ge__(self, other: object) -> Predicate:
        field = self._field_name
        return Predicate(lambda r: (v := getattr(r, field, None)) is not None and v >= other)

    def contains(self, item: object) -> Predicate:
        """Check whether the field value (an iterable) contains *item*."""
        field = self._field_name
        return Predicate(lambda r: item in (getattr(r, field, None) or []))

    def is_in(self, choices: Iterable[object]) -> Predicate:
        """Check whether the field value is a member of *choices*."""
        choice_set = set(choices)
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) in choice_set)

    def is_none(self) -> Predicate:
        """Check whether the field value is ``None``."""
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) is None)

    def is_not_none(self) -> Predicate:
        """Check whether the field value is not ``None``."""
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) is not None)

    def asc(self) -> OrderSpec:
        """Return an ascending ``OrderSpec`` for this field."""
        return OrderSpec(self._field_name, descending=False)

    def desc(self) -> OrderSpec:
        """Return a descending ``OrderSpec`` for this field."""
        return OrderSpec(self._field_name, descending=True)

    def __hash__(self) -> int:
        return hash(self._field_name)

    def __repr__(self) -> str:
        return f"FieldRef({self._field_name!r})"



true: bool = True
"""Boolean sentinel for use in ``FieldRef`` comparisons: ``ImageF.nsfw == true``."""

false: bool = False
"""Boolean sentinel for use in ``FieldRef`` comparisons: ``ImageF.nsfw == false``."""


# ---------------------------------------------------------------------------
# Per-category field namespaces
# ---------------------------------------------------------------------------


class GenericFields:
    """Field references for ``GenericModelRecord``."""

    record_type: FieldRef = FieldRef("record_type")
    name: FieldRef = FieldRef("name")
    description: FieldRef = FieldRef("description")
    version: FieldRef = FieldRef("version")
    finetune_series: FieldRef = FieldRef("finetune_series")
    metadata: FieldRef = FieldRef("metadata")
    config: FieldRef = FieldRef("config")
    model_classification: FieldRef = FieldRef("model_classification")


class ImageFields(GenericFields):
    """Field references for ``ImageGenerationModelRecord``."""

    inpainting: FieldRef = FieldRef("inpainting")
    baseline: FieldRef = FieldRef("baseline")
    optimization: FieldRef = FieldRef("optimization")
    tags: FieldRef = FieldRef("tags")
    showcases: FieldRef = FieldRef("showcases")
    min_bridge_version: FieldRef = FieldRef("min_bridge_version")
    trigger: FieldRef = FieldRef("trigger")
    homepage: FieldRef = FieldRef("homepage")
    nsfw: FieldRef = FieldRef("nsfw")
    style: FieldRef = FieldRef("style")
    requirements: FieldRef = FieldRef("requirements")
    size_on_disk_bytes: FieldRef = FieldRef("size_on_disk_bytes")


class TextFields(GenericFields):
    """Field references for ``TextGenerationModelRecord``."""

    baseline: FieldRef = FieldRef("baseline")
    parameters_count: FieldRef = FieldRef("parameters_count")
    nsfw: FieldRef = FieldRef("nsfw")
    style: FieldRef = FieldRef("style")
    display_name: FieldRef = FieldRef("display_name")
    url: FieldRef = FieldRef("url")
    tags: FieldRef = FieldRef("tags")
    instruct_format: FieldRef = FieldRef("instruct_format")
    settings: FieldRef = FieldRef("settings")
    text_model_group: FieldRef = FieldRef("text_model_group")


class ControlNetFields(GenericFields):
    """Field references for ``ControlNetModelRecord``."""

    controlnet_style: FieldRef = FieldRef("controlnet_style")


class ClipFields(GenericFields):
    """Field references for ``ClipModelRecord``."""

    pretrained_name: FieldRef = FieldRef("pretrained_name")


class BlipFields(GenericFields):
    """Field references for ``BlipModelRecord``."""


class CodeformerFields(GenericFields):
    """Field references for ``CodeformerModelRecord``."""


class EsrganFields(GenericFields):
    """Field references for ``EsrganModelRecord``."""


class GfpganFields(GenericFields):
    """Field references for ``GfpganModelRecord``."""


class SafetyCheckerFields(GenericFields):
    """Field references for ``SafetyCheckerModelRecord``."""


class VideoFields(GenericFields):
    """Field references for ``VideoGenerationModelRecord``."""

    baseline: FieldRef = FieldRef("baseline")
    nsfw: FieldRef = FieldRef("nsfw")
    tags: FieldRef = FieldRef("tags")


class AudioFields(GenericFields):
    """Field references for ``AudioGenerationModelRecord``."""

    baseline: FieldRef = FieldRef("baseline")
    nsfw: FieldRef = FieldRef("nsfw")
    tags: FieldRef = FieldRef("tags")


class MiscellaneousFields(GenericFields):
    """Field references for ``MiscellaneousModelRecord``."""
