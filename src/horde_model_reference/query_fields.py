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

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------


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

    # -- Comparison operators → Predicate --

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

    # -- Collection operators → Predicate --

    def contains(self, item: object) -> Predicate:
        """Check whether the field value (an iterable) contains *item*."""
        field = self._field_name
        return Predicate(lambda r: item in (getattr(r, field, None) or []))

    def is_in(self, choices: Iterable[object]) -> Predicate:
        """Check whether the field value is a member of *choices*."""
        choice_set = set(choices)
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) in choice_set)

    # -- Nullability operators → Predicate --

    def is_none(self) -> Predicate:
        """Check whether the field value is ``None``."""
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) is None)

    def is_not_none(self) -> Predicate:
        """Check whether the field value is not ``None``."""
        field = self._field_name
        return Predicate(lambda r: getattr(r, field, None) is not None)

    # -- Ordering → OrderSpec --

    def asc(self) -> OrderSpec:
        """Return an ascending ``OrderSpec`` for this field."""
        return OrderSpec(self._field_name, descending=False)

    def desc(self) -> OrderSpec:
        """Return a descending ``OrderSpec`` for this field."""
        return OrderSpec(self._field_name, descending=True)

    # -- Identity --

    def __hash__(self) -> int:
        return hash(self._field_name)

    def __repr__(self) -> str:
        return f"FieldRef({self._field_name!r})"


# ---------------------------------------------------------------------------
# Boolean sentinels (avoid linter warnings when comparing FieldRef to bool)
# ---------------------------------------------------------------------------

true: bool = True
"""Boolean sentinel for use in ``FieldRef`` comparisons: ``ImageF.nsfw == true``."""

false: bool = False
"""Boolean sentinel for use in ``FieldRef`` comparisons: ``ImageF.nsfw == false``."""


# ---------------------------------------------------------------------------
# Per-category field namespaces
# ---------------------------------------------------------------------------


class GenericF:
    """Field references for ``GenericModelRecord``."""

    record_type = FieldRef("record_type")
    name = FieldRef("name")
    description = FieldRef("description")
    version = FieldRef("version")
    finetune_series = FieldRef("finetune_series")
    metadata = FieldRef("metadata")
    config = FieldRef("config")
    model_classification = FieldRef("model_classification")


class ImageF(GenericF):
    """Field references for ``ImageGenerationModelRecord``."""

    inpainting = FieldRef("inpainting")
    baseline = FieldRef("baseline")
    optimization = FieldRef("optimization")
    tags = FieldRef("tags")
    showcases = FieldRef("showcases")
    min_bridge_version = FieldRef("min_bridge_version")
    trigger = FieldRef("trigger")
    homepage = FieldRef("homepage")
    nsfw = FieldRef("nsfw")
    style = FieldRef("style")
    requirements = FieldRef("requirements")
    size_on_disk_bytes = FieldRef("size_on_disk_bytes")


class TextF(GenericF):
    """Field references for ``TextGenerationModelRecord``."""

    baseline = FieldRef("baseline")
    parameters_count = FieldRef("parameters_count")
    nsfw = FieldRef("nsfw")
    style = FieldRef("style")
    display_name = FieldRef("display_name")
    url = FieldRef("url")
    tags = FieldRef("tags")
    instruct_format = FieldRef("instruct_format")
    settings = FieldRef("settings")
    text_model_group = FieldRef("text_model_group")


class ControlNetF(GenericF):
    """Field references for ``ControlNetModelRecord``."""

    controlnet_style = FieldRef("controlnet_style")


class ClipF(GenericF):
    """Field references for ``ClipModelRecord``."""

    pretrained_name = FieldRef("pretrained_name")


class BlipF(GenericF):
    """Field references for ``BlipModelRecord``."""


class CodeformerF(GenericF):
    """Field references for ``CodeformerModelRecord``."""


class EsrganF(GenericF):
    """Field references for ``EsrganModelRecord``."""


class GfpganF(GenericF):
    """Field references for ``GfpganModelRecord``."""


class SafetyCheckerF(GenericF):
    """Field references for ``SafetyCheckerModelRecord``."""


class VideoF(GenericF):
    """Field references for ``VideoGenerationModelRecord``."""

    baseline = FieldRef("baseline")
    nsfw = FieldRef("nsfw")
    tags = FieldRef("tags")


class AudioF(GenericF):
    """Field references for ``AudioGenerationModelRecord``."""

    baseline = FieldRef("baseline")
    nsfw = FieldRef("nsfw")
    tags = FieldRef("tags")


class MiscellaneousF(GenericF):
    """Field references for ``MiscellaneousModelRecord``."""
