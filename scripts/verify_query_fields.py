"""Verify that query_fields.py FieldRef attributes match model_reference_records.py fields.

Run this script to detect drift between the Pydantic model fields and the
typed field namespaces in ``query_fields.py``. It prints any mismatches and
exits with a non-zero status if any are found.

Usage::

    python scripts/verify_query_fields.py
"""

from __future__ import annotations

import sys

from horde_model_reference.model_reference_records import (
    AudioGenerationModelRecord,
    BlipModelRecord,
    ClipModelRecord,
    CodeformerModelRecord,
    ControlNetModelRecord,
    EsrganModelRecord,
    GenericModelRecord,
    GfpganModelRecord,
    ImageGenerationModelRecord,
    MiscellaneousModelRecord,
    SafetyCheckerModelRecord,
    TextGenerationModelRecord,
    VideoGenerationModelRecord,
)
from horde_model_reference.query_fields import (
    AudioF,
    BlipF,
    ClipF,
    CodeformerF,
    ControlNetF,
    EsrganF,
    FieldRef,
    GenericF,
    GfpganF,
    ImageF,
    MiscellaneousF,
    SafetyCheckerF,
    TextF,
    VideoF,
)

_MAPPING: list[tuple[str, type[GenericModelRecord], type]] = [
    ("GenericF", GenericModelRecord, GenericF),
    ("ImageF", ImageGenerationModelRecord, ImageF),
    ("TextF", TextGenerationModelRecord, TextF),
    ("ControlNetF", ControlNetModelRecord, ControlNetF),
    ("ClipF", ClipModelRecord, ClipF),
    ("BlipF", BlipModelRecord, BlipF),
    ("CodeformerF", CodeformerModelRecord, CodeformerF),
    ("EsrganF", EsrganModelRecord, EsrganF),
    ("GfpganF", GfpganModelRecord, GfpganF),
    ("SafetyCheckerF", SafetyCheckerModelRecord, SafetyCheckerF),
    ("VideoF", VideoGenerationModelRecord, VideoF),
    ("AudioF", AudioGenerationModelRecord, AudioF),
    ("MiscellaneousF", MiscellaneousModelRecord, MiscellaneousF),
]


def _get_field_refs(cls: type) -> set[str]:
    """Return the set of FieldRef attribute names on *cls* (including inherited)."""
    refs: set[str] = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        val = getattr(cls, name, None)
        if isinstance(val, FieldRef):
            refs.add(name)
    return refs


def main() -> int:
    """Compare FieldRef attributes against Pydantic model_fields and report drift."""
    errors: list[str] = []

    for label, record_cls, field_cls in _MAPPING:
        model_fields = set(record_cls.model_fields.keys())
        field_refs = _get_field_refs(field_cls)

        missing = model_fields - field_refs
        extra = field_refs - model_fields

        if missing:
            errors.append(f"{label}: missing FieldRef for model fields: {sorted(missing)}")
        if extra:
            errors.append(f"{label}: extra FieldRef not in model fields: {sorted(extra)}")

    if errors:
        print("query_fields.py drift detected:\n")
        for err in errors:
            print(f"  - {err}")
        print(f"\n{len(errors)} issue(s) found.")
        return 1

    print(f"All {len(_MAPPING)} field namespaces match their record types.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
