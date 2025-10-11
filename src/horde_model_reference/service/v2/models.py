"""Request and response models for the v2 API."""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from horde_model_reference.model_reference_records import (
    AudioGenerationModelRecord,
    CLIPModelRecord,
    ControlNetModelRecord,
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
    VideoGenerationModelRecord,
)

ModelRecordUnion = Annotated[
    ImageGenerationModelRecord
    | TextGenerationModelRecord
    | CLIPModelRecord
    | ControlNetModelRecord
    | VideoGenerationModelRecord
    | AudioGenerationModelRecord
    | GenericModelRecord,
    Field(
        description="A model record conforming to one of the category-specific schemas",
    ),
]
"""Union of all possible model record types for OpenAPI documentation."""


class ErrorDetail(BaseModel):
    """Detail about a specific error."""

    loc: list[str | int] | None = None
    """Location of the error (for validation errors)."""
    msg: str
    """Error message."""
    type: str | None = None
    """Error type."""


class ErrorResponse(BaseModel):
    """Standardized error response."""

    detail: str | list[ErrorDetail]
    """Error details - either a string message or list of validation errors."""
