"""Request and response models for the v2 API."""

from typing import Annotated

from pydantic import Field

from horde_model_reference.model_reference_records import (
    ControlNetModelRecord,
    GenericModelRecord,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)

ModelRecordUnion = Annotated[
    ImageGenerationModelRecord | TextGenerationModelRecord | ControlNetModelRecord | GenericModelRecord,
    Field(
        description="A model record conforming to one of the category-specific schemas",
    ),
]
"""Union of all possible model record types for OpenAPI documentation."""

ModelRecordUnionType = (
    ImageGenerationModelRecord | TextGenerationModelRecord | ControlNetModelRecord | GenericModelRecord
)
"""Union of all possible model record types for type hints."""
