"""Request and response models for the v2 API."""

from typing import Annotated

from pydantic import BaseModel, Field

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
    LoraModelRecord,
    MiscellaneousModelRecord,
    SafetyCheckerModelRecord,
    TextGenerationModelRecord,
    TextualInversionModelRecord,
    VideoGenerationModelRecord,
)


class UserRolesResponse(BaseModel):
    """Response model for the user roles endpoint."""

    user_id: str
    """The unique Horde user ID (e.g., '6572')."""

    username: str
    """The full Horde username including discriminator (e.g., 'Tazlin#6572')."""

    roles: list[str]
    """List of roles assigned to the user (e.g., ['approver', 'requestor'])."""

    is_approver: bool
    """Whether the user has approver privileges for the pending queue."""

    is_requestor: bool
    """Whether the user has requestor privileges for the pending queue."""


ModelRecordUnion = Annotated[
    ImageGenerationModelRecord
    | TextGenerationModelRecord
    | ControlNetModelRecord
    | BlipModelRecord
    | ClipModelRecord
    | CodeformerModelRecord
    | EsrganModelRecord
    | GfpganModelRecord
    | SafetyCheckerModelRecord
    | VideoGenerationModelRecord
    | AudioGenerationModelRecord
    | MiscellaneousModelRecord
    | LoraModelRecord
    | TextualInversionModelRecord
    | GenericModelRecord,
    Field(
        description="A model record conforming to one of the category-specific schemas",
    ),
]
"""Union of all possible model record types for OpenAPI documentation."""

ModelRecordUnionType = (
    ImageGenerationModelRecord
    | TextGenerationModelRecord
    | ControlNetModelRecord
    | BlipModelRecord
    | ClipModelRecord
    | CodeformerModelRecord
    | EsrganModelRecord
    | GfpganModelRecord
    | SafetyCheckerModelRecord
    | VideoGenerationModelRecord
    | AudioGenerationModelRecord
    | MiscellaneousModelRecord
    | LoraModelRecord
    | TextualInversionModelRecord
    | GenericModelRecord
)
"""Union of all possible model record types for type hints."""
