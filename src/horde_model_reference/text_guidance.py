"""Typed records for reusable text-model prompting guidance."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator
from strenum import StrEnum

TEXT_GUIDANCE_SCHEMA_VERSION = 1
"""Current serialized schema version for the text guidance catalog."""

_PROFILE_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_RAW_HTML_PATTERN = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")


class TextInteractionMode(StrEnum):
    """Durable ways in which a text model accepts prompts."""

    COMPLETION = "completion"
    INSTRUCTION = "instruction"
    CHAT = "chat"


class TextCapability(StrEnum):
    """Machine-consumable text capabilities tracked by the reference."""

    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"


class SupportStatus(StrEnum):
    """Conclusion for a durable model support claim."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class GuidanceProfileKind(StrEnum):
    """Kinds of reusable guidance profile."""

    PROMPT_CONTRACT = "prompt_contract"
    USAGE_RECIPE = "usage_recipe"


class TemplateSyntax(StrEnum):
    """Syntax labels for raw templates that the service stores but never executes."""

    JINJA2 = "jinja2"
    HANDLEBARS = "handlebars"
    PYTHON_FORMAT = "python_format"
    LITERAL = "literal"
    OTHER = "other"


class GuidanceSource(BaseModel):
    """Evidence or documentation source for a durable claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl
    title: str | None = None
    note: str | None = None


class SupportClaim(BaseModel):
    """Reviewed durable support conclusion with optional evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SupportStatus
    sources: list[GuidanceSource] = Field(default_factory=list)
    qualification: str | None = None


class TextContextWindow(BaseModel):
    """Publisher-advertised maximum context window for a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_tokens: int = Field(ge=1)
    sources: list[GuidanceSource] = Field(default_factory=list)
    qualification: str | None = None


class GuidanceRecordMetadata(BaseModel):
    """Lifecycle metadata for a published guidance record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int = Field(default=1, ge=1)
    created_at: int | None = None
    created_by: str | None = None
    updated_at: int | None = None
    updated_by: str | None = None


class GuidanceAudienceContent(BaseModel):
    """Structured prose shared by the web view and Markdown renderer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overview: str = ""
    use_cases: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    @field_validator("overview", "use_cases", "tips", "caveats")
    @classmethod
    def reject_raw_html(cls, value: str | list[str]) -> str | list[str]:
        """Reject raw HTML while retaining safe CommonMark source text."""
        values = [value] if isinstance(value, str) else value
        if any(_RAW_HTML_PATTERN.search(item) for item in values):
            raise ValueError("Guidance CommonMark must not contain raw HTML.")
        return value


class PromptMessage(BaseModel):
    """One structured message used by a guidance example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class PromptExample(BaseModel):
    """Portable input messages and their expected serialized prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    description: str | None = None
    messages: list[PromptMessage] = Field(min_length=1)
    rendered_prompt: str | None = None


class TemplateVariable(BaseModel):
    """Declared variable accepted by a raw prompting template."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    description: str
    required: bool = True


class RawPromptTemplate(BaseModel):
    """Display-only raw template tagged with its external syntax."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: str = Field(min_length=1)
    name: str
    syntax: TemplateSyntax
    syntax_name: str | None = None
    template: str
    variables: list[TemplateVariable] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_other_syntax_name(self) -> RawPromptTemplate:
        """Require a concrete label when the syntax is not in the known enum."""
        if self.syntax is TemplateSyntax.OTHER and not self.syntax_name:
            raise ValueError("syntax_name is required when template syntax is 'other'.")
        return self


class AIHordeRequestExample(BaseModel):
    """Validated example for the AI Horde text generation request surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    prompt: str
    parameters: dict[str, int | float | str | bool | list[int] | list[float] | list[str]] = Field(
        default_factory=dict,
    )
    notes: str | None = None


class TextUsageProfileBase(BaseModel):
    """Fields common to prompt contracts and usage recipes."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    kind: GuidanceProfileKind
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    summary: str
    user: GuidanceAudienceContent = Field(default_factory=GuidanceAudienceContent)
    developer: GuidanceAudienceContent = Field(default_factory=GuidanceAudienceContent)
    examples: list[PromptExample] = Field(default_factory=list)
    recommended_settings: dict[str, int | float | str | bool | list[int] | list[float] | list[str]] = Field(
        default_factory=dict,
    )
    ai_horde_examples: list[AIHordeRequestExample] = Field(default_factory=list)
    sources: list[GuidanceSource] = Field(default_factory=list)
    deprecated: bool = False
    metadata: GuidanceRecordMetadata = Field(default_factory=GuidanceRecordMetadata)

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        """Require stable URL-safe lowercase identifiers."""
        if not _PROFILE_ID_PATTERN.fullmatch(value):
            raise ValueError("profile_id must be a lowercase URL-safe identifier.")
        return value

    @field_validator("summary")
    @classmethod
    def reject_summary_html(cls, value: str) -> str:
        """Reject raw HTML in CommonMark summaries."""
        if _RAW_HTML_PATTERN.search(value):
            raise ValueError("Guidance CommonMark must not contain raw HTML.")
        return value


class TextPromptContract(TextUsageProfileBase):
    """Reusable base prompt serialization contract."""

    kind: Literal[GuidanceProfileKind.PROMPT_CONTRACT] = GuidanceProfileKind.PROMPT_CONTRACT
    interaction_modes: list[TextInteractionMode] = Field(min_length=1)
    accepted_roles: list[Literal["system", "user", "assistant", "tool"]] = Field(default_factory=list)
    role_markers: dict[str, str] = Field(default_factory=dict)
    stop_sequences: list[str] = Field(default_factory=list)
    templates: list[RawPromptTemplate] = Field(default_factory=list)


class TextUsageRecipe(TextUsageProfileBase):
    """Reusable scenario guidance that supplements a prompt contract."""

    kind: Literal[GuidanceProfileKind.USAGE_RECIPE] = GuidanceProfileKind.USAGE_RECIPE
    capability: TextCapability | None = None
    scenario: str | None = None


TextUsageProfile = Annotated[TextPromptContract | TextUsageRecipe, Field(discriminator="kind")]
"""Discriminated union of published guidance profile kinds."""


class TextGuidanceAssignment(BaseModel):
    """Explicit reusable-guidance assignment for one canonical model identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    primary_profile_id: str
    supplemental_profile_ids: list[str] = Field(default_factory=list)
    metadata: GuidanceRecordMetadata = Field(default_factory=GuidanceRecordMetadata)

    @model_validator(mode="after")
    def reject_duplicate_profiles(self) -> TextGuidanceAssignment:
        """Reject duplicate recipe identifiers and primary-as-supplemental assignment."""
        identifiers = [self.primary_profile_id, *self.supplemental_profile_ids]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("A guidance profile may be assigned only once to a model.")
        return self


class TextGuidanceStatus(StrEnum):
    """Consumer-facing guidance availability state for an exact model."""

    PUBLISHED = "published"
    LEGACY_LABEL = "legacy_label"
    UNDOCUMENTED = "undocumented"


class TextGuidanceSummary(BaseModel):
    """Compact guidance projection embedded in text-model reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: TextGuidanceStatus
    primary_profile_id: str | None = None
    supplemental_profile_ids: list[str] = Field(default_factory=list)


class TextGuidanceCatalogMetadata(BaseModel):
    """Revision metadata for the complete guidance catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = TEXT_GUIDANCE_SCHEMA_VERSION
    revision: int = Field(default=1, ge=1)
    updated_at: int | None = None


class TextGuidanceCatalog(BaseModel):
    """Atomic machine-readable catalog of profiles and explicit assignments."""

    model_config = ConfigDict(extra="forbid")

    metadata: TextGuidanceCatalogMetadata = Field(default_factory=TextGuidanceCatalogMetadata)
    profiles: dict[str, TextUsageProfile] = Field(default_factory=dict)
    assignments: dict[str, TextGuidanceAssignment] = Field(default_factory=dict)


class GuidanceProfileChange(BaseModel):
    """One profile operation and the value against which it was reviewed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["create", "update", "deprecate"]
    profile_id: str
    profile: TextUsageProfile | None = None
    expected_before: TextUsageProfile | None = None


class GuidanceAssignmentChange(BaseModel):
    """One explicit assignment replacement/removal with a reviewed prior value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    assignment: TextGuidanceAssignment | None = None
    expected_before: TextGuidanceAssignment | None = None


class TextGuidanceChangeSet(BaseModel):
    """Coherent guidance edits submitted as one pending-queue resource."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1, max_length=160)
    profile_changes: list[GuidanceProfileChange] = Field(default_factory=list)
    assignment_changes: list[GuidanceAssignmentChange] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_changes(self) -> TextGuidanceChangeSet:
        """Require at least one profile or assignment edit."""
        if not self.profile_changes and not self.assignment_changes:
            raise ValueError("A guidance change set must contain at least one change.")
        return self


class ResolvedTextGuidance(BaseModel):
    """Fully resolved guidance for one exact text model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    summary: TextGuidanceSummary
    primary_profile: TextPromptContract | None = None
    supplemental_profiles: list[TextUsageRecipe] = Field(default_factory=list)
    legacy_instruct_format: str | None = None
    catalog_metadata: TextGuidanceCatalogMetadata
