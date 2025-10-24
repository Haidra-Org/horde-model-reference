"""Pydantic models encapsulating the legacy-to-new reference normalization rules."""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_serializer, model_validator

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.util import model_name_to_showcase_folder_name

_ALLOWED_CONFIG_FILENAMES = {"v2-inference-v.yaml", "v1-inference.yaml"}
_BASELINE_NORMALIZATION_MAP = {
    "stable diffusion 1": "stable_diffusion_1",
    "stable diffusion 2": "stable_diffusion_2_768",
    "stable diffusion 2 512": "stable_diffusion_2_512",
    "stable_diffusion_xl": "stable_diffusion_xl",
    "stable_cascade": "stable_cascade",
}


def _record_issue(info: ValidationInfo, message: str) -> None:
    """Append a validation issue message to the shared conversion context."""
    issues: list[str] | None = None
    if info.context is not None:
        issues = info.context.get("issues")
    if issues is None:
        return
    model_key = None
    if info.context is not None:
        model_key = info.context.get("model_key")
    if model_key:
        issues.append(f"{model_key} {message}")
    else:
        issues.append(message)


def _increment_host_counter(info: ValidationInfo, host: str) -> None:
    if info.context is None:
        return
    host_counter = info.context.get("host_counter")
    if isinstance(host_counter, dict):
        host_counter[host] = host_counter.get(host, 0) + 1


class LegacyConfigFile(BaseModel):
    """A single legacy config file entry."""

    model_config = ConfigDict(extra="allow")

    path: str
    md5sum: str | None = None
    sha256sum: str | None = None
    file_type: str | None = None

    @model_validator(mode="after")
    def _ensure_sha256sum_for_model_files(self, info: ValidationInfo) -> LegacyConfigFile:
        if not self.path:
            _record_issue(info, "has a config file with no path.")
            return self
        if ".yaml" in self.path or ".json" in self.path:
            return self
        if ".ckpt" not in self.path:
            _record_issue(info, "has a config file with an invalid path.")
        if self.sha256sum is None:
            _record_issue(info, "has a config file with no sha256sum.")
        return self


class LegacyConfigDownload(BaseModel):
    """A single legacy config download entry."""

    model_config = ConfigDict(extra="allow")

    file_name: str | None = None
    file_path: str | None = ""
    file_url: str | None = None

    @model_validator(mode="after")
    def _validate_download_fields(self, info: ValidationInfo) -> LegacyConfigDownload:
        if not self.file_name:
            _record_issue(info, "has a download with no file_name.")
        if not self.file_url:
            category = None
            if info.context is not None:
                category = info.context.get("category")
            if category != MODEL_REFERENCE_CATEGORY.clip:
                _record_issue(info, "has a download with no file_url.")
        return self


class LegacyConfig(BaseModel):
    """Typed representation of the legacy `config` payload."""

    model_config = ConfigDict(extra="ignore")

    files: list[LegacyConfigFile] = Field(default_factory=list)
    download: list[LegacyConfigDownload] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_config_dict(cls, value: object, info: ValidationInfo) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("config entries must be provided as a mapping")
        if len(value) > 2:
            _record_issue(info, "has more than 2 config entries.")
        coerced: dict[str, Any] = {}
        for key in ("files", "download"):
            entries = value.get(key) or []
            if not isinstance(entries, Iterable):
                raise TypeError(f"config[{key!s}] must be iterable")
            coerced[key] = list(entries)
        return coerced

    @model_validator(mode="after")
    def _normalize_entries(self, info: ValidationInfo) -> LegacyConfig:
        sha_lookup: dict[str, str] = {}

        for file_entry in self.files:
            if not file_entry.path:
                _record_issue(info, "has a config file with no path.")
                continue
            path_obj = Path(file_entry.path)
            if path_obj.suffix == ".yaml":
                if path_obj.name not in _ALLOWED_CONFIG_FILENAMES:
                    _record_issue(info, "has a non-standard config file.")
                continue
            if ".ckpt" not in file_entry.path:
                _record_issue(info, "has a config file with an invalid path.")
            if not file_entry.sha256sum:
                _record_issue(info, "has a config file with no sha256sum.")
            else:
                if len(file_entry.sha256sum) != 64:
                    _record_issue(info, "has a config file with an invalid sha256sum.")
                sha_lookup[file_entry.path] = file_entry.sha256sum

        for download_entry in self.download:
            if download_entry.file_path is None:
                _record_issue(info, "has a download with a file_path.")
                download_entry.file_path = ""
            elif download_entry.file_path != "":
                _record_issue(info, "has a download with a file_path.")
            if not download_entry.file_url:
                continue
            try:
                parsed_url = urllib.parse.urlparse(download_entry.file_url)
            except Exception:
                _record_issue(info, "has a download with an invalid file_url.")
                raise ValueError("invalid download url") from None
            _increment_host_counter(info, parsed_url.netloc)

        return self

    def is_empty(self) -> bool:
        """Return True when no download entries survive normalization.

        Note: files are always cleared during normalization (new format doesn't use them).
        """
        return len(self.download) == 0


class LegacyGenericRecord(BaseModel):
    """Base legacy record representation with shared validation rules."""

    model_config = ConfigDict(extra="allow")

    name: str
    type: str | None = None
    description: str | None = None
    version: str | None = None
    style: str | None = None
    nsfw: bool | None = None
    download_all: bool | None = None
    config: LegacyConfig = Field(default_factory=LegacyConfig)
    available: bool | None = None
    features_not_supported: list[str] | None = None

    @model_validator(mode="after")
    def _validate_common_rules(self, info: ValidationInfo) -> LegacyGenericRecord:
        model_key = None
        debug_mode = False
        if info.context is not None:
            model_key = info.context.get("model_key")
            debug_mode = bool(info.context.get("debug_mode"))
        if model_key and self.name != model_key:
            _record_issue(info, "name mismatch.")
        if self.available:
            _record_issue(info, "is flagged 'available'.")
        if self.download_all and debug_mode:
            _record_issue(info, "has download_all set.")
        if self.config.is_empty():
            _record_issue(info, "has no config.")
        if self.description is None:
            _record_issue(info, "has no description.")
        if self.style == "":
            _record_issue(info, "has no style.")
        return self


class LegacyStableDiffusionRecord(LegacyGenericRecord):
    """Stable Diffusion legacy record with category-specific validation."""

    type: Literal["ckpt"] = "ckpt"
    inpainting: bool = False
    baseline: str
    tags: list[str] | None = None
    showcases: list[str] | None = None
    min_bridge_version: int | None = None
    trigger: list[str] | None = None
    homepage: str | None = None
    size_on_disk_bytes: int | None = None
    optimization: str | None = None
    requirements: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None

    @model_validator(mode="after")
    def _validate_stable_diffusion_rules(self, info: ValidationInfo) -> LegacyStableDiffusionRecord:
        if self.type != "ckpt":
            _record_issue(info, "is not a ckpt!")
        if self.baseline in _BASELINE_NORMALIZATION_MAP:
            self.baseline = _BASELINE_NORMALIZATION_MAP[self.baseline]
        model_key = self.name
        existing_showcases: dict[str, list[str]] = {}
        if info.context is not None:
            model_key = info.context.get("model_key", self.name)
            existing_showcases = info.context.get("existing_showcase_files", {})
        expected_showcase_foldername = model_name_to_showcase_folder_name(model_key)
        on_disk_showcases = existing_showcases.get(expected_showcase_foldername)
        if self.showcases:
            if on_disk_showcases is None:
                _record_issue(
                    info,
                    f"is expected to have a showcase folder named {expected_showcase_foldername}",
                )
                # Don't raise, just record the issue and continue
            elif len(on_disk_showcases) == 0:
                _record_issue(info, "has no showcases defined on disk.")
            elif len(self.showcases) != len(on_disk_showcases):
                _record_issue(
                    info,
                    "has a mismatch between defined showcases and the files present on disk.",
                )
            if any("huggingface" in showcase for showcase in self.showcases):
                _record_issue(info, "has a huggingface showcase.")
        return self

    @model_serializer(when_used="always")
    def sort_model_fields(self) -> dict[str, Any]:
        """Sort fields for consistent JSON output."""
        field_order = [
            "name",
            "baseline",
            "optimization",
            "type",
            "inpainting",
            "description",
            "tags",
            "showcases",
            "version",
            "style",
            "trigger",
            "homepage",
            "nsfw",
            "download_all",
            "requirements",
            "config",
            "available",
            "features_not_supported",
            "size_on_disk_bytes",
        ]

        return_dict = {
            field: getattr(self, field)
            for field in field_order
            if hasattr(self, field) and getattr(self, field) is not None
        }

        # translate _BASELINE_NORMALIZATION_MAP keys back to their original values for output
        if "baseline" in return_dict:
            for original, normalized in _BASELINE_NORMALIZATION_MAP.items():
                if return_dict["baseline"] == normalized:
                    return_dict["baseline"] = original
                    break

        return return_dict


class LegacyTextGenerationRecord(LegacyGenericRecord):
    """Text generation legacy record with category-specific validation."""

    model_name: str | None = None
    baseline: str | None = None
    parameters: int | None = None
    display_name: str | None = None
    url: str | None = None
    tags: list[str] | None = None
    settings: dict[str, int | float | str | list[int] | list[float] | list[str] | bool] | None = None

    @model_validator(mode="after")
    def _validate_text_generation_rules(self, info: ValidationInfo) -> LegacyTextGenerationRecord:
        """Validate text generation specific rules."""
        if self.parameters is None:
            _record_issue(info, "has no parameters count.")
        return self


class LegacyBlipRecord(LegacyGenericRecord):
    """BLIP legacy record with category-specific normalization."""

    type: Literal["blip"] = "blip"


class LegacyClipRecord(LegacyGenericRecord):
    """CLIP legacy record with category-specific normalization."""

    type: Literal["clip", "coca"] = "clip"
    pretrained_name: str | None = None


class LegacyCodeformerRecord(LegacyGenericRecord):
    """Codeformers legacy record with category-specific normalization."""

    type: Literal["CodeFormers"] = "CodeFormers"


class LegacyEsrganRecord(LegacyGenericRecord):
    """ESRGAN legacy record with category-specific normalization."""

    type: Literal["realesrgan"] = "realesrgan"


class LegacyGfpganRecord(LegacyGenericRecord):
    """GFPGAN legacy record with category-specific normalization."""

    type: Literal["gfpgan"] = "gfpgan"


class LegacySafetyCheckerRecord(LegacyGenericRecord):
    """Safety Checker legacy record with category-specific normalization."""

    type: Literal["safety_checker"] = "safety_checker"


class LegacyMiscellaneousRecord(LegacyGenericRecord):
    """Miscellaneous legacy record with category-specific normalization."""

    type: Literal["layer_diffuse",]


class LegacyControlnetRecord(LegacyGenericRecord):
    """ControlNet legacy record with category-specific normalization."""

    type: Literal[
        "control_canny",
        "control_depth",
        "control_hed",
        "control_mlsd",
        "control_normal",
        "control_openpose",
        "control_fakescribbles",
        "control_scribble",
        "control_seg",
        "control_qr",
        "control_qr_xl",
    ]


LegacyRecordUnion = (
    LegacyStableDiffusionRecord
    | LegacyTextGenerationRecord
    | LegacyBlipRecord
    | LegacyClipRecord
    | LegacyCodeformerRecord
    | LegacyControlnetRecord
    | LegacyEsrganRecord
    | LegacyGfpganRecord
    | LegacyGenericRecord
    | LegacySafetyCheckerRecord
    | LegacyMiscellaneousRecord
)


def get_legacy_model_type(category: MODEL_REFERENCE_CATEGORY) -> type[LegacyRecordUnion]:
    """Get the appropriate Pydantic model class for a given category."""
    category_model_map: dict[MODEL_REFERENCE_CATEGORY, type[LegacyRecordUnion]] = {
        MODEL_REFERENCE_CATEGORY.image_generation: LegacyStableDiffusionRecord,
        MODEL_REFERENCE_CATEGORY.text_generation: LegacyTextGenerationRecord,
        MODEL_REFERENCE_CATEGORY.blip: LegacyBlipRecord,
        MODEL_REFERENCE_CATEGORY.clip: LegacyClipRecord,
        MODEL_REFERENCE_CATEGORY.codeformer: LegacyCodeformerRecord,
        MODEL_REFERENCE_CATEGORY.controlnet: LegacyControlnetRecord,
        MODEL_REFERENCE_CATEGORY.esrgan: LegacyEsrganRecord,
        MODEL_REFERENCE_CATEGORY.gfpgan: LegacyGfpganRecord,
        MODEL_REFERENCE_CATEGORY.safety_checker: LegacySafetyCheckerRecord,
        MODEL_REFERENCE_CATEGORY.miscellaneous: LegacyMiscellaneousRecord,
    }
    return category_model_map.get(category, LegacyGenericRecord)
