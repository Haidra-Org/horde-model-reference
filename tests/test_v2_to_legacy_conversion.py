"""Tests for v2 image-generation record to legacy dict conversion (validity + hash preservation)."""

from __future__ import annotations

from horde_model_reference.legacy.classes.legacy_converters import image_generation_record_to_legacy_dict
from horde_model_reference.legacy.classes.legacy_models import LegacyStableDiffusionRecord
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
)

_VAE_HASH = "a" * 64
_TE_HASH = "b" * 64
_EMBEDDED_VAE_HASH = "d" * 64
_SHA = "f" * 64


def test_split_file_record_converts_and_preserves_hashes() -> None:
    """A split-file record's per-file content hashes survive conversion and the result validates."""
    record = ImageGenerationModelRecord(
        name="Split Model",
        baseline="flux_1",
        nsfw=False,
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(file_name="unet.safetensors", file_url="https://x/unet", file_purpose="unet"),
                DownloadRecord(
                    file_name="ae.safetensors",
                    file_url="https://x/vae",
                    file_purpose="vae",
                    sha256sum=_SHA,
                    content_hash=_VAE_HASH,
                ),
                DownloadRecord(
                    file_name="te.safetensors",
                    file_url="https://x/te",
                    file_purpose="text_encoders",
                    sha256sum=_SHA,
                    content_hash=_TE_HASH,
                ),
            ],
        ),
    )

    legacy_dict = image_generation_record_to_legacy_dict(record)
    legacy = LegacyStableDiffusionRecord(**legacy_dict)

    by_type = {file.file_type: file for file in legacy.config.files}
    assert by_type["vae"].content_hash == _VAE_HASH
    assert by_type["text_encoders"].content_hash == _TE_HASH


def test_monolithic_record_preserves_embedded_hashes() -> None:
    """A monolithic record's embedded VAE hash survives conversion and the result validates."""
    record = ImageGenerationModelRecord(
        name="Monolithic SDXL",
        baseline="stable_diffusion_xl",
        nsfw=False,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name="model.safetensors", file_url="https://x/model", sha256sum=_SHA)],
            embedded_component_hashes={"vae": _EMBEDDED_VAE_HASH},
        ),
    )

    legacy_dict = image_generation_record_to_legacy_dict(record)
    legacy = LegacyStableDiffusionRecord(**legacy_dict)

    assert legacy.config.embedded_component_hashes == {"vae": _EMBEDDED_VAE_HASH}


def test_baseline_mapped_back_to_legacy_form() -> None:
    """A normalized baseline is mapped back to its legacy human string form."""
    record = ImageGenerationModelRecord(
        name="SD1 Model",
        baseline="stable_diffusion_1",
        nsfw=False,
        config=GenericModelRecordConfig(
            download=[DownloadRecord(file_name="model.safetensors", file_url="https://x/model", sha256sum=_SHA)],
        ),
    )
    legacy_dict = image_generation_record_to_legacy_dict(record)
    assert legacy_dict["baseline"] == "stable diffusion 1"
