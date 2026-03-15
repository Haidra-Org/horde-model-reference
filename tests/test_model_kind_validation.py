from __future__ import annotations

import pytest

from horde_model_reference.model_kind_validation import FieldPolicy, KindPolicy, KindPolicyRegistry
from horde_model_reference.model_reference_records import ControlNetModelRecord, ImageGenerationModelRecord


def test_policy_registry_registers_once() -> None:
    """KindPolicyRegistry should allow registering a kind once, but not multiple times."""
    registry = KindPolicyRegistry()
    policy = KindPolicy(field_policies={"style": FieldPolicy(severity="warning")})

    registry.register("demo", policy)
    assert registry.get("demo") is policy

    with pytest.raises(ValueError):
        registry.register("demo", policy)


def test_image_generation_unknown_baseline_raises() -> None:
    """ImageGenerationModelRecord should reject unknown baselines."""
    with pytest.raises(ValueError):
        ImageGenerationModelRecord(name="test-model", baseline="unknown", nsfw=False)


def test_image_generation_unknown_style_raises() -> None:
    """ImageGenerationModelRecord should reject unknown styles."""
    with pytest.raises(ValueError):
        ImageGenerationModelRecord(
            name="test-model",
            baseline="stable_diffusion_1",
            nsfw=False,
            style="unknown-style",
        )


def test_controlnet_unknown_style_warns() -> None:
    """ControlNetModelRecord should allow unknown styles but emit a warning."""
    record = ControlNetModelRecord(name="test-model", controlnet_style="unknown-style")

    assert record.controlnet_style == "unknown-style"
