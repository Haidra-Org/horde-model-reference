"""Test canonical model-name aliases and the guarded key migration utility."""

import pytest

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_aliases import model_name_candidates, resolve_model_alias
from scripts.rename_model_record import rename_model_record_key


def test_controlnet_qr_alias_resolves_to_the_canonical_key() -> None:
    """Verify deprecated direct lookups resolve without duplicating category keys."""
    category = MODEL_REFERENCE_CATEGORY.controlnet

    assert resolve_model_alias(category, "control_qr_sdxl") == "control_qr_xl"
    assert resolve_model_alias(category, "control_qr_xl") == "control_qr_xl"
    assert model_name_candidates(category, "control_qr_sdxl") == ("control_qr_xl", "control_qr_sdxl")
    assert model_name_candidates(category, "control_qr_xl") == ("control_qr_xl", "control_qr_sdxl")


def test_key_rename_moves_only_the_canonical_mapping_key() -> None:
    """Verify the guarded migration preserves the already-canonical record body."""
    source = {
        "control_qr_sdxl": {"name": "control_qr_xl", "controlnet_style": "control_qr_xl"},
        "unrelated": {"name": "unrelated"},
    }

    renamed = rename_model_record_key(source, old_name="control_qr_sdxl", new_name="control_qr_xl")

    assert "control_qr_sdxl" not in renamed
    assert renamed["control_qr_xl"] == source["control_qr_sdxl"]
    assert renamed["unrelated"] == source["unrelated"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({}, "Source model key does not exist"),
        (
            {"control_qr_sdxl": {"name": "control_qr_xl"}, "control_qr_xl": {"name": "control_qr_xl"}},
            "Destination model key already exists",
        ),
        ({"control_qr_sdxl": {"name": "wrong"}}, "Source record name must match"),
    ],
)
def test_key_rename_refuses_unsafe_input(source: dict[str, object], message: str) -> None:
    """Verify the migration fails closed before changing ambiguous source data."""
    with pytest.raises(ValueError, match=message):
        rename_model_record_key(source, old_name="control_qr_sdxl", new_name="control_qr_xl")
