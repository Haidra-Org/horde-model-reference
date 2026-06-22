"""Canonical on-disk presence contract for a multi-file (Z-Image-Turbo-shaped) model.

Z-Image-Turbo sits behind a user report: a worker surface said the model "will download" while the
weights were on disk and the download subsystem reported them present. These tests pin the *canonical*
presence semantics that every consumer should delegate to: existence-only (no checksum/sidecar gate),
sibling-folder component routing, and multi-root search. The fixture mirrors the real record shape: a
unet in ``compvis``, a VAE in ``../vae`` and a text encoder in ``../text_encoders``.
"""

from __future__ import annotations

from pathlib import Path

from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)
from horde_model_reference.on_disk_layout import file_paths_for, is_present

# The real Z-Image-Turbo file layout (file_name -> file_purpose), as produced by the legacy converter
# (file_purpose is taken from the legacy ``files[].file_type``).
_ZIMAGE_FILES: tuple[tuple[str, str], ...] = (
    ("z_image_turbo_bf16.safetensors", "unet"),
    ("ae.safetensors", "vae"),
    ("qwen_3_4b.safetensors", "text_encoders"),
)


def _zimage_record() -> GenericModelRecord:
    return GenericModelRecord(
        name="Z-Image-Turbo",
        record_type=MODEL_REFERENCE_CATEGORY.image_generation,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
        config=GenericModelRecordConfig(
            download=[
                DownloadRecord(
                    file_name=file_name,
                    file_url=f"https://example.com/{file_name}",
                    file_purpose=file_purpose,
                )
                for file_name, file_purpose in _ZIMAGE_FILES
            ],
        ),
    )


def _place_all_files(root: Path, *, with_sidecars: bool) -> None:
    """Lay the three Z-Image files in their canonical folders (compvis / vae / text_encoders)."""
    targets = {
        "z_image_turbo_bf16.safetensors": root / "compvis",
        "ae.safetensors": root / "vae",
        "qwen_3_4b.safetensors": root / "text_encoders",
    }
    for file_name, folder in targets.items():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / file_name).write_bytes(b"x")
        if with_sidecars:
            (folder / f"{Path(file_name).stem}.sha256").write_text("deadbeef *" + file_name)


def test_present_when_all_components_on_disk_without_sidecars(tmp_path: Path) -> None:
    """The canonical check is existence-only: a bare weight (no .sha256 sidecar) still counts present.

    This is the contract that distinguishes presence from integrity. hordelib's manager additionally
    requires a sidecar; the centralization makes presence defer to this existence-only answer.
    """
    _place_all_files(tmp_path, with_sidecars=False)
    assert is_present(_zimage_record(), tmp_path) is True


def test_absent_when_a_sibling_component_is_missing(tmp_path: Path) -> None:
    """Only the unet on disk (VAE/text-encoder folders empty) is NOT present: every file must exist."""
    (tmp_path / "compvis").mkdir(parents=True)
    (tmp_path / "compvis" / "z_image_turbo_bf16.safetensors").write_bytes(b"x")
    assert is_present(_zimage_record(), tmp_path) is False


def test_components_route_to_sibling_folders(tmp_path: Path) -> None:
    """file_paths_for routes the VAE to ../vae and the text encoder to ../text_encoders."""
    paths = file_paths_for(_zimage_record(), tmp_path)
    resolved = {p.name: p.parent.name for p in paths}
    assert resolved["z_image_turbo_bf16.safetensors"] == "compvis"
    assert resolved["ae.safetensors"] == "vae"
    assert resolved["qwen_3_4b.safetensors"] == "text_encoders"


def test_present_when_components_spread_across_extra_root(tmp_path: Path) -> None:
    """Files split across a second disk count as present via extra_roots (the multi-root contract)."""
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    # Unet on the primary disk; VAE + text encoder live only on the extra disk.
    (primary / "compvis").mkdir(parents=True)
    (primary / "compvis" / "z_image_turbo_bf16.safetensors").write_bytes(b"x")
    (extra / "vae").mkdir(parents=True)
    (extra / "vae" / "ae.safetensors").write_bytes(b"x")
    (extra / "text_encoders").mkdir(parents=True)
    (extra / "text_encoders" / "qwen_3_4b.safetensors").write_bytes(b"x")

    assert is_present(_zimage_record(), primary) is False
    assert is_present(_zimage_record(), primary, extra_roots=[extra]) is True
