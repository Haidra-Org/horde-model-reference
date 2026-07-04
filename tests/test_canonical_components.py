"""Tests for canonical component derivation (frequency threshold, curated override, sources, lookup)."""

from __future__ import annotations

from horde_model_reference import ModelClassification
from horde_model_reference.canonical_components import derive_canonical_registry
from horde_model_reference.component_hash import ComponentKind
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)

_VAE_A = "a" * 64
_VAE_B = "b" * 64
_TE = "c" * 64


def _vae_download(content_hash: str) -> DownloadRecord:
    return DownloadRecord(
        file_name="model.vae.safetensors",
        file_url="https://example.invalid/vae",
        file_purpose="vae",
        content_hash=content_hash,
    )


def _record(
    downloads: list[DownloadRecord] | None = None,
    embedded: dict[str, str] | None = None,
) -> GenericModelRecord:
    return GenericModelRecord(
        record_type="image_generation",
        name="unused",
        model_classification=ModelClassification(domain="image", purpose="generation"),
        config=GenericModelRecordConfig(download=downloads or [], embedded_component_hashes=embedded),
    )


def test_shared_component_is_promoted() -> None:
    """A component two models carry is promoted, with both models as sources."""
    records = {"m1": _record([_vae_download(_VAE_A)]), "m2": _record([_vae_download(_VAE_A)])}
    registry = derive_canonical_registry(records)
    component = registry.by_hash(_VAE_A)
    assert component is not None
    assert component.kind is ComponentKind.VAE
    assert component.shared_by_model_count == 2
    assert {source.model_name for source in component.sources} == {"m1", "m2"}


def test_single_model_component_not_promoted() -> None:
    """A component only one model carries stays out of the set at the default threshold."""
    registry = derive_canonical_registry({"m1": _record([_vae_download(_VAE_A)])})
    assert registry.by_hash(_VAE_A) is None


def test_allowlist_forces_single_model_component() -> None:
    """The allowlist promotes a component regardless of how few models carry it."""
    registry = derive_canonical_registry({"m1": _record([_vae_download(_VAE_A)])}, allow={_VAE_A})
    assert registry.by_hash(_VAE_A) is not None


def test_denylist_excludes_frequent_component() -> None:
    """The denylist excludes a component even when it clears the frequency threshold."""
    records = {"m1": _record([_vae_download(_VAE_B)]), "m2": _record([_vae_download(_VAE_B)])}
    registry = derive_canonical_registry(records, deny={_VAE_B})
    assert registry.by_hash(_VAE_B) is None


def test_embedded_and_splitfile_sources_combine() -> None:
    """The same hash embedded in one model and standalone in another counts as two models."""
    records = {
        "embedded_model": _record(embedded={"vae": _VAE_A}),
        "splitfile_model": _record([_vae_download(_VAE_A)]),
    }
    component = derive_canonical_registry(records).by_hash(_VAE_A)
    assert component is not None
    assert component.shared_by_model_count == 2
    by_model = {source.model_name: source for source in component.sources}
    assert by_model["embedded_model"].embedded is True
    assert by_model["splitfile_model"].embedded is False
    assert by_model["splitfile_model"].file_purpose == "vae"


def test_for_kind_partitions_by_component() -> None:
    """for_kind returns only components of the requested kind."""
    records = {
        "m1": _record([_vae_download(_VAE_A)]),
        "m2": _record([_vae_download(_VAE_A)]),
        "m3": _record(embedded={"text_encoders": _TE}),
        "m4": _record(embedded={"text_encoders": _TE}),
    }
    registry = derive_canonical_registry(records)
    assert [component.content_hash for component in registry.for_kind(ComponentKind.VAE)] == [_VAE_A]
    assert [component.content_hash for component in registry.for_kind(ComponentKind.TEXT_ENCODERS)] == [_TE]
