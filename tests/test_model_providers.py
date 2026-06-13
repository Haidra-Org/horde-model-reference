"""Tests for the third-party model provider system.

Covers provider registration/unregistration, source-aware reads and queries
(``"horde"`` / ``"any"`` / specific source ids), canonical-wins collision handling
plus duplicate detection, custom record-type subclassing, async parity, and
error isolation when a provider raises.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from horde_model_reference.meta_consts import (
    KNOWN_IMAGE_GENERATION_BASELINE,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_manager import ModelReferenceManager, PrefetchStrategy
from horde_model_reference.model_reference_records import (
    MODEL_RECORD_TYPE_LOOKUP,
    GenericModelRecord,
    ImageGenerationModelRecord,
    LoraModelRecord,
    register_record_type,
)
from horde_model_reference.providers import (
    ANY_SOURCE,
    HORDE_SOURCE_ID,
    ModelProvider,
    ModelProviderRegistry,
    StaticModelProvider,
)
from horde_model_reference.source_consts import normalize_source_selector

CanonicalView = dict[MODEL_REFERENCE_CATEGORY, dict[str, GenericModelRecord] | None]


def _make_image_model(name: str, *, nsfw: bool = False) -> ImageGenerationModelRecord:
    """Create a minimal image-generation record for tests."""
    return ImageGenerationModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
        nsfw=nsfw,
    )


def _make_lora_model(name: str, *, nsfw: bool = False) -> LoraModelRecord:
    """Create a minimal Lora model record for tests."""
    return LoraModelRecord(
        name=name,
        baseline=KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
        nsfw=nsfw,
        trigger=["LoraSDXLTrigger"],
    )


class StubProvider(ModelProvider):
    """A minimal in-memory :class:`ModelProvider` for tests."""

    def __init__(
        self,
        source_id: str,
        records_by_category: dict[MODEL_REFERENCE_CATEGORY | str, dict[str, GenericModelRecord]],
        *,
        raise_on_fetch: bool = False,
    ) -> None:
        """Store the canned data this provider should serve."""
        self._source_id = source_id
        self._records_by_category = records_by_category
        self._raise_on_fetch = raise_on_fetch
        self.fetch_calls = 0

    @property
    def source_id(self) -> str:
        """Return the configured source id."""
        return self._source_id

    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        """Return the categories this provider can serve."""
        return set(self._records_by_category)

    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        """Return the canned records for *category*, or raise/None per configuration."""
        self.fetch_calls += 1
        if self._raise_on_fetch:
            raise RuntimeError("provider boom")
        return self._records_by_category.get(category)


@pytest.fixture
def provider_env(
    restore_manager_singleton: None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModelReferenceManager, CanonicalView]:
    """Return a manager whose canonical reads are served from an in-memory view.

    The returned ``CanonicalView`` dict can be mutated by tests to control what the
    canonical ``"horde"`` source returns, fully isolating tests from the network.
    """
    manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
    canonical: CanonicalView = {}

    def _fake_all(overwrite_existing: bool = False, *, safe_mode: bool = False) -> CanonicalView:
        return dict(canonical)

    async def _fake_all_async(
        overwrite_existing: bool = False,
        *,
        safe_mode: bool = False,
        httpx_client: object | None = None,
    ) -> CanonicalView:
        return dict(canonical)

    monkeypatch.setattr(manager, "get_all_model_references_or_none", _fake_all)
    monkeypatch.setattr(manager, "get_all_model_references_or_none_async", _fake_all_async)
    return manager, canonical


def test_normalize_source_selector_variants() -> None:
    """A bare string, empty input, and sequences normalize as documented."""
    assert normalize_source_selector("civitai") == ["civitai"]
    assert normalize_source_selector([]) == [HORDE_SOURCE_ID]
    assert normalize_source_selector(["a", "a", "b"]) == ["a", "b"]
    assert normalize_source_selector(ANY_SOURCE) == [ANY_SOURCE]


def test_registry_register_and_lookup() -> None:
    """Registration is order-preserving and supports get/has/unregister."""
    registry = ModelProviderRegistry()
    provider_a = StubProvider("alpha", {})
    provider_b = StubProvider("beta", {})

    registry.register(provider_a)
    registry.register(provider_b)

    assert registry.source_ids() == ["alpha", "beta"]
    assert registry.has("alpha")
    assert registry.get("beta") is provider_b
    assert registry.unregister("alpha") is True
    assert registry.unregister("alpha") is False
    assert registry.source_ids() == ["beta"]


def test_registry_rejects_duplicate_without_replace() -> None:
    """Re-registering a source id raises unless ``replace=True``."""
    registry = ModelProviderRegistry()
    registry.register(StubProvider("dup", {}))

    with pytest.raises(ValueError, match="dup"):
        registry.register(StubProvider("dup", {}))

    replacement = StubProvider("dup", {})
    registry.register(replacement, replace=True)
    assert registry.get("dup") is replacement


@pytest.mark.parametrize("reserved", [HORDE_SOURCE_ID, ANY_SOURCE])
def test_registry_rejects_reserved_source_ids(reserved: str) -> None:
    """Reserved ids (``"horde"`` / ``"any"``) cannot be registered."""
    registry = ModelProviderRegistry()
    with pytest.raises(ValueError, match="reserved"):
        registry.register(StubProvider(reserved, {}))


def test_registry_rejects_empty_source_id() -> None:
    """An empty source id is rejected at registration."""
    registry = ModelProviderRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(StubProvider("", {}))


def test_query_default_source_is_canonical_only(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """Without a ``source`` argument, only canonical records are returned."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}

    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    names = {record.name for record in manager.query(MODEL_REFERENCE_CATEGORY.image_generation).to_list()}
    assert names == {"horde_model"}


def test_query_specific_source_returns_only_provider_records(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """Selecting a provider id yields just that provider's records."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    results = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source="civitai").to_list_with_source()
    assert results == [(results[0][0], "civitai")]
    assert results[0][0].name == "provider_model"


def test_query_any_source_merges_canonical_and_providers(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """``"any"`` merges canonical plus every registered provider."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    grouped = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE).group_by_source()
    assert grouped == {
        HORDE_SOURCE_ID: [grouped[HORDE_SOURCE_ID][0]],
        "civitai": [grouped["civitai"][0]],
    }
    assert grouped[HORDE_SOURCE_ID][0].name == "horde_model"
    assert grouped["civitai"][0].name == "provider_model"


def test_canonical_wins_collision_and_duplicate_detection(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """On a name collision canonical wins, and the collision is still detectable."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"shared": _make_image_model("shared", nsfw=False)}
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"shared": _make_image_model("shared", nsfw=True)}},
        ),
    )

    query = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE)
    results = query.to_list_with_source()

    assert len(results) == 1
    winning_record, winning_source = results[0]
    assert winning_source == HORDE_SOURCE_ID
    assert isinstance(winning_record, ImageGenerationModelRecord)
    assert winning_record.nsfw is False

    assert query.has_duplicate_names() is True
    assert query.duplicate_names() == {"shared": [HORDE_SOURCE_ID, "civitai"]}


def test_unknown_explicit_source_raises(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """Explicitly naming an unregistered source id raises ``ValueError``."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {}

    with pytest.raises(ValueError, match="No provider registered"):
        manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source="does_not_exist")


def test_provider_error_is_isolated(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """A provider raising during fetch is skipped, not propagated."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "broken",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"x": _make_image_model("x")}},
            raise_on_fetch=True,
        ),
    )
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    names = {
        record.name for record in manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE).to_list()
    }
    assert names == {"horde_model", "provider_model"}


def test_source_status_distinguishes_ok_empty_and_error(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """``source_status`` reports each selected source as ok/empty/error."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "broken",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"x": _make_image_model("x")}},
            raise_on_fetch=True,
        ),
    )
    manager.register_provider(
        StubProvider("empty_src", {MODEL_REFERENCE_CATEGORY.image_generation: {}}),
    )
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    query = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE)

    assert query.source_status() == {
        HORDE_SOURCE_ID: "ok",
        "broken": "error",
        "empty_src": "empty",
        "civitai": "ok",
    }
    assert query.failed_sources() == ["broken"]

    # The outcome map survives fluent chaining (filters never touch provenance).
    assert query.where(nsfw=False).source_status() == query.source_status()


def test_source_status_canonical_only_is_derived(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """A default (canonical-only) query derives its status from record presence."""
    manager, canonical = provider_env

    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    assert manager.query(MODEL_REFERENCE_CATEGORY.image_generation).source_status() == {HORDE_SOURCE_ID: "ok"}
    assert manager.query(MODEL_REFERENCE_CATEGORY.image_generation).failed_sources() == []

    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {}
    assert manager.query(MODEL_REFERENCE_CATEGORY.image_generation).source_status() == {HORDE_SOURCE_ID: "empty"}


def test_get_model_reference_threads_source(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """``get_model_reference`` merges sources just like the query API."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    merged = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE)
    assert set(merged) == {"horde_model", "provider_model"}

    canonical_only = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
    assert set(canonical_only) == {"horde_model"}


@pytest.mark.asyncio
async def test_get_model_reference_async_threads_source(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """The async read path honours the ``source`` selector via async fetch."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
        ),
    )

    merged = await manager.get_model_reference_async(
        MODEL_REFERENCE_CATEGORY.image_generation,
        source=ANY_SOURCE,
    )
    assert set(merged) == {"horde_model", "provider_model"}


@pytest.fixture
def custom_misc_record_type() -> Generator[type[GenericModelRecord]]:
    """Register a user-defined record subclass for ``miscellaneous`` and restore after."""
    previous = MODEL_RECORD_TYPE_LOOKUP.get(MODEL_REFERENCE_CATEGORY.miscellaneous)

    @register_record_type(MODEL_REFERENCE_CATEGORY.miscellaneous)
    class CustomMiscRecord(GenericModelRecord):
        record_type: str | MODEL_REFERENCE_CATEGORY = MODEL_REFERENCE_CATEGORY.miscellaneous
        model_classification: ModelClassification = ModelClassification(
            domain=MODEL_DOMAIN.image,
            purpose=MODEL_PURPOSE.generation,
        )
        custom_field: str = "default"

    try:
        yield CustomMiscRecord
    finally:
        if previous is None:
            MODEL_RECORD_TYPE_LOOKUP.pop(MODEL_REFERENCE_CATEGORY.miscellaneous, None)
        else:
            MODEL_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.miscellaneous] = previous


def test_custom_subclassed_record_type_is_queryable(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
    custom_misc_record_type: type[GenericModelRecord],
) -> None:
    """A user can subclass GenericModelRecord, register it, and query provider data of that type."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.miscellaneous] = {}

    custom_record = custom_misc_record_type(
        name="custom_model",
        record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.generation),
    )
    manager.register_provider(
        StubProvider(
            "third_party",
            {MODEL_REFERENCE_CATEGORY.miscellaneous: {"custom_model": custom_record}},
        ),
    )

    results = manager.query(MODEL_REFERENCE_CATEGORY.miscellaneous, source="third_party").to_list()
    assert len(results) == 1
    assert isinstance(results[0], custom_misc_record_type)
    assert results[0].name == "custom_model"


def test_list_and_get_provider(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """Manager-level provider introspection mirrors the registry."""
    manager, _canonical = provider_env
    provider = StubProvider("civitai", {})
    manager.register_provider(provider)

    assert manager.list_providers() == ["civitai"]
    assert manager.get_provider("civitai") is provider
    assert manager.unregister_provider("civitai") is True
    assert manager.list_providers() == []


def test_lora_provider(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """Test a more complex provider with a managed_elsewhere record type and multiple sources."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.lora] = {
        "horde_lora": _make_lora_model("horde_lora"),
    }
    manager.register_provider(
        StubProvider(
            "civitai",
            {
                MODEL_REFERENCE_CATEGORY.lora: {
                    "provider_lora": _make_lora_model("provider_lora"),
                },
            },
        ),
    )

    any_results = manager.query(MODEL_REFERENCE_CATEGORY.lora, source=ANY_SOURCE).to_list_with_source()
    assert any_results == [(any_results[0][0], HORDE_SOURCE_ID), (any_results[1][0], "civitai")]
    assert any_results[0][0].name == "horde_lora"
    assert any_results[1][0].name == "provider_lora"

    specific_results = manager.query(MODEL_REFERENCE_CATEGORY.lora, source="civitai").to_list_with_source()
    assert specific_results == [(specific_results[0][0], "civitai")]
    assert specific_results[0][0].name == "provider_lora"

    assert isinstance(any_results[0][0], LoraModelRecord)
    assert isinstance(specific_results[0][0], LoraModelRecord)

    assert any_results[0][0].trigger == ["LoraSDXLTrigger"]
    assert specific_results[0][0].trigger == ["LoraSDXLTrigger"]

    provider = manager.get_provider("civitai")
    assert provider is not None
    assert isinstance(provider, StubProvider)
    assert provider.fetch_calls == 2


def test_static_provider_serves_prebuilt_records(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """StaticModelProvider exposes already-built records and merges like any provider."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}

    provider = StaticModelProvider(
        "civitai",
        {MODEL_REFERENCE_CATEGORY.image_generation: {"provider_model": _make_image_model("provider_model")}},
    )
    assert provider.provided_categories() == {MODEL_REFERENCE_CATEGORY.image_generation}
    assert provider.fetch_category(MODEL_REFERENCE_CATEGORY.text_generation) is None

    manager.register_provider(provider)
    names = {
        record.name for record in manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source=ANY_SOURCE).to_list()
    }
    assert names == {"horde_model", "provider_model"}


def test_static_provider_from_raw_validates_and_injects_name(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """from_raw builds typed records from plain dicts, injecting the mapping key as name."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {}

    provider = StaticModelProvider.from_raw(
        "civitai",
        {
            MODEL_REFERENCE_CATEGORY.image_generation: {
                "provider_model": {
                    "baseline": KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
                    "nsfw": True,
                },
            },
        },
    )
    manager.register_provider(provider)

    results = manager.query(MODEL_REFERENCE_CATEGORY.image_generation, source="civitai").to_list()
    assert len(results) == 1
    record = results[0]
    assert isinstance(record, ImageGenerationModelRecord)
    assert record.name == "provider_model"
    assert record.nsfw is True


def test_static_provider_from_raw_raises_on_invalid_record() -> None:
    """An invalid raw record surfaces a pydantic ValidationError from from_raw."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StaticModelProvider.from_raw(
            "civitai",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"bad": {"baseline": "NotARealBaseline"}}},
        )


@pytest.mark.parametrize("reserved", [HORDE_SOURCE_ID, ANY_SOURCE, ""])
def test_static_provider_rejects_invalid_source_id(reserved: str) -> None:
    """StaticModelProvider validates its source id on construction."""
    with pytest.raises(ValueError):
        StaticModelProvider(reserved, {})


def test_ordered_selector_lets_provider_override_canonical(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """An explicit ``[provider, "horde"]`` selector lets the provider win a name collision.

    This is the precedence used to surface "beta" models over canonical ones.
    """
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"shared": _make_image_model("shared", nsfw=False)}
    manager.register_provider(
        StubProvider(
            "pending",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"shared": _make_image_model("shared", nsfw=True)}},
        ),
    )

    merged = manager.get_model_reference(
        MODEL_REFERENCE_CATEGORY.image_generation,
        source=["pending", HORDE_SOURCE_ID],
    )

    shared = merged["shared"]
    assert isinstance(shared, ImageGenerationModelRecord)
    assert shared.nsfw is True  # the provider (listed first) wins


def test_ordered_selector_canonical_first_keeps_canonical(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """``["horde", provider]`` keeps canonical-wins; ordering controls precedence."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"shared": _make_image_model("shared", nsfw=False)}
    manager.register_provider(
        StubProvider(
            "pending",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"shared": _make_image_model("shared", nsfw=True)}},
        ),
    )

    merged = manager.get_model_reference(
        MODEL_REFERENCE_CATEGORY.image_generation,
        source=[HORDE_SOURCE_ID, "pending"],
    )

    shared = merged["shared"]
    assert isinstance(shared, ImageGenerationModelRecord)
    assert shared.nsfw is False  # canonical (listed first) wins


def test_ordered_selector_adds_provider_only_models(
    provider_env: tuple[ModelReferenceManager, CanonicalView],
) -> None:
    """A provider-only model is additive regardless of override ordering."""
    manager, canonical = provider_env
    canonical[MODEL_REFERENCE_CATEGORY.image_generation] = {"horde_model": _make_image_model("horde_model")}
    manager.register_provider(
        StubProvider(
            "pending",
            {MODEL_REFERENCE_CATEGORY.image_generation: {"beta_model": _make_image_model("beta_model")}},
        ),
    )

    merged = manager.get_model_reference(
        MODEL_REFERENCE_CATEGORY.image_generation,
        source=["pending", HORDE_SOURCE_ID],
    )
    assert set(merged) == {"horde_model", "beta_model"}
