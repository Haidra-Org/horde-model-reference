from __future__ import annotations

from collections.abc import Generator
from enum import Enum

import pytest

from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.meta_consts import (
    CONTROLNET_STYLE,
    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP,
    KNOWN_IMAGE_GENERATION_BASELINE,
    KNOWN_TAGS,
    KNOWN_TEXT_BACKENDS,
    MODEL_CLASSIFICATION_LOOKUP,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    TEXT_BACKENDS,
    BaselineDescriptor,
    CategoryDescriptor,
    _matching_image_baseline_exists,
    alternative_sdxl_baseline_names,
    get_all_registered_baselines,
    get_all_registered_categories,
    get_baseline_descriptor,
    get_baseline_native_resolution,
    get_baselines_by_resolution,
    get_category_descriptor,
    get_github_image_categories,
    get_github_text_categories,
    get_known_tags,
    get_model_classification,
    get_no_legacy_format_categories,
    is_known_controlnet_style,
    is_known_image_baseline,
    is_known_model_domain,
    is_known_model_purpose,
    is_known_model_style,
    is_known_tag,
    is_known_text_backend,
    register_category,
    register_controlnet_style,
    register_image_baseline,
    register_model_domain,
    register_model_purpose,
    register_model_style,
    register_tag,
    register_text_backend,
)
from horde_model_reference.registries import DescriptorRegistry, EnumRegistry
from horde_model_reference.text_backend_names import (
    get_model_name_variants,
    has_legacy_text_backend_prefix,
    strip_backend_prefix,
)


@pytest.fixture(autouse=True)
def reset_registries() -> Generator[None, None, None]:
    """Snapshot registry state and restore after each test to avoid cross-test coupling."""
    import copy

    import horde_model_reference.meta_consts as mc

    category_snapshot = mc._CATEGORY_REGISTRY.all()
    baseline_snapshot = mc._IMAGE_BASELINE_REGISTRY.all()

    tag_snapshot = set(mc._TAG_REGISTRY._known)
    domain_snapshot = set(mc._MODEL_DOMAIN_REGISTRY._known)
    purpose_snapshot = set(mc._MODEL_PURPOSE_REGISTRY._known)
    style_snapshot = set(mc._MODEL_STYLE_REGISTRY._known)
    controlnet_snapshot = set(mc._CONTROLNET_STYLE_REGISTRY._known)
    text_backend_snapshot = set(mc._TEXT_BACKEND_REGISTRY._known)

    derived_category_lists = (
        list(mc.github_image_model_reference_categories),
        list(mc.github_text_model_reference_categories),
        list(mc.no_legacy_format_available_categories),
        list(mc.categories_managed_elsewhere),
        dict(mc.MODEL_CLASSIFICATION_LOOKUP),
    )

    derived_baseline = (
        dict(mc.IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP),
        dict(mc._ALTERNATIVE_NAME_TO_BASELINE),
        list(mc.alternative_sdxl_baseline_names),
    )

    yield

    mc._CATEGORY_REGISTRY._data = copy.deepcopy(category_snapshot)
    mc._CATEGORY_REGISTRY._init_complete = True
    mc._rebuild_category_derived_data(mc._CATEGORY_REGISTRY._data)

    mc._IMAGE_BASELINE_REGISTRY._data = copy.deepcopy(baseline_snapshot)
    mc._IMAGE_BASELINE_REGISTRY._init_complete = True
    mc._rebuild_baseline_derived_data(mc._IMAGE_BASELINE_REGISTRY._data)
    mc.alternative_sdxl_baseline_names = list(
        mc._IMAGE_BASELINE_REGISTRY.get(mc.KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl).alternative_names
    )

    mc._TAG_REGISTRY._known.clear()
    mc._TAG_REGISTRY._known.update(tag_snapshot)
    mc._MODEL_DOMAIN_REGISTRY._known.clear()
    mc._MODEL_DOMAIN_REGISTRY._known.update(domain_snapshot)
    mc._MODEL_PURPOSE_REGISTRY._known.clear()
    mc._MODEL_PURPOSE_REGISTRY._known.update(purpose_snapshot)
    mc._MODEL_STYLE_REGISTRY._known.clear()
    mc._MODEL_STYLE_REGISTRY._known.update(style_snapshot)
    mc._CONTROLNET_STYLE_REGISTRY._known.clear()
    mc._CONTROLNET_STYLE_REGISTRY._known.update(controlnet_snapshot)
    mc._TEXT_BACKEND_REGISTRY._known.clear()
    mc._TEXT_BACKEND_REGISTRY._known.update(text_backend_snapshot)

    (
        mc.github_image_model_reference_categories,
        mc.github_text_model_reference_categories,
        mc.no_legacy_format_available_categories,
        mc.categories_managed_elsewhere,
        mc.MODEL_CLASSIFICATION_LOOKUP,
    ) = derived_category_lists

    (
        mc.IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP,
        mc._ALTERNATIVE_NAME_TO_BASELINE,
        mc.alternative_sdxl_baseline_names,
    ) = derived_baseline


class TestEnumRegistry:
    """Covers the enum-backed registry to guard API expectations and live-view behavior."""

    def test_registers_and_is_known(self) -> None:
        """Registry accepts strings, avoids duplicates, and exposes stable values view."""
        registry = EnumRegistry(["a"])

        assert registry.is_known("a")

        registry.register("b")
        assert registry.is_known("b")

        registry.register("b")  # idempotent
        assert registry.values() == {"a", "b"}

    def test_mutable_values_is_live_view(self) -> None:
        """mutable_values returns a live set that reflects later registrations."""
        registry = EnumRegistry(["seed"])

        live = registry.mutable_values()
        registry.register("later")

        assert "later" in live

    def test_accepts_enum_members(self) -> None:
        """Enum members can be registered and queried without conversion by callers."""

        class Demo(Enum):
            foo = "foo"

        registry = EnumRegistry([Demo.foo])
        assert registry.is_known(Demo.foo)

        registry.register("bar")
        assert registry.is_known("bar")


class TestDescriptorRegistry:
    """Validates descriptor registry rebuild timing, idempotence, and duplicate guards."""

    def test_rebuild_on_finalize_and_register(self) -> None:
        """Rebuild hook runs on finalize and subsequent register calls when finalized."""
        rebuild_calls: list[dict[str, int]] = []

        def rebuild(data: dict[str, int]) -> None:
            rebuild_calls.append(dict(data))

        registry = DescriptorRegistry[str, int](rebuild)

        registry.register("k1", 1)
        assert rebuild_calls == []

        registry.finalize()
        assert rebuild_calls[-1] == {"k1": 1}

        registry.register("k2", 2)
        assert rebuild_calls[-1] == {"k1": 1, "k2": 2}
        assert registry.get("k2") == 2
        assert registry.contains("k1")

    def test_rejects_duplicate_registration(self) -> None:
        """Duplicate keys raise to prevent silent clobbering."""
        registry = DescriptorRegistry[str, int](lambda _: None)
        registry.register("k1", 1)

        with pytest.raises(ValueError):
            registry.register("k1", 2)

    def test_finalize_idempotent(self) -> None:
        """Calling finalize multiple times does not trigger extra rebuilds."""
        rebuild_calls: list[dict[str, int]] = []

        def rebuild(data: dict[str, int]) -> None:
            rebuild_calls.append(dict(data))

        registry = DescriptorRegistry[str, int](rebuild)
        registry.register("k1", 1)

        registry.finalize()
        registry.finalize()

        assert len(rebuild_calls) == 1
        assert rebuild_calls[0] == {"k1": 1}


class TestMetaConstsInvariants:
    """Guards derived data and live globals populated from meta_consts registries."""

    def test_category_derived_lists_populated(self) -> None:
        """Built-in categories should be present in derived lists after finalize."""
        assert MODEL_REFERENCE_CATEGORY.image_generation in get_github_image_categories()
        assert MODEL_REFERENCE_CATEGORY.text_generation in get_github_text_categories()
        assert MODEL_REFERENCE_CATEGORY.lora in get_no_legacy_format_categories()

    def test_category_classification_lookup_complete(self) -> None:
        """Every category should have a classification entry built from the registry."""
        missing = [c for c in MODEL_REFERENCE_CATEGORY if c not in MODEL_CLASSIFICATION_LOOKUP]
        assert not missing

    def test_baseline_derived_data_initialized(self) -> None:
        """Baseline registry should hydrate alt-name map and native resolutions."""
        assert "SDXL" in alternative_sdxl_baseline_names
        assert is_known_image_baseline("SDXL")
        assert IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP

    def test_style_and_tag_registries_initialized(self) -> None:
        """Initial style/tag seeds should be recognized by the registry helpers."""
        assert is_known_model_style("anime")
        assert is_known_tag("anime")

    def test_register_style_and_tag_updates_globals(self) -> None:
        """Registering new styles/tags should update both lookups and live globals."""
        register_model_style("brand_new_style")
        register_tag("brand_new_tag")

        assert is_known_model_style("brand_new_style")
        assert is_known_tag("brand_new_tag")

    def test_every_baseline_enum_member_is_registered(self) -> None:
        """Every KNOWN_IMAGE_GENERATION_BASELINE member must have a BaselineDescriptor."""
        for baseline in KNOWN_IMAGE_GENERATION_BASELINE:
            desc = get_baseline_descriptor(baseline)
            assert desc is not None

    def test_known_tags_contains_all_initial_tags(self) -> None:
        """The KNOWN_TAGS export must include all tags that were seeded at module load."""
        expected_seeds = {
            "anime",
            "manga",
            "cyberpunk",
            "tv show",
            "booru",
            "retro",
            "character",
            "hentai",
            "scenes",
            "low poly",
            "cg",
            "sketch",
            "high resolution",
            "landscapes",
            "comic",
            "cartoon",
            "painting",
            "game",
        }
        for tag in expected_seeds:
            assert tag in KNOWN_TAGS, f"Seed tag '{tag}' missing from KNOWN_TAGS"
            assert is_known_tag(tag), f"Seed tag '{tag}' not recognized by is_known_tag"


class TestControlNetStyleRegistry:
    """Every ControlNet style enum member should be discoverable and the registry extensible."""

    def test_all_enum_members_are_discoverable(self) -> None:
        """Each CONTROLNET_STYLE member must be recognized by is_known_controlnet_style."""
        for style in CONTROLNET_STYLE:
            assert is_known_controlnet_style(style), f"{style} not recognized"

    def test_registration_makes_new_style_discoverable(self) -> None:
        """Registering a custom ControlNet style makes it queryable."""
        register_controlnet_style("control_custom_test")
        assert is_known_controlnet_style("control_custom_test")

    def test_unknown_style_is_not_discoverable(self) -> None:
        """A never-registered value should not be recognized."""
        assert not is_known_controlnet_style("control_nonexistent_xyz")


class TestModelDomainRegistry:
    """Every model domain enum member should be discoverable and the registry extensible."""

    def test_all_enum_members_are_discoverable(self) -> None:
        """Built-in enum values must remain discoverable to preserve API contract stability."""
        for domain in MODEL_DOMAIN:
            assert is_known_model_domain(domain), f"{domain} not recognized"

    def test_registration_makes_new_domain_discoverable(self) -> None:
        """Runtime registrations must refresh the registry to allow downstream lookups."""
        register_model_domain("holographic")
        assert is_known_model_domain("holographic")

    def test_unknown_domain_is_not_discoverable(self) -> None:
        """Unregistered domains should not leak through discovery helpers."""
        assert not is_known_model_domain("quantum_xyz")


class TestModelPurposeRegistry:
    """Every model purpose enum member should be discoverable and the registry extensible."""

    def test_all_enum_members_are_discoverable(self) -> None:
        """Built-in enum values must remain discoverable to preserve API contract stability."""
        for purpose in MODEL_PURPOSE:
            assert is_known_model_purpose(purpose), f"{purpose} not recognized"

    def test_registration_makes_new_purpose_discoverable(self) -> None:
        """Runtime registrations must refresh the registry to allow downstream lookups."""
        register_model_purpose("alignment")
        assert is_known_model_purpose("alignment")

    def test_unknown_purpose_is_not_discoverable(self) -> None:
        """Unregistered purposes should not leak through discovery helpers."""
        assert not is_known_model_purpose("teleportation_xyz")


class TestTextBackendRegistry:
    """Text backend registration and discovery should behave consistently."""

    def test_all_enum_members_are_known(self) -> None:
        """Every TEXT_BACKENDS enum member should be recognized."""
        for backend in TEXT_BACKENDS:
            assert is_known_text_backend(backend.value), f"{backend} not recognized"

    def test_register_new_backend_makes_it_discoverable(self) -> None:
        register_text_backend("vllm_test")
        assert is_known_text_backend("vllm_test")

    def test_duplicate_registration_is_idempotent(self) -> None:
        """Re-registering an existing backend should not raise."""
        register_text_backend("aphrodite")
        assert is_known_text_backend("aphrodite")

    def test_unknown_backend_is_not_discoverable(self) -> None:
        assert not is_known_text_backend("nonexistent_backend_xyz")


class TestTextBackendPrefixFunctions:
    """Text backend prefix detection, stripping, and variant generation."""

    def test_recognizes_aphrodite_prefix(self) -> None:
        assert has_legacy_text_backend_prefix("aphrodite/SomeOrg/SomeModel")

    def test_recognizes_koboldcpp_prefix(self) -> None:
        assert has_legacy_text_backend_prefix("koboldcpp/SomeModel")

    def test_rejects_unprefixed_name(self) -> None:
        assert not has_legacy_text_backend_prefix("SomeOrg/SomeModel")

    def test_rejects_partial_prefix_match(self) -> None:
        """A name that merely contains 'aphrodite' as a substring should not match."""
        assert not has_legacy_text_backend_prefix("not_aphrodite_model")

    def test_strip_removes_aphrodite_prefix(self) -> None:
        assert strip_backend_prefix("aphrodite/Org/Model") == "Org/Model"

    def test_strip_removes_koboldcpp_prefix(self) -> None:
        assert strip_backend_prefix("koboldcpp/Model") == "Model"

    def test_strip_leaves_unprefixed_name_intact(self) -> None:
        assert strip_backend_prefix("Org/Model") == "Org/Model"

    def test_strip_is_idempotent(self) -> None:
        """Stripping an already-stripped name should return it unchanged."""
        once = strip_backend_prefix("koboldcpp/Model")
        twice = strip_backend_prefix(once)
        assert once == twice == "Model"

    def test_strip_handles_name_containing_slash_not_a_prefix(self) -> None:
        """A name with '/' that is not a known backend prefix should be left alone."""
        assert strip_backend_prefix("ReadyArt/Broken-Tutu-24B") == "ReadyArt/Broken-Tutu-24B"

    def test_variants_canonical_name_is_always_first(self) -> None:
        canonical = "SomeOrg/SomeModel-7B"
        variants = get_model_name_variants(canonical)
        assert variants[0] == canonical

    def test_variants_no_duplicates(self) -> None:
        """Variant lists should never contain duplicate entries."""
        for name in ["Org/Model-7B", "Model-7B", "a/b/c"]:
            variants = get_model_name_variants(name)
            assert len(variants) == len(set(variants)), f"Duplicates in variants for {name!r}: {variants}"

    def test_variants_with_org_prefix(self) -> None:
        """Name with org prefix should produce aphrodite, koboldcpp (short), and koboldcpp (sanitized) variants."""
        variants = get_model_name_variants("ReadyArt/Broken-Tutu-24B")
        assert "aphrodite/ReadyArt/Broken-Tutu-24B" in variants
        assert "koboldcpp/Broken-Tutu-24B" in variants
        assert "koboldcpp/ReadyArt_Broken-Tutu-24B" in variants

    def test_variants_without_org_prefix(self) -> None:
        """Name without org prefix should not produce a spurious sanitized variant."""
        variants = get_model_name_variants("Broken-Tutu-24B")
        assert variants.count("koboldcpp/Broken-Tutu-24B") == 1


class TestCategoryDescriptorAccessors:
    """Category descriptor lookup functions should return correct data and enforce contracts."""

    def test_known_category_returns_descriptor(self) -> None:
        desc = get_category_descriptor(MODEL_REFERENCE_CATEGORY.image_generation)
        assert desc.domain == MODEL_DOMAIN.image
        assert desc.purpose == MODEL_PURPOSE.generation

    def test_text_generation_category_is_text_domain(self) -> None:
        desc = get_category_descriptor(MODEL_REFERENCE_CATEGORY.text_generation)
        assert desc.domain == MODEL_DOMAIN.text

    def test_unknown_category_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_category_descriptor("nonexistent_category_xyz")

    def test_get_all_returns_copy(self) -> None:
        """Mutating the returned dict should not affect the registry."""
        all_cats = get_all_registered_categories()
        original_len = len(all_cats)
        all_cats["fake_category"] = get_category_descriptor(MODEL_REFERENCE_CATEGORY.clip)  # type: ignore[index]
        assert len(get_all_registered_categories()) == original_len

    def test_get_all_contains_every_enum_member(self) -> None:
        """Every enum category should remain registered after seed-time setup."""
        all_cats = get_all_registered_categories()
        for cat in MODEL_REFERENCE_CATEGORY:
            assert cat in all_cats, f"{cat} missing from get_all_registered_categories()"

    def test_filename_override_for_image_generation(self) -> None:
        """image_generation uses 'stable_diffusion.json' not the default '{category}.json'."""
        desc = get_category_descriptor(MODEL_REFERENCE_CATEGORY.image_generation)
        assert desc.filename_override == "stable_diffusion.json"

    def test_managed_elsewhere_flag(self) -> None:
        """Lora and ti are managed by external systems."""
        assert get_category_descriptor(MODEL_REFERENCE_CATEGORY.lora).managed_elsewhere is True
        assert get_category_descriptor(MODEL_REFERENCE_CATEGORY.ti).managed_elsewhere is True
        assert get_category_descriptor(MODEL_REFERENCE_CATEGORY.image_generation).managed_elsewhere is False

    def test_runtime_category_registration_updates_derived_state(self) -> None:
        """Registering a new category should rebuild derived lists and classification lookups."""
        register_category(
            "runtime_category",
            CategoryDescriptor(
                domain=MODEL_DOMAIN.text,
                purpose=MODEL_PURPOSE.miscellaneous,
                github_source=None,
                has_legacy_format=False,
                managed_elsewhere=False,
            ),
        )

        assert "runtime_category" in get_no_legacy_format_categories()
        classification = get_model_classification("runtime_category")
        assert classification.domain == MODEL_DOMAIN.text
        assert classification.purpose == MODEL_PURPOSE.miscellaneous

    def test_duplicate_category_registration_raises(self) -> None:
        """Registering the same category twice should surface a ValueError to callers."""
        register_category(
            "dup_category",
            CategoryDescriptor(
                domain=MODEL_DOMAIN.image,
                purpose=MODEL_PURPOSE.miscellaneous,
                github_source=None,
            ),
        )

        with pytest.raises(ValueError):
            register_category(
                "dup_category",
                CategoryDescriptor(
                    domain=MODEL_DOMAIN.image,
                    purpose=MODEL_PURPOSE.miscellaneous,
                    github_source=None,
                ),
            )


class TestBaselineDescriptorAccessors:
    """Baseline descriptor lookup functions should return correct data and enforce contracts."""

    def test_known_baseline_returns_descriptor(self) -> None:
        desc = get_baseline_descriptor(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)
        assert desc.native_resolution == 512

    def test_unknown_baseline_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_baseline_descriptor("nonexistent_baseline_xyz")

    def test_get_all_returns_copy(self) -> None:
        """Mutating the returned dict should not affect the registry."""
        all_bl = get_all_registered_baselines()
        original_len = len(all_bl)
        all_bl["fake_baseline"] = get_baseline_descriptor(KNOWN_IMAGE_GENERATION_BASELINE.flux_1)  # type: ignore[index]
        assert len(get_all_registered_baselines()) == original_len

    def test_get_all_contains_every_enum_member(self) -> None:
        """Every seeded baseline should stay discoverable after initialization."""
        all_bl = get_all_registered_baselines()
        for bl in KNOWN_IMAGE_GENERATION_BASELINE:
            assert bl in all_bl, f"{bl} missing from get_all_registered_baselines()"

    def test_native_resolution_sd1_is_512(self) -> None:
        assert get_baseline_native_resolution(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1) == 512

    def test_native_resolution_sdxl_is_1024(self) -> None:
        assert get_baseline_native_resolution(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl) == 1024

    def test_native_resolution_infer_raises_key_error(self) -> None:
        """The 'infer' baseline has no native resolution; lookup should raise."""
        with pytest.raises(KeyError):
            get_baseline_native_resolution(KNOWN_IMAGE_GENERATION_BASELINE.infer)

    def test_baselines_by_resolution_512(self) -> None:
        """Resolution 512 should include SD1 and SD2-512."""
        baselines_512 = get_baselines_by_resolution(512)
        assert KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1 in baselines_512
        assert KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512 in baselines_512

    def test_baselines_by_resolution_1024_includes_sdxl(self) -> None:
        baselines_1024 = get_baselines_by_resolution(1024)
        assert KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl in baselines_1024

    def test_baselines_by_resolution_nonexistent_returns_empty(self) -> None:
        assert get_baselines_by_resolution(99999) == []

    def test_runtime_baseline_registration_updates_derived_state(self) -> None:
        """Registering a new baseline should refresh resolution and alias lookups."""
        import horde_model_reference.meta_consts as mc

        register_image_baseline(
            "runtime_baseline",
            BaselineDescriptor(native_resolution=2048, alternative_names=("rb_test",)),
        )

        assert get_baseline_native_resolution("runtime_baseline") == 2048
        assert is_known_image_baseline("rb_test")
        assert "runtime_baseline" in get_baselines_by_resolution(2048)
        assert mc._ALTERNATIVE_NAME_TO_BASELINE["rb_test"] == "runtime_baseline"

    def test_duplicate_baseline_registration_raises(self) -> None:
        """Registering the same baseline twice should surface a ValueError to callers."""
        register_image_baseline("dup_baseline", BaselineDescriptor(native_resolution=512))

        with pytest.raises(ValueError):
            register_image_baseline("dup_baseline", BaselineDescriptor(native_resolution=512))


class TestMatchingImageBaseline:
    """Alternative name matching should correctly map human-friendly names to baselines."""

    def test_sdxl_alternative_matches(self) -> None:
        assert _matching_image_baseline_exists("SDXL", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl)

    def test_sd15_alternative_matches_sd1(self) -> None:
        assert _matching_image_baseline_exists("SD1.5", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)

    def test_canonical_name_matches_when_no_alternatives(self) -> None:
        """Baselines without alternative names should still match their own canonical name."""
        assert _matching_image_baseline_exists("flux_1", KNOWN_IMAGE_GENERATION_BASELINE.flux_1)

    def test_wrong_alternative_does_not_match(self) -> None:
        assert not _matching_image_baseline_exists("SDXL", KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1)

    def test_alternative_name_is_known_baseline(self) -> None:
        """Alternative names like 'SDXL' should be recognized by is_known_image_baseline."""
        assert is_known_image_baseline("SDXL")
        assert is_known_image_baseline("SD1.5")
        assert is_known_image_baseline("flux schnell")

    def test_unknown_string_is_not_known_baseline(self) -> None:
        assert not is_known_image_baseline("totally_unknown_baseline_xyz")


class TestUnifiedRegistryPatterns:
    """Phase 2: Verify all registries follow the same EnumRegistry-backed pattern."""

    def test_known_text_backends_is_live_view(self) -> None:
        """KNOWN_TEXT_BACKENDS should be a live set that reflects new registrations."""
        register_text_backend("test_live_view_backend")
        assert "test_live_view_backend" in KNOWN_TEXT_BACKENDS

    def test_known_tags_is_live_view(self) -> None:
        """KNOWN_TAGS should be a live set that reflects new registrations."""
        register_tag("test_live_view_tag")
        assert "test_live_view_tag" in KNOWN_TAGS

    def test_get_known_tags_returns_list(self) -> None:
        """get_known_tags accessor should return a sorted list."""
        tags = get_known_tags()
        assert isinstance(tags, list)
        assert tags == sorted(tags)

    def test_get_known_tags_reflects_registrations(self) -> None:
        """Registering a new tag should appear in subsequent get_known_tags() calls."""
        register_tag("zzz_phase2_test_tag")
        assert "zzz_phase2_test_tag" in get_known_tags()

    def test_get_known_tags_contains_all_seeds(self) -> None:
        """All initial seed tags must appear in get_known_tags()."""
        tags = get_known_tags()
        for seed in ("anime", "manga", "cyberpunk", "landscapes", "painting"):
            assert seed in tags

    def test_removed_globals_no_longer_exist(self) -> None:
        """Dead KNOWN_* globals (styles, domains, purposes, controlnet styles) should be removed."""
        import horde_model_reference.meta_consts as mc

        assert not hasattr(mc, "KNOWN_MODEL_STYLES"), "KNOWN_MODEL_STYLES should be removed"
        assert not hasattr(mc, "KNOWN_CONTROLNET_STYLES"), "KNOWN_CONTROLNET_STYLES should be removed"
        assert not hasattr(mc, "KNOWN_MODEL_DOMAINS"), "KNOWN_MODEL_DOMAINS should be removed"
        assert not hasattr(mc, "KNOWN_MODEL_PURPOSES"), "KNOWN_MODEL_PURPOSES should be removed"


class TestCategoryAccessorFunctions:
    """Phase 5: Accessor functions for category-derived globals."""

    def test_get_github_image_categories_returns_list(self) -> None:
        result = get_github_image_categories()
        assert isinstance(result, list)
        assert MODEL_REFERENCE_CATEGORY.image_generation in result

    def test_get_github_text_categories_returns_list(self) -> None:
        result = get_github_text_categories()
        assert isinstance(result, list)
        assert MODEL_REFERENCE_CATEGORY.text_generation in result

    def test_get_no_legacy_format_categories_returns_list(self) -> None:
        result = get_no_legacy_format_categories()
        assert isinstance(result, list)
        assert MODEL_REFERENCE_CATEGORY.lora in result

    def test_accessors_return_copies(self) -> None:
        """Mutating the returned list should not affect future calls."""
        image_cats = get_github_image_categories()
        original_len = len(image_cats)
        image_cats.append("fake_category")
        assert len(get_github_image_categories()) == original_len

    def test_image_and_text_categories_are_disjoint(self) -> None:
        """No category should be in both the image and text GitHub lists."""
        image = set(get_github_image_categories())
        text = set(get_github_text_categories())
        assert image.isdisjoint(text), f"Overlap: {image & text}"
