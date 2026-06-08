"""Register a third-party provider and read it alongside canonical data.

Run with:

    uv run examples/04_register_provider.py

Providers are read-only sources that augment (never overwrite) the canonical
horde data. Consumers opt in per-call with the `source=` argument.
"""

from horde_model_reference import (
    ANY_SOURCE,
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    StaticModelProvider,
)


def main() -> None:
    """Register a static provider and read it via source selection."""
    manager = ModelReferenceManager()

    # `from_raw` validates plain dicts against the category's record type and
    # injects each mapping key as the record `name`.
    provider = StaticModelProvider.from_raw(
        "civitai",
        {
            MODEL_REFERENCE_CATEGORY.image_generation: {
                "my_cool_model": {"baseline": "stable_diffusion_xl", "nsfw": False},
            },
        },
    )
    manager.register_provider(provider, replace=True)
    print(f"Registered providers: {manager.list_providers()}")

    # Default source is "horde" (canonical only) - the provider is invisible.
    canonical_only = manager.query("image_generation").count()
    print(f"Canonical-only image model count: {canonical_only}")

    # source=ANY_SOURCE merges canonical + every provider; canonical wins
    # any name collisions.
    with_providers = manager.query("image_generation", source=ANY_SOURCE)
    print(f"With providers image model count: {with_providers.count()}")

    # Inspect provenance and collisions.
    collisions = manager.query("image_generation", source=ANY_SOURCE).duplicate_names()
    print(f"Colliding names across sources: {collisions or 'none'}")


if __name__ == "__main__":
    main()
