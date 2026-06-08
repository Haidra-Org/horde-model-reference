"""Quickstart: instantiate the manager and read model metadata.

Run with:

    uv run examples/01_quickstart.py

The first call fetches model reference data over the network (the PRIMARY
server at aihorde.net, with a GitHub fallback) and caches it in memory.
"""

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager


def main() -> None:
    """List image generation models and show detail for one."""
    # The manager is a singleton. Create it once and reuse it everywhere.
    manager = ModelReferenceManager()

    # Read every image generation model. Categories also accept the plain
    # string "image_generation" if you prefer not to import the enum.
    image_models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
    print(f"Found {len(image_models)} image generation models")

    for name, model in list(image_models.items())[:5]:
        print(f"  {name}: baseline={model.baseline}")

    # Look up a single model by name without crashing if it is missing.
    first_name = next(iter(image_models), None)
    if first_name is not None:
        model = manager.get_model_or_none(MODEL_REFERENCE_CATEGORY.image_generation, first_name)
        if model is not None:
            print(f"\nDetail for {first_name!r}:")
            print(f"  description: {model.description}")
            print(f"  downloads:   {model.download_count}")


if __name__ == "__main__":
    main()
