"""Scenario 1: Building a Model Browser UI.

This example demonstrates how to fetch and organize model data
for a web UI, with filtering and grouping capabilities.
"""


from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    ModelReferenceManager,
)
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


def get_model_browser_data() -> dict[str, list[dict]]:
    """Fetch and organize image generation models for a browser UI.

    Returns a dictionary grouping models by baseline, with each model
    containing display-friendly information.
    """
    # Initialize manager
    manager = ModelReferenceManager()

    # Fetch all image generation models
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Group by baseline
    grouped: dict[str, list[dict]] = {}

    for _model_name, model in models.items():
        # Type-safe access to ImageGenerationModelRecord fields
        baseline = model.baseline

        if baseline not in grouped:
            grouped[baseline] = []

        # Prepare model data for UI
        model_data = {
            "name": model.name,
            "description": model.description or "No description available",
            "baseline": baseline,
            "style": model.style.value if model.style else "unknown",
            "nsfw": model.nsfw,
            "tags": model.tags or [],
            "homepage": model.homepage,
            "trigger_words": model.trigger or [],
        }

        grouped[baseline].append(model_data)

    return grouped


def filter_models_by_style(
    style: MODEL_STYLE,
) -> list[ImageGenerationModelRecord]:
    """Get all image generation models matching a specific style.

    Args:
        style: The MODEL_STYLE to filter by (anime, realistic, etc.)

    Returns:
        List of models matching the specified style
    """
    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Filter models by style - type-safe comparison
    return [
        model for model in models.values()
        if model.style == style
    ]



def get_safe_models_only() -> dict[str, ImageGenerationModelRecord]:
    """Get only SFW (Safe For Work) image generation models.

    Returns:
        Dictionary of model names to ImageGenerationModelRecord objects
        for models where nsfw=False
    """
    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Filter out NSFW models
    return {
        name: model for name, model in models.items()
        if not model.nsfw
    }



def main() -> None:
    """Run the model browser examples."""
    # Get organized data for UI
    print("=== Fetching Models for Browser UI ===")
    browser_data = get_model_browser_data()

    for baseline, models in browser_data.items():
        print(f"\n{baseline}: {len(models)} models")
        # Show first 3 models in each category
        for model in models[:3]:
            print(f"  - {model['name']} ({model['style']})")

    # Get anime models
    print("\n=== Anime Style Models ===")
    anime_models = filter_models_by_style(MODEL_STYLE.anime)
    print(f"Found {len(anime_models)} anime models")
    for model in anime_models[:5]:
        print(f"  - {model.name}: {model.description}")

    # Get safe models
    print("\n=== Safe (SFW) Models ===")
    safe_models = get_safe_models_only()
    print(f"Found {len(safe_models)} SFW models")


if __name__ == "__main__":
    main()
