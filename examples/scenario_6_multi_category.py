"""
Scenario 6: Working with Multiple Categories

This example demonstrates working with different model categories
(text, image, ControlNet, utility models, etc.)
"""

from typing import Dict, Any
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
)


class MultiCategoryModelExplorer:
    """
    Explore models across different categories.
    """

    def __init__(self):
        self.manager = ModelReferenceManager()

    def get_category_summary(self) -> Dict[str, int]:
        """
        Get a count of models in each category.

        Returns:
            Dictionary mapping category names to model counts
        """
        summary = {}

        # Skip categories that don't have data yet
        skip_categories = {
            MODEL_REFERENCE_CATEGORY.video_generation,
            MODEL_REFERENCE_CATEGORY.audio_generation,
        }

        for category in MODEL_REFERENCE_CATEGORY:
            if category in skip_categories:
                summary[category.value] = 0
                continue

            try:
                models = self.manager.get_model_reference(category)
                summary[category.value] = len(models)
            except Exception as e:
                print(f"Warning: Could not fetch {category}: {e}")
                summary[category.value] = 0

        return summary

    def get_text_generation_models(self) -> Dict[str, Any]:
        """
        Get information about text generation models (LLMs).

        Returns:
            Dictionary of model names to their info
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.text_generation
        )

        model_info = {}
        for name, model in models.items():
            model_info[name] = {
                "name": model.name,
                "description": model.description,
                "parameters": getattr(model, "parameters_count", "Unknown"),
                "context_length": getattr(model, "context_length", "Unknown"),
            }

        return model_info

    def get_controlnet_models(self) -> Dict[str, Any]:
        """
        Get information about ControlNet models.

        Returns:
            Dictionary of ControlNet model info
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.controlnet
        )

        controlnet_info = {}
        for name, model in models.items():
            controlnet_info[name] = {
                "name": model.name,
                "description": model.description,
                "style": getattr(model, "style", "Unknown"),
            }

        return controlnet_info

    def get_utility_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all utility models (CLIP, BLIP, upscalers, etc.)

        Returns:
            Dictionary organized by utility type
        """
        utility_categories = [
            MODEL_REFERENCE_CATEGORY.clip,
            MODEL_REFERENCE_CATEGORY.blip,
            MODEL_REFERENCE_CATEGORY.esrgan,
            MODEL_REFERENCE_CATEGORY.gfpgan,
            MODEL_REFERENCE_CATEGORY.codeformer,
        ]

        utilities = {}
        for category in utility_categories:
            try:
                models = self.manager.get_model_reference(category)
                utilities[category.value] = {
                    name: {
                        "name": model.name,
                        "description": model.description,
                    }
                    for name, model in models.items()
                }
            except Exception:
                utilities[category.value] = {}

        return utilities


def main():
    """Run the multi-category exploration examples."""
    explorer = MultiCategoryModelExplorer()

    # Get summary of all categories
    print("=== Model Category Summary ===")
    summary = explorer.get_category_summary()
    for category, count in sorted(summary.items()):
        print(f"{category:<25} {count:>5} models")

    # Explore text generation models
    print("\n=== Text Generation Models (first 5) ===")
    text_models = explorer.get_text_generation_models()
    for name, info in list(text_models.items())[:5]:
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Parameters: {info['parameters']}")

    # Explore ControlNet models
    print("\n=== ControlNet Models ===")
    controlnet_models = explorer.get_controlnet_models()
    print(f"Found {len(controlnet_models)} ControlNet models")
    for name in list(controlnet_models.keys())[:5]:
        print(f"  - {name}")

    # Explore utility models
    print("\n=== Utility Models ===")
    utilities = explorer.get_utility_models()
    for category, models in utilities.items():
        print(f"{category}: {len(models)} models")


if __name__ == "__main__":
    main()
