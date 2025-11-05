"""Scenario 4: Model Capability Checker.

This example demonstrates how to check model capabilities and requirements.
"""

from typing import Any

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
)
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


class ModelCapabilityChecker:
    """Check model capabilities and requirements."""

    def __init__(self) -> None:
        """Initialize the model capability checker."""
        self.manager = ModelReferenceManager()

    def check_inpainting_support(self) -> list[str]:
        """Get all models that support inpainting.

        Returns:
            List of model names that support inpainting
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        return [
            name for name, model in models.items()
            if model.inpainting
        ]


    def check_model_requirements(
        self,
        model_name: str
    ) -> dict[str, Any] | None:
        """Get the requirements for a specific model.

        Args:
            model_name: Name of the model to check

        Returns:
            Dictionary of requirements, or None if model not found
        """
        try:
            model = self.manager.get_model(
                MODEL_REFERENCE_CATEGORY.image_generation,
                model_name
            )
        except KeyError:
            return None

        return {
            "baseline": model.baseline,
            "min_bridge_version": model.min_bridge_version,
            "inpainting": model.inpainting,
            "nsfw": model.nsfw,
            "requirements": model.requirements or {},
            "size_on_disk": model.size_on_disk_bytes,
        }

    def find_models_by_baseline(
        self,
        baseline: str
    ) -> list[ImageGenerationModelRecord]:
        """Find all models based on a specific baseline.

        Args:
            baseline: The baseline to search for (e.g., "stable_diffusion_xl")

        Returns:
            List of models with the specified baseline
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        return [
            model for model in models.values()
            if model.baseline == baseline
        ]


    def get_models_with_tags(
        self,
        required_tags: list[str]
    ) -> list[ImageGenerationModelRecord]:
        """Find models that have all specified tags.

        Args:
            required_tags: List of tags that models must have

        Returns:
            List of models containing all required tags
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        matching = []
        for model in models.values():
            if not model.tags:
                continue

            # Check if all required tags are present
            if all(tag in model.tags for tag in required_tags):
                matching.append(model)

        return matching


def main() -> None:
    """Run the capability checker examples."""
    checker = ModelCapabilityChecker()

    # Check inpainting support
    print("=== Inpainting Support ===")
    inpainting_models = checker.check_inpainting_support()
    print(f"Found {len(inpainting_models)} models with inpainting support")
    for name in inpainting_models[:5]:
        print(f"  - {name}")

    # Check specific model requirements
    print("\n=== Model Requirements ===")
    model_name = "stable_diffusion_xl"
    requirements = checker.check_model_requirements(model_name)
    if requirements:
        print(f"Requirements for {model_name}:")
        for key, value in requirements.items():
            print(f"  {key}: {value}")

    # Find SDXL models
    print("\n=== SDXL-based Models ===")
    sdxl_models = checker.find_models_by_baseline("stable_diffusion_xl")
    print(f"Found {len(sdxl_models)} SDXL-based models")
    for model in sdxl_models[:5]:
        print(f"  - {model.name}")

    # Find anime models
    print("\n=== Anime Models ===")
    anime_models = checker.get_models_with_tags(["anime"])
    print(f"Found {len(anime_models)} anime models")
    for model in anime_models[:5]:
        print(f"  - {model.name}: {model.description}")


if __name__ == "__main__":
    main()
