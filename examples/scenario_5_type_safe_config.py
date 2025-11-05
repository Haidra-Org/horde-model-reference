"""Scenario 5: Type-Safe Configuration.

This example demonstrates using Pydantic models for type-safe
application configuration with validation.
"""


from pydantic import BaseModel, Field, field_validator

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    ModelReferenceManager,
)


class WorkerConfig(BaseModel):
    """Type-safe configuration for an AI-Horde worker.

    Uses Pydantic for validation and type safety.
    """

    worker_name: str = Field(
        ...,
        min_length=1,
        description="Name of the worker"
    )

    supported_models: list[str] = Field(
        default_factory=list,
        description="List of model names this worker supports"
    )

    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of concurrent generation jobs"
    )

    supported_baselines: list[str] = Field(
        default_factory=lambda: ["stable_diffusion_xl"],
        description="Baselines supported by this worker"
    )

    filter_nsfw: bool = Field(
        default=True,
        description="Whether to filter out NSFW models"
    )

    preferred_styles: list[MODEL_STYLE] = Field(
        default_factory=list,
        description="Preferred model styles to prioritize"
    )

    @field_validator("supported_models")
    @classmethod
    def validate_models_exist(cls, v: list[str]) -> list[str]:
        """Validate that all specified models exist in the reference."""
        if not v:
            return v

        manager = ModelReferenceManager()
        all_models = manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        for model_name in v:
            if model_name not in all_models:
                raise ValueError(f"Model '{model_name}' not found in model reference")

        return v

    def get_available_models(self) -> list[str]:
        """Get the list of models this worker can serve based on configuration.

        Returns:
            List of model names
        """
        manager = ModelReferenceManager()
        all_models = manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        available = []
        for name, model in all_models.items():
            # If specific models are configured, only use those
            if self.supported_models and name not in self.supported_models:
                continue

            # Check baseline support
            if model.baseline not in self.supported_baselines:
                continue

            # Filter NSFW if requested
            if self.filter_nsfw and model.nsfw:
                continue

            # Filter by preferred styles if specified
            if self.preferred_styles and model.style not in self.preferred_styles:
                continue

            available.append(name)

        return available


def main() -> None:
    """Run the type-safe configuration examples."""
    # Create a type-safe configuration
    config = WorkerConfig(
        worker_name="my-anime-worker",
        supported_baselines=["stable_diffusion_xl"],
        filter_nsfw=True,
        preferred_styles=[MODEL_STYLE.anime],
        max_concurrent_jobs=2
    )

    print("=== Worker Configuration ===")
    print(f"Worker Name: {config.worker_name}")
    print(f"Max Jobs: {config.max_concurrent_jobs}")
    print(f"Filter NSFW: {config.filter_nsfw}")
    print(f"Preferred Styles: {[s.value for s in config.preferred_styles]}")

    # Get available models based on config
    available = config.get_available_models()
    print(f"\n{len(available)} models match configuration:")
    for model_name in available[:10]:
        print(f"  - {model_name}")

    # Try to create invalid config (will raise validation error)
    print("\n=== Testing Validation ===")
    try:
        WorkerConfig(
            worker_name="test",
            supported_models=["non_existent_model_xyz_123"],
        )
    except ValueError as e:
        print(f"✓ Validation caught error: {e}")


if __name__ == "__main__":
    main()
