"""
Scenario 3: Model Downloader Tool

This example demonstrates how to create a CLI tool for downloading
and verifying AI models using horde-model-reference metadata.

Note: This is a demonstration. Actual downloading is commented out.
"""

from pathlib import Path
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
)


class ModelDownloader:
    """
    Download and verify AI models using horde-model-reference metadata.
    """

    def __init__(self, download_dir: Path):
        """
        Initialize the model downloader.

        Args:
            download_dir: Directory where models will be downloaded
        """
        self.download_dir = download_dir
        self.manager = ModelReferenceManager()

    def list_available_models(self, category: MODEL_REFERENCE_CATEGORY) -> None:
        """
        List all available models in a category.

        Args:
            category: The category to list models from
        """
        models = self.manager.get_model_reference(category)

        print(f"\nAvailable models in {category}:")
        print(f"{'Model Name':<40} {'Baseline':<30}")
        print("-" * 70)

        for name, model in sorted(models.items()):
            baseline = getattr(model, "baseline", "N/A")
            print(f"{name:<40} {baseline:<30}")

    def get_model_info(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> bool:
        """
        Display information about a specific model.

        Args:
            category: The model category
            model_name: Name of the model

        Returns:
            True if model found, False otherwise
        """
        try:
            model = self.manager.get_model(category, model_name)
        except KeyError:
            print(f"Error: Model '{model_name}' not found in category '{category}'")
            return False

        print(f"\n=== Model Information ===")
        print(f"Name: {model.name}")
        print(f"Description: {model.description or 'N/A'}")

        if hasattr(model, "baseline"):
            print(f"Baseline: {model.baseline}")

        if hasattr(model, "nsfw"):
            print(f"NSFW: {model.nsfw}")

        if model.config.download:
            print(f"\nDownload files ({len(model.config.download)}):")
            for download in model.config.download:
                print(f"  - {download.file_name}")
                print(f"    URL: {download.file_url[:60]}...")
                print(f"    SHA256: {download.sha256sum}")
                if download.known_slow_download:
                    print(f"    ⚠ Known slow download")

        return True


def main():
    """Run the model downloader examples."""
    downloader = ModelDownloader(download_dir=Path("./models"))

    # List available image models
    print("=== Image Generation Models (first 10) ===")
    models = downloader.manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
    for i, name in enumerate(sorted(models.keys())[:10]):
        print(f"{i+1}. {name}")

    # Show detailed info for a specific model
    print("\n" + "="*70)
    downloader.get_model_info(
        MODEL_REFERENCE_CATEGORY.image_generation,
        "stable_diffusion_xl"
    )


if __name__ == "__main__":
    main()
