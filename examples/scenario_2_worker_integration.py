"""
Scenario 2: AI-Horde Worker Integration

This example demonstrates how to integrate model references into
an AI-Horde worker to manage available models based on capabilities.
"""

from pathlib import Path
from typing import Set, Dict
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
)
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


class WorkerModelManager:
    """
    Manages model references for an AI-Horde worker.

    Handles filtering models by GPU capabilities, baseline support,
    and version requirements.
    """

    def __init__(
        self,
        supported_baselines: Set[str],
        min_bridge_version: int,
        models_path: Path,
    ):
        """
        Initialize the worker model manager.

        Args:
            supported_baselines: Set of baseline models the worker supports
                                (e.g., {"stable_diffusion_xl", "stable_diffusion_2"})
            min_bridge_version: The worker's bridge version number
            models_path: Path where model files are stored
        """
        self.supported_baselines = supported_baselines
        self.min_bridge_version = min_bridge_version
        self.models_path = models_path
        self.manager = ModelReferenceManager()

        # Cache of available models
        self._available_models: Dict[str, ImageGenerationModelRecord] | None = None

    def get_available_models(self) -> Dict[str, ImageGenerationModelRecord]:
        """
        Get all models the worker can serve based on capabilities.

        Returns:
            Dictionary mapping model names to their records
        """
        if self._available_models is not None:
            return self._available_models

        # Fetch all image generation models
        all_models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        # Filter by supported baselines and version
        available = {}
        for name, model in all_models.items():
            # Check baseline support
            if model.baseline not in self.supported_baselines:
                continue

            # Check bridge version requirement
            if model.min_bridge_version and model.min_bridge_version > self.min_bridge_version:
                continue

            # Check if model files exist locally
            if self._model_files_exist(model):
                available[name] = model

        self._available_models = available
        return available

    def _model_files_exist(self, model: ImageGenerationModelRecord) -> bool:
        """
        Check if all required model files exist locally.

        Args:
            model: The model record to check

        Returns:
            True if all files exist, False otherwise
        """
        for download in model.config.download:
            file_path = self.models_path / download.file_name
            if not file_path.exists():
                return False
        return True

    def get_models_to_download(self) -> Dict[str, ImageGenerationModelRecord]:
        """
        Get models that match capabilities but aren't downloaded yet.

        Returns:
            Dictionary of models that need to be downloaded
        """
        all_models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        to_download = {}
        for name, model in all_models.items():
            # Check baseline support
            if model.baseline not in self.supported_baselines:
                continue

            # Check bridge version
            if model.min_bridge_version and model.min_bridge_version > self.min_bridge_version:
                continue

            # Check if NOT downloaded
            if not self._model_files_exist(model):
                to_download[name] = model

        return to_download

    def get_model_download_info(self, model_name: str) -> list[dict] | None:
        """
        Get download URLs and checksums for a specific model.

        Args:
            model_name: Name of the model to get download info for

        Returns:
            List of download information dicts, or None if model not found
        """
        try:
            model = self.manager.get_model(
                MODEL_REFERENCE_CATEGORY.image_generation,
                model_name
            )
        except KeyError:
            return None

        download_info = []
        for download in model.config.download:
            download_info.append({
                "file_name": download.file_name,
                "url": download.file_url,
                "sha256": download.sha256sum,
                "slow_download": download.known_slow_download or False,
            })

        return download_info


def main():
    """Run the worker integration examples."""
    # Simulate a worker with SDXL support
    worker = WorkerModelManager(
        supported_baselines={"stable_diffusion_xl"},
        min_bridge_version=10,
        models_path=Path("/tmp/models_test"),
    )

    print("=== Worker Model Status ===")

    # Get available models (will be 0 since files don't exist)
    available = worker.get_available_models()
    print(f"\nModels available to serve: {len(available)}")

    # Get models to download
    to_download = worker.get_models_to_download()
    print(f"\nModels that need downloading: {len(to_download)}")
    for name, model in list(to_download.items())[:3]:
        print(f"  ⬇ {name}")

        # Get download info
        dl_info = worker.get_model_download_info(name)
        if dl_info:
            for file_info in dl_info[:2]:  # Show first 2 files
                print(f"      - {file_info['file_name']}")
                print(f"        SHA256: {file_info['sha256'][:16]}...")


if __name__ == "__main__":
    main()
