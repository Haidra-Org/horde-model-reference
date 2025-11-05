# Onboarding Guide

Welcome to the Horde Model Reference onboarding guide! This tutorial will walk you through realistic scenarios to help you integrate the library into your projects.

## Table of Contents

1. [Installation](#installation)
2. [Scenario 1: Building a Model Browser UI](#scenario-1-building-a-model-browser-ui)
3. [Scenario 2: AI-Horde Worker Integration](#scenario-2-ai-horde-worker-integration)
4. [Scenario 3: Model Downloader Tool](#scenario-3-model-downloader-tool)
5. [Scenario 4: Model Capability Checker](#scenario-4-model-capability-checker)
6. [Scenario 5: Type-Safe Configuration](#scenario-5-type-safe-configuration)
7. [Scenario 6: Working with Multiple Categories](#scenario-6-working-with-multiple-categories)
8. [Controlling Logging Verbosity](#controlling-logging-verbosity)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Basic Installation

For most users who just need to access model references:

```bash
pip install horde-model-reference
```

### With Type Checking Support

If you're using mypy or other type checkers:

```bash
pip install horde-model-reference
# Type stubs are included automatically
```

### For Service Deployment

If you plan to run your own PRIMARY server:

```bash
pip install horde-model-reference[service]
```

### Development Installation

For contributors or those who want the latest features:

```bash
git clone https://github.com/Haidra-Org/horde-model-reference.git
cd horde-model-reference
pip install -e .
```

---

## Scenario 1: Building a Model Browser UI

**Goal**: Build a web UI that displays available AI models with their metadata.

### Requirements
- Display all image generation models
- Show model descriptions, baselines, and NSFW status
- Filter by model style (anime, realistic, etc.)
- Group by baseline

### Implementation

```python
from typing import Dict, List
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
)
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


def get_model_browser_data() -> Dict[str, List[dict]]:
    """
    Fetch and organize image generation models for a browser UI.

    Returns a dictionary grouping models by baseline, with each model
    containing display-friendly information.
    """
    # Initialize manager
    manager = ModelReferenceManager()

    # Fetch all image generation models
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Group by baseline
    grouped: Dict[str, List[dict]] = {}

    for model_name, model in models.items():
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
) -> List[ImageGenerationModelRecord]:
    """
    Get all image generation models matching a specific style.

    Args:
        style: The MODEL_STYLE to filter by (anime, realistic, etc.)

    Returns:
        List of models matching the specified style
    """
    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Filter models by style - type-safe comparison
    filtered = [
        model for model in models.values()
        if model.style == style
    ]

    return filtered


def get_safe_models_only() -> Dict[str, ImageGenerationModelRecord]:
    """
    Get only SFW (Safe For Work) image generation models.

    Returns:
        Dictionary of model names to ImageGenerationModelRecord objects
        for models where nsfw=False
    """
    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    # Filter out NSFW models
    safe_models = {
        name: model for name, model in models.items()
        if not model.nsfw
    }

    return safe_models


# Example usage
if __name__ == "__main__":
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
```

### Key Takeaways

- **Type Safety**: The `ImageGenerationModelRecord` type provides IDE autocomplete and type checking
- **Enum Usage**: `MODEL_STYLE` and `MODEL_REFERENCE_CATEGORY` prevent typos
- **Automatic Updates**: Manager fetches latest data from PRIMARY server or GitHub
- **Flexible Filtering**: Python's list/dict comprehensions work seamlessly with the type-safe models

---

## Scenario 2: AI-Horde Worker Integration

**Goal**: Integrate model references into an AI-Horde worker to advertise available models.

### Requirements
- Load available models based on GPU capabilities
- Check minimum bridge version requirements
- Filter by supported baselines
- Cache model information

### Implementation

```python
import sys
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


# Example usage
if __name__ == "__main__":
    # Simulate a worker with SDXL support
    worker = WorkerModelManager(
        supported_baselines={"stable_diffusion_xl"},
        min_bridge_version=10,
        models_path=Path("/path/to/models"),
    )

    print("=== Worker Model Status ===")

    # Get available models
    available = worker.get_available_models()
    print(f"\nModels available to serve: {len(available)}")
    for name in list(available.keys())[:5]:
        print(f"  ✓ {name}")

    # Get models to download
    to_download = worker.get_models_to_download()
    print(f"\nModels that need downloading: {len(to_download)}")
    for name, model in list(to_download.items())[:3]:
        print(f"  ⬇ {name}")

        # Get download info
        dl_info = worker.get_model_download_info(name)
        if dl_info:
            for file_info in dl_info:
                size_mb = "unknown"  # In real scenario, could fetch from headers
                print(f"      - {file_info['file_name']} ({size_mb})")
                print(f"        SHA256: {file_info['sha256'][:16]}...")
```

### Key Takeaways

- **Capability Filtering**: Filter models based on GPU/bridge capabilities
- **File Validation**: Check local model files against references
- **Download Management**: Get URLs and checksums for missing models
- **Caching**: Cache model lists to avoid repeated fetches
- **Type Safety**: Strong typing prevents configuration errors

---

## Scenario 3: Model Downloader Tool

**Goal**: Create a CLI tool to download and verify AI models.

### Requirements
- Download models by name
- Verify checksums after download
- Support resumable downloads
- Show progress

### Implementation

```python
import hashlib
import sys
from pathlib import Path
from typing import Optional

import httpx
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
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ModelReferenceManager()

    def download_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        verify_checksum: bool = True,
    ) -> bool:
        """
        Download a model and optionally verify its checksum.

        Args:
            category: The model category (e.g., image_generation)
            model_name: Name of the model to download
            verify_checksum: Whether to verify SHA256 after download

        Returns:
            True if download succeeded, False otherwise
        """
        try:
            # Get model reference
            model = self.manager.get_model(category, model_name)
        except KeyError:
            print(f"Error: Model '{model_name}' not found in category '{category}'")
            return False

        if not model.config.download:
            print(f"Error: No download files configured for '{model_name}'")
            return False

        print(f"Downloading model: {model.name}")
        if model.description:
            print(f"Description: {model.description}")

        # Download each file
        success = True
        for download in model.config.download:
            file_path = self.download_dir / download.file_name

            # Check if already exists
            if file_path.exists():
                print(f"  File exists: {download.file_name}")
                if verify_checksum:
                    if self._verify_checksum(file_path, download.sha256sum):
                        print(f"    ✓ Checksum valid")
                        continue
                    else:
                        print(f"    ✗ Checksum mismatch, re-downloading")

            # Download file
            print(f"  Downloading: {download.file_name}")
            if download.known_slow_download:
                print(f"    ⚠ Warning: This download is known to be slow")

            try:
                self._download_file(download.file_url, file_path)
            except Exception as e:
                print(f"    ✗ Download failed: {e}")
                success = False
                continue

            # Verify checksum
            if verify_checksum:
                if self._verify_checksum(file_path, download.sha256sum):
                    print(f"    ✓ Checksum verified")
                else:
                    print(f"    ✗ Checksum mismatch!")
                    success = False

        return success

    def _download_file(self, url: str, dest_path: Path, chunk_size: int = 8192) -> None:
        """
        Download a file with progress indication.

        Args:
            url: URL to download from
            dest_path: Destination file path
            chunk_size: Size of chunks to download
        """
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"    Progress: {percent:.1f}%", end="\r")

            if total_size > 0:
                print()  # New line after progress

    def _verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify a file's SHA256 checksum.

        Args:
            file_path: Path to file to verify
            expected_sha256: Expected SHA256 hash

        Returns:
            True if checksum matches, False otherwise
        """
        if expected_sha256 == "FIXME":
            # Some models might not have checksums set yet
            print(f"    ⚠ No checksum available for verification")
            return True

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest().lower() == expected_sha256.lower()

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


# Example usage
if __name__ == "__main__":
    downloader = ModelDownloader(download_dir=Path("./models"))

    # List available models
    print("=== Available Image Generation Models ===")
    downloader.list_available_models(MODEL_REFERENCE_CATEGORY.image_generation)

    # Example: Download a specific model
    # Uncomment to actually download
    # success = downloader.download_model(
    #     MODEL_REFERENCE_CATEGORY.image_generation,
    #     "stable_diffusion_xl",
    #     verify_checksum=True
    # )
    #
    # if success:
    #     print("\n✓ Download completed successfully")
    # else:
    #     print("\n✗ Download failed")
    #     sys.exit(1)
```

### Key Takeaways

- **Checksum Verification**: Use SHA256 hashes from model references
- **Error Handling**: Gracefully handle missing models and download failures
- **Progress Indication**: Provide feedback during long downloads
- **Type Safety**: Enum parameters prevent category name typos

---

## Scenario 4: Model Capability Checker

**Goal**: Check if models support specific features or requirements.

### Implementation

```python
from typing import List, Dict, Any
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
)
from horde_model_reference.model_reference_records import ImageGenerationModelRecord


class ModelCapabilityChecker:
    """
    Check model capabilities and requirements.
    """

    def __init__(self):
        self.manager = ModelReferenceManager()

    def check_inpainting_support(self) -> List[str]:
        """
        Get all models that support inpainting.

        Returns:
            List of model names that support inpainting
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        inpainting_models = [
            name for name, model in models.items()
            if model.inpainting
        ]

        return inpainting_models

    def check_model_requirements(
        self,
        model_name: str
    ) -> Dict[str, Any] | None:
        """
        Get the requirements for a specific model.

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
    ) -> List[ImageGenerationModelRecord]:
        """
        Find all models based on a specific baseline.

        Args:
            baseline: The baseline to search for (e.g., "stable_diffusion_xl")

        Returns:
            List of models with the specified baseline
        """
        models = self.manager.get_model_reference(
            MODEL_REFERENCE_CATEGORY.image_generation
        )

        matching_models = [
            model for model in models.values()
            if model.baseline == baseline
        ]

        return matching_models

    def get_models_with_tags(
        self,
        required_tags: List[str]
    ) -> List[ImageGenerationModelRecord]:
        """
        Find models that have all specified tags.

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


# Example usage
if __name__ == "__main__":
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

    # Find anime models
    print("\n=== Anime Models ===")
    anime_models = checker.get_models_with_tags(["anime"])
    print(f"Found {len(anime_models)} anime models")
    for model in anime_models[:5]:
        print(f"  - {model.name}: {model.description}")
```

---

## Scenario 5: Type-Safe Configuration

**Goal**: Use Pydantic models for type-safe application configuration.

### Implementation

```python
from typing import List
from pydantic import BaseModel, Field, field_validator
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
)


class WorkerConfig(BaseModel):
    """
    Type-safe configuration for an AI-Horde worker.

    Uses Pydantic for validation and type safety.
    """

    worker_name: str = Field(
        ...,
        min_length=1,
        description="Name of the worker"
    )

    supported_models: List[str] = Field(
        default_factory=list,
        description="List of model names this worker supports"
    )

    max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of concurrent generation jobs"
    )

    supported_baselines: List[str] = Field(
        default_factory=lambda: ["stable_diffusion_xl"],
        description="Baselines supported by this worker"
    )

    filter_nsfw: bool = Field(
        default=True,
        description="Whether to filter out NSFW models"
    )

    preferred_styles: List[MODEL_STYLE] = Field(
        default_factory=list,
        description="Preferred model styles to prioritize"
    )

    @field_validator("supported_models")
    @classmethod
    def validate_models_exist(cls, v: List[str]) -> List[str]:
        """
        Validate that all specified models exist in the reference.
        """
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

    def get_available_models(self) -> List[str]:
        """
        Get the list of models this worker can serve based on configuration.

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


# Example usage
if __name__ == "__main__":
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
    try:
        invalid_config = WorkerConfig(
            worker_name="test",
            supported_models=["non_existent_model"],
        )
    except ValueError as e:
        print(f"\n✓ Validation caught error: {e}")
```

### Key Takeaways

- **Pydantic Integration**: Seamless integration with Pydantic for validation
- **Field Validation**: Validate model names against the reference
- **Type Safety**: Strong typing throughout the configuration
- **Enum Support**: Use enums for style and category fields

---

## Scenario 6: Working with Multiple Categories

**Goal**: Work with different model categories (text, image, ControlNet, etc.)

### Implementation

```python
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

        for category in MODEL_REFERENCE_CATEGORY:
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


# Example usage
if __name__ == "__main__":
    explorer = MultiCategoryModelExplorer()

    # Get summary of all categories
    print("=== Model Category Summary ===")
    summary = explorer.get_category_summary()
    for category, count in sorted(summary.items()):
        print(f"{category:<25} {count:>5} models")

    # Explore text generation models
    print("\n=== Text Generation Models ===")
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
```

---

## Controlling Logging Verbosity

### The Problem

By default, the library logs only WARNING and above to keep your console clean. However, when troubleshooting, you might want to see what's happening under the hood.

### Solution: Logging Control Functions

The library provides several ways to control logging:

```python
from horde_model_reference import (
    ModelReferenceManager,
    enable_debug_logging,
    configure_logger,
    disable_logging,
)

# Option 1: Enable debug logging (most verbose)
enable_debug_logging()

# Option 2: Set specific log level
configure_logger("INFO")  # INFO, DEBUG, WARNING, ERROR, CRITICAL

# Option 3: Completely silence the library
disable_logging()
```

### Using Environment Variables

You can also control logging without code changes:

```bash
# Enable debug logging
export HORDE_MODEL_REFERENCE_LOG_LEVEL=DEBUG
python your_script.py

# Or for a single run
HORDE_MODEL_REFERENCE_LOG_LEVEL=INFO python your_script.py
```

### Recommended Patterns

**Production Code:**
```python
# Keep default (WARNING only) or disable completely
from horde_model_reference import ModelReferenceManager, disable_logging

disable_logging()  # Completely quiet
manager = ModelReferenceManager()
```

**Development:**
```python
# Use INFO level for visibility without too much detail
from horde_model_reference import ModelReferenceManager, configure_logger

configure_logger("INFO")
manager = ModelReferenceManager()
```

**Debugging:**
```python
# Use DEBUG to see everything
from horde_model_reference import ModelReferenceManager, enable_debug_logging

enable_debug_logging()
manager = ModelReferenceManager()
```

### Complete Example

See `examples/scenario_logging_control.py` for a complete demonstration of all logging control options.

---

## Best Practices

### 1. Use Enums for Type Safety

**DO:**
```python
from horde_model_reference import MODEL_REFERENCE_CATEGORY, MODEL_STYLE

# Type-safe, IDE autocomplete works
models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
anime_filter = MODEL_STYLE.anime
```

**DON'T:**
```python
# Prone to typos, no IDE autocomplete
models = manager.get_model_reference("image_generation")
anime_filter = "anime"
```

### 2. Cache the Manager Instance

**DO:**
```python
class MyApp:
    def __init__(self):
        # Create once, reuse
        self.manager = ModelReferenceManager()

    def get_models(self):
        return self.manager.get_model_reference(...)
```

**DON'T:**
```python
def get_models():
    # Creates new instance every call
    manager = ModelReferenceManager()
    return manager.get_model_reference(...)
```

### 3. Handle Missing Models Gracefully

**DO:**
```python
try:
    model = manager.get_model(category, model_name)
except KeyError:
    print(f"Model {model_name} not found")
    # Fall back to alternative
```

**DON'T:**
```python
# Assumes model exists - will crash if not found
model = manager.get_model(category, model_name)
```

### 4. Use Type Annotations

**DO:**
```python
from horde_model_reference.model_reference_records import ImageGenerationModelRecord

def process_model(model: ImageGenerationModelRecord) -> dict:
    # Type checker knows model structure
    return {"baseline": model.baseline, "nsfw": model.nsfw}
```

**DON'T:**
```python
def process_model(model):  # No type hints
    return {"baseline": model.baseline, "nsfw": model.nsfw}
```

### 5. Check Model Capabilities Before Use

**DO:**
```python
if model.inpainting:
    # Use inpainting features
    enable_inpainting_mode()

if model.min_bridge_version and model.min_bridge_version > my_version:
    print(f"Model requires bridge version {model.min_bridge_version}")
```

---

## Troubleshooting

### Issue: Models Not Loading

**Symptoms:** Empty model dictionaries or None values

**Solutions:**
1. Check network connectivity to PRIMARY server
2. Verify GitHub is accessible (fallback source)
3. Check cache directory permissions
4. Try clearing cache: `rm -rf ~/.cache/horde_model_reference/`

### Issue: Type Checking Errors

**Symptoms:** mypy or IDE complaining about types

**Solutions:**
1. Ensure you're using proper enum types:
   ```python
   MODEL_REFERENCE_CATEGORY.image_generation  # Not "image_generation"
   ```

2. Import the specific record types:
   ```python
   from horde_model_reference.model_reference_records import ImageGenerationModelRecord
   ```

### Issue: Model Not Found

**Symptoms:** `KeyError` when accessing specific model

**Solutions:**
1. Check if model name is correct (case-sensitive)
2. Verify model is in the expected category
3. Check if model was recently added/removed
4. Force refresh: create new `ModelReferenceManager()` instance

### Issue: Download Failures

**Symptoms:** Cannot download model files

**Solutions:**
1. Check download URLs are accessible
2. Verify sufficient disk space
3. Check firewall/proxy settings
4. Some models have `known_slow_download=True` - be patient

### Issue: Performance Problems

**Symptoms:** Slow model reference fetching

**Solutions:**
1. Reuse `ModelReferenceManager` instance (don't recreate)
2. Cache model lists in your application
3. Consider running your own PRIMARY server for better performance
4. Check TTL settings in `HordeModelReferenceSettings`

---

## Next Steps

Now that you understand the basics:

1. **Explore the [API Reference](horde_model_reference/)** for detailed documentation
2. **Check out the [Backend Architecture](model_reference_backend.md)** to understand data sources
3. **Read the [Deployment Guide](../DEPLOYMENT.md)** if running your own server
4. **Join the [Discord](https://discord.gg/3DxrhksKzn)** for community support

Happy coding! 🚀
