# Horde Model Reference

<p align="center">
  <img src="docs/haidra-assets/haidra-logo/haidra-transparent.webp" alt="Haidra Logo" width="50"/>
</p>

<p align="center">
  <a href="https://github.com/Haidra-Org/horde-model-reference/actions"><img src="https://github.com/Haidra-Org/horde-model-reference/workflows/Tests/badge.svg" alt="Build Status"></a>
  <a href="https://pypi.org/project/horde-model-reference/"><img src="https://img.shields.io/pypi/v/horde-model-reference.svg" alt="PyPI Version"></a>
<a href="https://pypi.org/project/horde-model-reference/"><img src="https://img.shields.io/pypi/pyversions/horde-model-reference.svg" alt="PyPI badge showing supported Python versions for horde-model-reference package in blue and white color scheme"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
</p>

**Horde Model Reference** is the authoritative source for AI model metadata in the [AI-Horde](https://aihorde.net) ecosystem. It provides information about image generation models (Stable Diffusion, FLUX, etc.), text generation models (LLMs), and utility models (CLIP, ControlNet, upscalers, etc.) used by AI-Horde tools, workers and clients.

## 📋 Table of Contents

- [Horde Model Reference](#horde-model-reference)
    - [📋 Table of Contents](#-table-of-contents)
    - [What is Horde Model Reference?](#what-is-horde-model-reference)
        - [Role in AI-Horde](#role-in-ai-horde)
    - [Key Features](#key-features)
    - [Quick Start](#quick-start)
        - [Use Case 1: Python Library (Most Common)](#use-case-1-python-library-most-common)
        - [Use Case 2: Direct JSON Access](#use-case-2-direct-json-access)
        - [Use Case 3: FastAPI Service](#use-case-3-fastapi-service)
    - [Installation](#installation)
        - [From PyPI (Recommended)](#from-pypi-recommended)
        - [With Optional Dependencies](#with-optional-dependencies)
        - [Using uv (Faster)](#using-uv-faster)
        - [From Source](#from-source)
    - [Usage Examples](#usage-examples)
        - [Fetching Model References](#fetching-model-references)
        - [Checking Model Availability](#checking-model-availability)
        - [Using with AI-Horde Worker](#using-with-ai-horde-worker)
        - [Accessing via REST API](#accessing-via-rest-api)
    - [Documentation](#documentation)
    - [Contributing](#contributing)
    - [Support \& Community](#support--community)
    - [License](#license)
        - [What This Means](#what-this-means)
    - [Acknowledgments](#acknowledgments)
        - [Related Projects](#related-projects)

## What is Horde Model Reference?

**Horde Model Reference** serves three purposes in the AI-Horde ecosystem:

1. **📄 JSON Reference Files**: Canonical model metadata (names, checksums, download URLs, capabilities) used by workers and clients
2. **🐍 Python Library**: Programmatic access to model references with automatic updates, validation, and conversion
3. **🌐 FastAPI Service**: REST API for fetching, managing, and distributing model references across the horde

### Role in AI-Horde

The [AI-Horde](https://aihorde.net) ([github](https://github.com/Haidra-Org/AI-Horde)) is a free and open sourced, crowd-sourced distributed compute network for AI generations. Workers need to know which models are approved, where to download them, and what capabilities they offer. This package provides that authoritative information.

For more context on AI-Horde concepts (workers, kudos, jobs, etc.), see the [AI-Horde Glossary](docs/haidra-assets/docs/definitions.md).

## Key Features

- ✅ **Validated Model Metadata**: SHA256 checksums, download URLs, baselines, NSFW flags, capabilities
- 🗃️ **Multiple Categories**: Image generation, text generation, CLIP, ControlNet, ESRGAN, GFPGAN, and more
- 🌐 **REST API**: FastAPI service with OpenAPI documentation
- 📦 **Legacy Compatibility**: Automatic conversion from legacy GitHub format to new standardized format
- 🔒 **Type-Safe**: Strict type checking with Pydantic models
- 🔍 **Fluent Query API**: Filter, sort, and aggregate models with a chainable, type-safe query builder
- 🐳 **Docker Ready**: Docker and docker-compose configurations for deployment

## Quick Start

Choose your use case:

### Use Case 1: Python Library (Most Common)

**For AI-Horde workers, client applications, or scripts that need model metadata:**

```bash
pip install horde-model-reference
```

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

# Automatically fetches from PRIMARY server or GitHub
manager = ModelReferenceManager()

# Get all image generation models (using enum - strings like "image_generation" also work)
image_models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

for model_name, model_data in image_models.items():
    print(f"{model_name}: {model_data.description}")

# Check if a specific model exists
if "stable_diffusion_xl" in image_models:
    model = image_models["stable_diffusion_xl"]
    print(f"Baseline: {model.baseline}")
    print(f"NSFW: {model.nsfw}")

# Query API: filter and sort with a fluent builder
from horde_model_reference import ImageFields, false
sfw_xl = (
    manager.query_image_generation()
    .where(ImageFields.nsfw == false, ImageFields.baseline == "stable_diffusion_xl")
    .order_by("name")
    .to_list()
)
print(f"Found {len(sfw_xl)} SFW SDXL models")
```

See [Querying Models](docs/tutorials/querying_models.md) for the full query API reference.

### Use Case 2: Direct JSON Access

**For non-Python applications or manual inspection:**

The JSON files are available directly from the PRIMARY server:

```bash
# Get all image generation models
curl https://aihorde.net/api/model_references/v2/image_generation

# Get specific model
curl https://aihorde.net/api/model_references/v2/image_generation/stable_diffusion_xl

# List all categories
curl https://aihorde.net/api/model_references/v2/model_categories
```

Or clone the legacy GitHub repositories:

- Image models: [AI-Horde-image-model-reference](https://github.com/Haidra-Org/AI-Horde-image-model-reference)
- Text models: [AI-Horde-text-model-reference](https://github.com/Haidra-Org/AI-Horde-text-model-reference)

### Use Case 3: FastAPI Service

**For deploying your own PRIMARY server (advanced):**

See the comprehensive [DEPLOYMENT.md](DEPLOYMENT.md) guide for:

- Docker single-worker deployment
- Docker multi-worker deployment with Redis
- Optional GitHub sync service (auto-sync to legacy repos)
- Non-Docker deployment
- Production recommendations

Quick start:

```bash
# Start PRIMARY server
docker-compose up -d

# Optional: Enable GitHub sync service
docker-compose --profile sync up -d

# Verify
curl http://localhost:19800/api/heartbeat
```

For GitHub sync setup, see [the sync README.md](./scripts/sync/README.md).

## Installation

### From PyPI (Recommended)

```bash
pip install horde-model-reference
```

### With Optional Dependencies

```bash
# For running the FastAPI service
pip install horde-model-reference[service]
```

### Using uv (Faster)

```bash
uv add horde-model-reference
```

### From Source

```bash
git clone https://github.com/Haidra-Org/horde-model-reference.git
cd horde-model-reference
uv sync --all-groups  # or: pip install -e .
```

## Usage Examples

### Fetching Model References

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()

# Get all available categories
print(list(MODEL_REFERENCE_CATEGORY))
# ['image_generation', 'text_generation', 'clip', 'controlnet', ...]

# Get all models in a category (using enum - strings like "text_generation" also work)
text_models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.text_generation)
print(f"Found {len(text_models)} text models")

# Get a specific model directly
model = manager.get_model(MODEL_REFERENCE_CATEGORY.text_generation, "llama-3-70b")
print(f"Parameters: {model.parameters_count}")
print(f"Description: {model.description}")

# Get just the model names in a category
model_names = manager.get_model_names(MODEL_REFERENCE_CATEGORY.image_generation)
print(f"Available image models: {', '.join(model_names[:3])}...")

# Get all references across all categories
all_refs = manager.get_all_model_references()
for category, models in all_refs.items():
    print(f"{category}: {len(models)} models")
```

### Checking Model Availability

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()

def is_model_available(category: MODEL_REFERENCE_CATEGORY, model_name: str) -> bool:
    """Check if a model is in the model reference."""
    try:
        models = manager.get_model_reference(category)
        return model_name in models
    except Exception as e:
        print(f"Error checking model: {e}")
        return False

# Usage (enum recommended, but strings like "image_generation" also work)
if is_model_available(MODEL_REFERENCE_CATEGORY.image_generation, "stable_diffusion_xl"):
    print("SDXL is available!")
```

### Using with AI-Horde Worker

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

# Worker initialization
manager = ModelReferenceManager()

# Get approved models for your worker (using enum - strings also work)
available_models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

# Filter by what your GPU can handle
worker_models = {
    name: model
    for name, model in available_models.items()
    if model.baseline in ["stable_diffusion_1", "stable_diffusion_xl"]
}

print(f"Worker can serve {len(worker_models)} models")
```

### Querying Models

The fluent query API lets you filter, sort, and aggregate models without manual dict comprehensions:

```python
from horde_model_reference import ModelReferenceManager, ImageFields, TextFields, false

manager = ModelReferenceManager()

# Find SFW SDXL models sorted by name
sfw_sdxl = (
    manager.query_image_generation()
    .exclude_nsfw()
    .for_baseline("stable_diffusion_xl")
    .order_by("name")
    .to_list()
)

# Text models with 7B+ parameters
big_llms = (
    manager.query_text_generation()
    .where(TextFields.parameters_count > 7_000_000_000)
    .to_list()
)

# Group text models by base model
groups = manager.query_text_generation().group_by_base_model()
for base, variants in groups.items():
    print(f"{base}: {len(variants)} variants")

# Cross-category search
total = manager.query_all().count()
print(f"Total models across all categories: {total}")
```

Field references (`ImageFields`, `TextFields`, etc.) provide IDE autocomplete and support comparison operators (`==`, `!=`, `<`, `>`, etc.), boolean composition (`&`, `|`, `~`), and ordering (`.asc()`, `.desc()`).

For the full query API, see the [Querying Models tutorial](docs/tutorials/querying_models.md).

### Accessing via REST API

If you're running the FastAPI service:

```python
import requests

BASE_URL = "http://localhost:19800/api/model_references/v2"

# Get all image models
response = requests.get(f"{BASE_URL}/image_generation")
models = response.json()

# Get specific model
response = requests.get(f"{BASE_URL}/image_generation/stable_diffusion_xl")
model = response.json()
print(f"Model: {model['name']}")
print(f"Description: {model['description']}")
```

## Documentation

### Getting Started

- [Getting Started](docs/tutorials/getting_started.md) -- Installation, first query, singleton pattern, prefetch strategies
- [Querying Models](docs/tutorials/querying_models.md) -- Fluent query API, filtering, sorting, aggregation
- [Working with Records](docs/tutorials/working_with_records.md) -- Record types, fields, serialization
- [Configuration & Troubleshooting](docs/tutorials/configuration_and_troubleshooting.md) -- Env vars, debugging, common issues

### Deployment & Operations

- [Deployment Guide](DEPLOYMENT.md) -- Docker and non-Docker deployment
- [GitHub Sync (Docker)](DOCKER_SYNC.md) -- Optional automated sync to legacy repos
- [Primary Deployments](docs/primary_deployments.md) -- Backend selection, Redis, multi-worker setup
- [Canonical Format](docs/canonical_format.md) -- API versioning and format settings

### Architecture Reference

- [Model Reference Backend](docs/model_reference_backend.md) -- Backend ABC and implementations
- [Model Reference Records](docs/model_reference_records.md) -- Record hierarchy and validation
- [Replica Backend Base](docs/replica_backend_base.md) -- TTL caching and staleness tracking
- [Design Decisions](docs/design_decisions.md) -- Trade-offs and known limitations

### Operations

- [Pending Queue](docs/pending_queue.md) -- Write approval workflow
- [Audit Trail](docs/audit_trail.md) -- Operation logging and replay

### Other

- [API Reference](http://localhost:19800/docs) -- Run service and visit for interactive Swagger UI
- [Legacy CSV Conversion](docs/legacy_csv_conversion.md) -- Text generation CSV format details
- [Contributing](.CONTRIBUTING.md) -- Development setup and guidelines

## Contributing

We welcome contributions of all sizes! Before contributing:

1. Read [.CONTRIBUTING.md](.CONTRIBUTING.md) for setup and guidelines
2. Check [open issues](https://github.com/Haidra-Org/horde-model-reference/issues) or [start a discussion](https://github.com/Haidra-Org/horde-model-reference/discussions)
3. Follow the [Haidra Python Style Guide](docs/haidra-assets/docs/meta/python.md)

## Support & Community

- **💬 Discord**: [AI Horde Discord](https://discord.gg/3DxrhksKzn) - `#horde-model-reference` channel
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Haidra-Org/horde-model-reference/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/Haidra-Org/horde-model-reference/discussions)
- **📧 Contact**: See [Haidra-Org](https://github.com/Haidra-Org) maintainers

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

See [LICENSE](LICENSE) for the full text.

### What This Means

- ✅ **Free to use** for any purpose (personal, commercial, research)
- ✅ **Free to modify** and distribute modifications
- ⚠️ **Must disclose source** if you run a modified version as a network service
- ⚠️ **Must use same license** for derivative works

For network service deployments, you must make your source code available to users. See [GNU AGPL FAQ](https://www.gnu.org/licenses/agpl-3.0.html) for details.

## Acknowledgments

- **[Haidra-Org](https://github.com/Haidra-Org)**: Core development team
- **[db0](https://github.com/db0)**: AI-Horde creator and lead maintainer
- **AI-Horde Community**: Workers, contributors, and supporters

### Related Projects

- **[AI-Horde](https://github.com/Haidra-Org/AI-Horde)**: Main AI-Horde API server
- **[horde-worker-reGen](https://github.com/Haidra-Org/horde-worker-reGen)**: Official image generation worker
- **[horde-sdk](https://github.com/Haidra-Org/horde-sdk)**: Python SDK for AI-Horde API
- **[hordelib](https://github.com/Haidra-Org/hordelib)**: Library wrapper around ComfyUI
