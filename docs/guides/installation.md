# Installation

Install `horde-model-reference` from PyPI or from source.

## From PyPI

```bash
pip install horde-model-reference
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add horde-model-reference
```

## Optional dependencies

The `[service]` extra pulls in FastAPI, uvicorn, and httpx for running the HTTP service:

```bash
pip install horde-model-reference[service]
```

## From source

```bash
git clone https://github.com/Haidra-Org/horde-model-reference.git
cd horde-model-reference
uv sync --all-groups
```

## Verify

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()
print(len(manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)))
```

On first run the manager fetches model reference data from the network and caches it in memory.
Subsequent calls serve from cache.

## Configuration

The library works out of the box with no configuration. For the full list of environment
variables (all prefixed with `HORDE_MODEL_REFERENCE_`), see the
[Configuration & Troubleshooting](../tutorials/configuration_and_troubleshooting.md) tutorial.

## Next

- [Getting Started](../tutorials/getting_started.md) -- first query, singleton pattern, prefetch strategies
- [Deployment Guide](https://github.com/Haidra-Org/horde-model-reference/blob/main/DEPLOYMENT.md) -- run your own PRIMARY server
