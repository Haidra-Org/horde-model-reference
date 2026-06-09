# Getting Started

This tutorial walks you through installing the library, running your first query, and understanding the core concepts you will use in every interaction.

## Installation

```bash
pip install horde-model-reference
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add horde-model-reference
```

## Your First Query

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()

image_models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
print(f"Found {len(image_models)} image generation models")

for name, model in list(image_models.items())[:5]:
    print(f"  {name}: {model.baseline}")
```

On first run the manager fetches model reference data over the network and caches it in memory with a configurable TTL (default 60 seconds). Subsequent reads inside the TTL are served from cache.

By default the library runs in **REPLICA mode** -- a read-only consumer. It fetches from the PRIMARY server (the authoritative service hosting the data at `models.aihorde.net`) and, if that is unreachable, automatically falls back to the original GitHub repositories. You do not need to configure anything to get started. If these terms are new, see the [Glossary](../reference/glossary.md).

## The Singleton Pattern

`ModelReferenceManager` is a singleton. The first instantiation locks in its configuration (backend, base path, prefetch strategy). Any subsequent call to `ModelReferenceManager()` returns the same instance. Attempting to create a second instance with different parameters raises `RuntimeError`.

Initialize once, reuse everywhere:

```python
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

# At startup
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)

# Elsewhere in your code
manager = ModelReferenceManager()  # returns the same instance
```

Avoid creating the manager multiple times with different parameters:

```python
# This will raise RuntimeError because prefetch_strategy differs
manager_a = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
manager_b = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.SYNC)  # RuntimeError!
```

If you need the singleton instance without risking a conflicting re-init, use `ModelReferenceManager.get_instance()`.

## Prefetch Strategy

The `prefetch_strategy` parameter controls when model data is fetched. For most users, the default `LAZY` strategy is the right choice -- it defers fetching until you first access data.

```python
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
```

For async services (FastAPI), construct the manager lazily and warm the cache from the application's lifespan hook:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from horde_model_reference import ModelReferenceManager

manager = ModelReferenceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await manager.ensure_ready_async()
    yield

app = FastAPI(lifespan=lifespan)
```

For the full list of strategies (`SYNC`, `ASYNC`, `DEFERRED`, `NONE`) and their trade-offs, see the [PrefetchStrategy reference](../reference/api_deployments.md#backend-architecture).

## Model Categories

Every model in the reference belongs to a category. List them all with:

```python
from horde_model_reference import MODEL_REFERENCE_CATEGORY

for cat in MODEL_REFERENCE_CATEGORY:
    print(cat.value)
```

Categories include: `image_generation`, `text_generation`, `clip`, `controlnet`, `blip`, `esrgan`, `gfpgan`, `codeformer`, `safety_checker`, and `miscellaneous`. The `video_generation` and `audio_generation` categories are reserved for future use and do not currently contain models.

You can use either the enum member or a plain string:

```python
# These are equivalent
models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
models = manager.get_model_reference("image_generation")
```

## What Happens Under the Hood

When you call `manager.get_model_reference(category)`:

1. The manager checks its in-memory cache
2. If stale or missing, it delegates to the backend
3. In REPLICA mode (the default), the backend fetches JSON from `models.aihorde.net`, falling back to GitHub if the PRIMARY is unavailable
4. The raw JSON is validated and converted into typed Pydantic model records
5. Results are cached with a TTL (default 60 seconds)

For more detail, see the [Architecture Overview](../concepts/architecture_overview.md).

## Next

- [Querying Models](querying_models.md) -- filter, sort, and aggregate with the fluent query API
- [Configuration & Troubleshooting](configuration_and_troubleshooting.md) -- environment variables, debugging, and common issues
