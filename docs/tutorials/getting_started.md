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

On first run the manager fetches model reference data from the PRIMARY server (`aihorde.net`) with a GitHub fallback. Results are cached in memory with a configurable TTL (default 60 seconds).

## The Singleton Pattern

`ModelReferenceManager` is a **singleton**. The first instantiation locks in its configuration (backend, base path, prefetch strategy). Any subsequent call to `ModelReferenceManager()` returns the same instance. Attempting to create a second instance with _different_ parameters raises `RuntimeError`.

**Correct pattern** -- initialize once, reuse everywhere:

```python
# At startup
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)

# Elsewhere in your code
manager = ModelReferenceManager()  # returns the same instance
```

**What to avoid:**

```python
# This will raise RuntimeError because prefetch_strategy differs
manager_a = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
manager_b = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.SYNC)  # RuntimeError!
```

If you need the singleton instance without risking a conflicting re-init, use:

```python
manager = ModelReferenceManager.get_instance()  # raises RuntimeError if not yet created
```

## Prefetch Strategies

The `prefetch_strategy` parameter controls when model data is fetched from the backend:

| Strategy         | Behavior                                                                                 | Best For                                                        |
| ---------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `LAZY` (default) | Defers fetching until you first access data                                              | Scripts, CLIs, most consumers                                   |
| `SYNC`           | Fetches all categories immediately during init                                           | Latency-sensitive services that are OK with blocking on startup |
| `ASYNC`          | Schedules a background async fetch if an event loop is running                           | FastAPI / async services                                        |
| `DEFERRED`       | Creates a handle you trigger later via `handle.run_sync()` or `await handle.run_async()` | Fine-grained startup control                                    |
| `NONE`           | No automatic fetching at all; you call cache helpers manually                            | Testing, custom orchestration                                   |

```python
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

# For a FastAPI app
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.ASYNC)

# For a CLI script
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
```

## Model Categories

Every model in the reference belongs to a category. List them all with:

```python
from horde_model_reference import MODEL_REFERENCE_CATEGORY

for cat in MODEL_REFERENCE_CATEGORY:
    print(cat.value)
```

Categories include: `image_generation`, `text_generation`, `video_generation`, `audio_generation`, `clip`, `controlnet`, `blip`, `esrgan`, `gfpgan`, `codeformer`, `safety_checker`, `miscellaneous`.

You can use either the enum member or a plain string:

```python
# These are equivalent
models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
models = manager.get_model_reference("image_generation")
```

## What Happens Under the Hood

When you call `manager.get_model_reference(category)`:

1. The manager checks its in-memory cache
2. If stale or missing, it delegates to the **backend**
3. In REPLICA mode (the default), the backend fetches JSON from the PRIMARY API (`aihorde.net`), falling back to GitHub if the PRIMARY is unavailable
4. The raw JSON is validated and converted into typed Pydantic model records
5. Results are cached with a TTL (default 60 seconds)

## Next Steps

- [Querying Models](querying_models.md) -- Learn the fluent query API for filtering, sorting, and aggregating models
- [Working with Records](working_with_records.md) -- Understand the model record types and their fields
- [Configuration & Troubleshooting](configuration_and_troubleshooting.md) -- Environment variables, debugging, and common issues
