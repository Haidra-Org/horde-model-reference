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

By default the library runs in **REPLICA mode** - a read-only consumer. It fetches from the **PRIMARY server** (the authoritative service hosting the data, `models.aihorde.net`) and, if that is unreachable, automatically uses the **GitHub fallback** (the original model-reference repositories). You don't need to configure any of this to get started; the defaults work out of the box. If these terms are new, see the [Glossary](../reference/glossary.md).

## The Singleton Pattern

`ModelReferenceManager` is a **singleton**. The first instantiation locks in its configuration (backend, base path, prefetch strategy). Any subsequent call to `ModelReferenceManager()` returns the same instance. Attempting to create a second instance with _different_ parameters raises `RuntimeError`.

**Correct pattern** -- initialize once, reuse everywhere:

```python
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

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

| Strategy         | Behavior                                                                                   | Best For                                                        |
| ---------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| `LAZY` (default) | Defers fetching until you first access data                                                | Scripts, CLIs, most consumers                                   |
| `SYNC`           | Fetches all categories immediately during init                                             | Latency-sensitive services that are OK with blocking on startup |
| `ASYNC`          | Schedules a background async fetch **if an event loop is already running** at construction | Code that constructs the manager *inside* a running loop        |
| `DEFERRED`       | Creates a handle you trigger later via `handle.run_sync()` or `await handle.run_async()`   | Fine-grained startup control                                    |
| `NONE`           | No automatic fetching at all; you call cache helpers manually                              | Testing, custom orchestration                                   |

```python
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

# CLI script or worker: lazy is fine -- the first read warms the cache
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
```

### Warming the cache in a FastAPI (or other async) service

!!! warning "`ASYNC` needs a running event loop *at construction time*"
    `PrefetchStrategy.ASYNC` only schedules the background fetch if an event loop is already running
    when the manager is constructed. If you construct the manager at module import -- the usual case,
    before the server's loop exists -- it logs a warning and degrades to a manual deferred handle that
    is **not triggered automatically**. Your first requests then pay the full fetch latency, which is
    exactly what `ASYNC` was supposed to avoid. The degrade is now discoverable rather than silent:
    `manager.prefetch_pending` is `True` while a warm-up is still owed, and the exposed
    `manager.deferred_prefetch_handle` can be run manually (see below).

The robust pattern is to construct the manager lazily and warm the cache from the application's
**lifespan / startup hook**, where a loop is guaranteed to be running, and `await` it:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from horde_model_reference import ModelReferenceManager, PrefetchStrategy

# Lazy construction at import is safe; the warm-up happens once the loop is up.
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await manager.ensure_ready_async()  # fetch + cache now, while the loop runs
    yield


app = FastAPI(lifespan=lifespan)
```

`ensure_ready_async()` loads every category into the cache so subsequent request handlers read from
memory. Reach for `PrefetchStrategy.ASYNC` only when you construct the manager *inside* a task that
is already running on the loop.

### Checking and forcing readiness

Two manager members let you assert readiness instead of trusting a log line:

- `manager.is_warm` -- `True` once every category is loaded into the in-memory cache, so reads are
  served without a backend fetch. Use it in a health check or a test.
- `manager.prefetch_pending` -- `True` when a deferred warm-up is still owed (e.g. an `ASYNC`
  prefetch that degraded because no loop was running). It clears once the cache is warm.

From synchronous code, `manager.ensure_ready()` is the sync mirror of `ensure_ready_async()`; it also
completes a degraded `ASYNC` warm-up. The exposed handle does the same:

```python
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.ASYNC)  # at import: no loop yet

if manager.prefetch_pending:           # the ASYNC schedule degraded to a manual handle
    manager.ensure_ready()             # ...or: manager.deferred_prefetch_handle.run_sync()

assert manager.is_warm
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

Prefer to read working code? The [`examples/`](https://github.com/Haidra-Org/horde-model-reference/tree/main/examples)
directory has small, runnable scripts for each of these topics.
