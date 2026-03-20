# Configuration & Troubleshooting

This tutorial covers the environment variables consumers care about, how data flows through the system, and solutions for common issues.

## Consumer-Relevant Environment Variables

All settings use the `HORDE_MODEL_REFERENCE_` prefix. Most consumers only need a few:

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_TTL_SECONDS` | `60` | How long (seconds) cached data stays valid before re-checking the backend |
| `PRIMARY_API_URL` | `https://stablehorde.net/api/model_references/` | URL of the PRIMARY server to fetch from. Set to empty to use GitHub only |
| `ENABLE_GITHUB_FALLBACK` | `True` | Whether to fall back to GitHub if the PRIMARY API is unreachable |
| `PRIMARY_API_TIMEOUT` | `10` | Timeout (seconds) for PRIMARY API requests |

Set them via environment variables or a `.env` file:

```bash
export HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS=120
export HORDE_MODEL_REFERENCE_PRIMARY_API_URL="https://aihorde.net/api/model_references/"
```

## Data Flow

```
Your Code
    |
    v
ModelReferenceManager (in-memory cache, TTL-based)
    |
    v
Backend (selected automatically)
    |
    +---> HTTPBackend (REPLICA mode, default)
    |         |
    |         +---> PRIMARY API (aihorde.net)
    |         |         |
    |         |         +---> (on failure) GitHub fallback
    |         |
    |         v
    |     Raw JSON
    |
    +---> GitHubBackend (REPLICA mode, no PRIMARY API URL)
    |         |
    |         v
    |     Download + legacy conversion
    |
    +---> FileSystemBackend (PRIMARY mode)
              |
              v
          Local JSON files
```

In the typical consumer scenario (REPLICA mode), the manager fetches from the PRIMARY API first, falls back to GitHub if needed, and caches the results in memory.

## Common Issues

### RuntimeError: Singleton Conflict

**Symptom:**
```
RuntimeError: ModelReferenceManager is a singleton and has already been instantiated
with different settings.
```

**Cause:** Two parts of your code create `ModelReferenceManager()` with conflicting parameters.

**Fix:** Initialize the manager once at startup and reuse it. If you need the instance elsewhere, call `ModelReferenceManager.get_instance()` or `ModelReferenceManager()` with no arguments (which returns the existing instance).

### Empty Results or None

**Symptom:** `get_model_reference_unsafe()` returns `None`, or `get_model_reference()` raises `RuntimeError`.

**Cause:** The backend couldn't fetch data for that category (network issue, invalid category, etc.).

**Understanding safe vs. unsafe methods:**

| Method | Returns | On failure |
|--------|---------|------------|
| `get_model_reference(cat)` | `dict[str, GenericModelRecord]` | Raises `RuntimeError` |
| `get_model_reference_unsafe(cat)` | `dict[str, GenericModelRecord] \| None` | Returns `None` |
| `get_model(cat, name)` | `GenericModelRecord` | Raises `RuntimeError` |
| `get_model_unsafe(cat, name)` | `GenericModelRecord \| None` | Returns `None` |

Use the non-`unsafe` variants when you need guaranteed data. Use `unsafe` variants when you want to handle missing data gracefully.

### Stale Data

**Symptom:** Model data doesn't reflect recent changes.

**Fix:** The cache respects TTL (`CACHE_TTL_SECONDS`, default 60s). To force a refresh:

```python
# Force re-fetch from backend
models = manager.get_model_reference("image_generation", overwrite_existing=True)
```

### GitHub Rate Limits

**Symptom:** `HTTPError 403` or `rate limit exceeded` in logs when using GitHub fallback.

**Fix:** Ensure you have `PRIMARY_API_URL` configured (the default points to `aihorde.net`). GitHub is only used as a fallback; the PRIMARY API does not have rate limits.

### Network Timeouts

**Symptom:** Slow startup or timeout errors.

**Fix:** Adjust the timeout or use `LAZY` prefetch:

```bash
export HORDE_MODEL_REFERENCE_PRIMARY_API_TIMEOUT=30
```

```python
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)
```

## Debug Logging

The library uses [loguru](https://github.com/Delgan/loguru) for logging. Enable debug output:

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Now all horde_model_reference log messages will appear
manager = ModelReferenceManager()
```

This will show backend selection, cache hits/misses, fetch attempts, and conversion details.

## Async Usage

The manager provides async variants of all read methods:

```python
import asyncio
from horde_model_reference import ModelReferenceManager

async def main():
    manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.LAZY)

    # Async fetch
    models = await manager.get_model_reference_async("image_generation")
    print(f"Found {len(models)} models")

    # Async warm-up
    await manager.warm_cache_async()

asyncio.run(main())
```

**Do not mix sync and async calls in the same context.** The backends use different HTTP clients internally (`httpx.Client` vs `httpx.AsyncClient`). Mixing them can lead to event loop conflicts.

For FastAPI services, use the `ASYNC` prefetch strategy:

```python
manager = ModelReferenceManager(prefetch_strategy=PrefetchStrategy.ASYNC)
```

## Previous Tutorials

- [Getting Started](getting_started.md) -- Installation, first query, singleton pattern
- [Querying Models](querying_models.md) -- Fluent query API
- [Working with Records](working_with_records.md) -- Record types and fields
