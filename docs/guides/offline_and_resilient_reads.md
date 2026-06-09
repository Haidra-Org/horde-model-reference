# Read resiliently (caching, fallback, and being offline)

**Goal:** understand what happens when the network is slow or down, and write
read code that degrades gracefully instead of crashing.

## What a read actually does

By default the library runs in **REPLICA mode** - a read-only consumer (the
opposite is **PRIMARY**, the authoritative server that owns the data). The first
access for a category fetches JSON from the **PRIMARY server** (the service
hosting the data, `models.aihorde.net`), and if that is unreachable it **falls
back to GitHub** - the original model-reference repositories, which the library
reads and converts on the fly. The result is validated into Pydantic records and
cached in memory with a TTL (default 60 s). Subsequent reads inside the TTL are
served from cache with no network call.

So a single cold read can succeed even if one source is down, and warm reads
don't touch the network at all. (New to these terms? See the
[Glossary](../reference/glossary.md).)

See [Architecture Overview](../concepts/architecture_overview.md) for the full
backend/caching picture.

## Don't crash on missing data

Every read has two variants:

- `get_model_reference(category)` / `get_model(category, name)` - **raise** if the
  data can't be produced.
- `get_model_reference_or_none(...)` / `get_model_or_none(...)` - return `None`
  instead.

For resilient code, prefer the `_or_none` variants and handle the empty case:

```python
from horde_model_reference import ModelReferenceManager

manager = ModelReferenceManager()

models = manager.get_model_reference_or_none("image_generation")
if not models:
    # Network down on a cold cache, or category genuinely empty.
    print("Model reference unavailable right now; using last-known/default set.")
    models = {}
```

## Force a refresh (and when not to)

Reads serve from cache until the TTL expires. To bypass the cache and re-fetch
immediately, pass `overwrite_existing=True`:

```python
fresh = manager.get_model_reference("image_generation", overwrite_existing=True)
```

Use this sparingly - on a long-running service, prefer letting the TTL handle
freshness so you don't add latency or load on every call.

## Tune freshness vs. resilience

Two environment variables matter most for consumers:

| Variable | Effect |
| -------- | ------ |
| `HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS` | How long cached data is reused before a refetch (default `60`). |
| `HORDE_MODEL_REFERENCE_ENABLE_GITHUB_FALLBACK` | Whether to fall back to GitHub when PRIMARY fails (default `True`). |
| `HORDE_MODEL_REFERENCE_PRIMARY_API_URL` | Override the PRIMARY server, or set to skip and use GitHub only. |

A longer TTL means fewer network calls and better resilience to transient
outages, at the cost of staleness. See
[Configuration & Troubleshooting](../tutorials/configuration_and_troubleshooting.md)
for the complete list and debugging tips.

## Next

- [Configuration & Troubleshooting](../tutorials/configuration_and_troubleshooting.md) -- environment variables, debugging, common issues
