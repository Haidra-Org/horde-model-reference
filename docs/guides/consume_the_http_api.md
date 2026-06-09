# Consume the HTTP API

**Goal:** integrate the model-reference read API into a worker, client, or tool - reliably and
without hammering the server. This guide covers the two ways to consume the data (direct HTTP vs.
the Python library) and how to detect changes.

For a first tour of the endpoints, do the [Using the HTTP API](../tutorials/using_the_http_api.md)
tutorial first.

## Choose your integration style

| | Direct HTTP | Python library (REPLICA mode) |
| --- | ----------- | ----------------------------- |
| You get | Raw JSON | Typed Pydantic records + a fluent query API |
| Caching | You implement it | Built-in TTL cache |
| Resilience | You implement retries/fallback | PRIMARY -> GitHub fallback built in |
| Best for | Non-Python consumers, dashboards, browsers | Python workers and clients |

If you are in Python, prefer the library - it calls the **same endpoints** under the hood but hands
you caching, retries, and GitHub fallback for free.

## Option A - Direct HTTP

Point at the public base URL and fetch what you need:

```python
import httpx

BASE = "https://models.aihorde.net/api"

with httpx.Client(base_url=BASE, timeout=15) as client:
    image_models = client.get("/model_references/v2/image_generation").json()
    names = list(image_models)
```

Guidelines for a well-behaved consumer:

- **Cache responses.** The data changes infrequently (a moderated queue gates every change). Cache
  per category for at least a minute.
- **Use change detection** (below) instead of re-downloading on every tick.
- **Handle `503`/`5xx`** by backing off and serving your last-good cache.

## Option B - The library in REPLICA mode

Set the PRIMARY URL (the `/api` base) and read typed records. The default already points at the
public deployment, so often you configure nothing:

```bash
export HORDE_MODEL_REFERENCE_PRIMARY_API_URL=https://models.aihorde.net/api
```

```python
from horde_model_reference import ModelReferenceManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()
records = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
sdxl = manager.get_model(MODEL_REFERENCE_CATEGORY.image_generation, "stable_diffusion_xl")
```

Under the hood the `HTTPBackend` issues `GET {PRIMARY_API_URL}/model_references/v2/{category}` and
caches the result. If the PRIMARY is unreachable it falls back to the raw GitHub repositories. See
[Offline & Resilient Reads](offline_and_resilient_reads.md) for the full fallback chain and
[Filter Models for a Worker](filter_models_for_a_worker.md) for narrowing the set a node serves.

## Detect changes without re-downloading

Each category exposes a last-updated timestamp. Poll it cheaply and only re-fetch when it moves:

```bash
curl https://models.aihorde.net/api/model_references/v2/metadata/image_generation/last_updated
# {"category":"image_generation","last_updated":1700000000}
```

```python
import httpx

BASE = "https://models.aihorde.net/api"
last_seen: dict[str, int] = {}

def refresh_if_changed(client: httpx.Client, category: str) -> dict | None:
    meta = client.get(f"/model_references/v2/metadata/{category}/last_updated").json()
    stamp = meta["last_updated"]
    if last_seen.get(category) == stamp:
        return None  # unchanged; keep your cache
    last_seen[category] = stamp
    return client.get(f"/model_references/v2/{category}").json()
```

The `metadata/*` endpoints return `503` on REPLICA deployments. The public PRIMARY supports them.

If you read through the library (REPLICA mode) you do not need to poll: the in-memory TTL cache
refetches on its own once data is older than `HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS` (default
60 s). Lengthen the TTL for fewer network calls, shorten it for fresher data. Pass
`overwrite_existing=True` on a single read to force an immediate refetch.

For explicit change polling (e.g. on a PRIMARY deployment), `manager.last_updated(category)`
returns the category's last-update unix timestamp without reaching into the backend. It returns
`None` when the backend does not track metadata (REPLICA backends). `manager.supports_metadata()`
and `manager.get_metadata(category)` give the full picture.

## Pick the right read shape

- **Whole category:** `GET /model_references/v2/{category}` - best for "load everything once".
- **One model:** `GET /model_references/v2/{category}/model/{model_name}`.
- **Filtered/sorted/paged:** `GET /model_references/v2/{category}/search` - best for UIs.
- **Ranked by usage:** `GET /model_references/v2/{category}/popular` (image/text only).

See the [v2 Endpoints reference](../reference/http_api/v2_endpoints.md) for every parameter.

## Legacy consumers

Existing AI-Horde workers that read the original JSON shape can use the **v1** endpoints
(`/model_references/v1/{category}`), which return the legacy format unchanged. New integrations
should use v2. See [v1 Endpoints](../reference/http_api/v1_endpoints.md).
