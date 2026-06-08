# Using the HTTP API

This tutorial walks through your first calls against the live Horde Model Reference service. Every
example targets the public PRIMARY deployment:

```
https://models.aihorde.net/api
```

You will list categories, fetch a category and a single model, search, and rank models by live
usage - all read-only and unauthenticated. For the moderated write workflow, see
[Submit Models via the API](../guides/submit_models_via_the_api.md).

!!! tip "Browse interactively first"
    Open [`/api/docs`](https://models.aihorde.net/api/docs) in a browser to see every endpoint with
    "Try it out". The rest of this page mirrors that surface from the command line and Python.

## 1. Check the deployment

Before assuming anything, ask the service what it is:

```bash
curl https://models.aihorde.net/api/replicate_mode
# {"replicate_mode":"PRIMARY","canonical_format":"v2","writable":true}

curl https://models.aihorde.net/api/heartbeat
# {"status":"ok","ai_horde":{"degraded":false,"consecutive_failures":0,"seconds_until_retry":null}}
```

`writable:true` means this deployment accepts (queued) writes; `canonical_format:"v2"` means writes
go through the v2 API.

## 2. List the model categories

```bash
curl https://models.aihorde.net/api/model_references/v2/model_categories
# ["image_generation","text_generation","clip","controlnet","blip","esrgan", ...]
```

## 3. Fetch a whole category

```bash
curl https://models.aihorde.net/api/model_references/v2/image_generation
```

The response is a JSON object keyed by model name. Each value is a full record - baseline, NSFW
flag, description, download URLs and SHA-256 checksums, size on disk, and metadata timestamps:

```json
{
  "stable_diffusion_xl": {
    "name": "stable_diffusion_xl",
    "baseline": "stable_diffusion_xl",
    "nsfw": false,
    "description": "...",
    "config": { "download": [ { "file_name": "...", "sha256sum": "...", "file_url": "..." } ] },
    "size_on_disk_bytes": 6938040714,
    "metadata": { "created_at": 1700000000, "updated_at": 1700000000 }
  }
}
```

## 4. Fetch a single model

```bash
curl https://models.aihorde.net/api/model_references/v2/image_generation/stable_diffusion_xl
```

Returns just that one record. A missing model returns `404` with `{"detail": "..."}`.

## 5. Search within a category

The search endpoint filters, sorts, and paginates server-side:

```bash
# Non-NSFW SDXL models, newest first, first page of 10
curl "https://models.aihorde.net/api/model_references/v2/image_generation/search?nsfw=false&baseline=stable_diffusion_xl&limit=10&sort_desc=true&sort_by=name"
```

```json
{ "results": [ ... ], "total": 23, "offset": 0, "limit": 10, "has_more": true }
```

Common filters: `nsfw`, `baseline`, `tags_any` / `tags_all` / `tags_none`, `name_contains`,
`inpainting` (image), `backend` / `quantized` (text). See the
[v2 search reference](../reference/http_api/v2_endpoints.md#search) for the full list.

## 6. Rank models by live usage

The popularity endpoint joins in live worker/usage data from the AI-Horde API
(`image_generation` and `text_generation` only):

```bash
curl "https://models.aihorde.net/api/model_references/v2/image_generation/popular?limit=10&sort_by=worker_count"
```

## 7. The same calls in Python

Any HTTP client works. With [`httpx`](https://www.python-httpx.org/):

```python
import httpx

BASE = "https://models.aihorde.net/api"

with httpx.Client(base_url=BASE, timeout=15) as client:
    categories = client.get("/model_references/v2/model_categories").json()
    sdxl = client.get("/model_references/v2/image_generation/stable_diffusion_xl").json()
    page = client.get(
        "/model_references/v2/image_generation/search",
        params={"nsfw": False, "limit": 10, "sort_by": "name"},
    ).json()

    print(len(categories), "categories")
    print("baseline:", sdxl["baseline"])
    print(page["total"], "matches; has_more:", page["has_more"])
```

!!! note "Or skip HTTP entirely"
    If you are writing Python, the [library](getting_started.md) gives you the same data as typed
    Pydantic records with caching and GitHub fallback built in - no need to hand-roll HTTP. The
    [Consume the HTTP API](../guides/consume_the_http_api.md) guide compares both approaches.

## Where to go next

- [HTTP API Conventions](../reference/http_api/conventions.md) - base URL, auth, errors, pagination.
- [v2 Endpoints](../reference/http_api/v2_endpoints.md) - every read and write endpoint in detail.
- [Submit Models via the API](../guides/submit_models_via_the_api.md) - the authenticated write flow.
- [The HTTP Service and Where It Fits](../concepts/http_service.md) - ecosystem context.
