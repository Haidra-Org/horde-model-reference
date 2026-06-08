# v2 Endpoints

The current-format API, rooted at `/api/model_references/v2`. All paths below are relative to
`https://models.aihorde.net/api`. Read the [Conventions](conventions.md) page first for base URL,
auth, error shape, and status codes. For the text-generation grouping toolkit see
[v2 Text Utilities](v2_text_utils.md); for the review workflow see
[Pending Queue endpoints](pending_queue_endpoints.md).

`{category}` is one of the [model categories](../../concepts/architecture_overview.md): `image_generation`,
`text_generation`, `clip`, `controlnet`, `blip`, `esrgan`, `gfpgan`, `codeformer`, `safety_checker`,
`video_generation`, `audio_generation`, `miscellaneous`, `lora`, `ti`.

## Read operations

Unauthenticated and safe to cache.

| Method & path | Description |
| ------------- | ----------- |
| `GET /model_references/v2/info` | API information message. |
| `GET /model_references/v2/model_categories` | List all category names. |
| `GET /model_references/v2/{category}` | All models in a category, as `{name: record}`. `404` if unknown/empty. |
| `GET /model_references/v2/{category}/{model_name}` | A single model record. `404` if missing. |
| `GET /model_references/v2/{category}/search` | Filter/sort/paginate within a category (see [Search](#search)). |
| `GET /model_references/v2/search` | Search across all categories. |
| `GET /model_references/v2/{category}/popular` | Rank by live Horde usage (`image_generation`, `text_generation` only). |
| `GET /model_references/v2/metadata/last_updated` | Canonical last-updated timestamp (PRIMARY, `v2` only). |
| `GET /model_references/v2/metadata/metadata` | All categories' metadata (`{category: CategoryMetadata}`). |
| `GET /model_references/v2/metadata/{category}` | Metadata for one category. |
| `GET /model_references/v2/metadata/{category}/last_updated` | Last-updated timestamp for one category. |
| `GET /model_references/statistics/{category}` | Aggregated category statistics (see [Statistics](#statistics)). |
| `GET /model_references/statistics/{category}/deletion-risk` | Usage-informed deletion-risk analysis. |
| `GET /model_references/v2/me/roles` | The authenticated caller's roles (requires `apikey`). |

!!! note "Why `metadata/metadata`?"
    The all-categories route really is `…/metadata/metadata` -- the metadata router is mounted under a
    `/metadata` prefix and its "everything" operation is itself named `metadata`. The doubling is
    intentional, not a typo; the single-category route is `…/metadata/{category}`.

### Typed per-category routes

Every category has a concrete record type, so the read routes above also exist as **typed
per-category operations** in the OpenAPI schema:

- `GET /model_references/v2/{category}` documents its response as `{name: <CategoryRecord>}`
  (e.g. `ImageGenerationModelRecord`, `ClipModelRecord`).
- `GET /model_references/v2/{category}/{model_name}` documents its response as the concrete
  `<CategoryRecord>`.

The runtime response is the raw stored JSON (passthrough); the concrete schema exists so generated
SDK clients get a real model type instead of a union. A generic parameterized route remains as a
uniform fallback (operation ids `read_v2_reference`, `read_v2_single_model`).

### Search

`GET /model_references/v2/{category}/search` and `GET /model_references/v2/search`.

| Query param | Type | Notes |
| ----------- | ---- | ----- |
| `nsfw` | bool | Filter by NSFW flag. |
| `baseline` | str | Exact baseline match. |
| `inpainting` | bool | `image_generation` only. |
| `tags_any` / `tags_all` / `tags_none` | list[str] | Tag set membership. |
| `name_contains` | str | Case-insensitive substring. |
| `backend` | str | `text_generation` backend filter. |
| `exclude_backend_variations` | bool | Collapse text backend duplicates. |
| `quantized` | bool | `text_generation` only. |
| `sort_by` | str | Any record field. |
| `sort_desc` | bool | Default `false`. |
| `limit` | int | 1–500, default 50. |
| `offset` | int | ≥ 0, default 0. |
| `source` | str | `horde` (default, canonical), `any` (canonical + providers), or a provider id. |

Response is a `SearchResponse` envelope: `{ "results": [...], "total", "offset", "limit", "has_more" }`.

### Popular

`GET /model_references/v2/{category}/popular` - joins live AI-Horde worker/usage data.

| Query param | Type | Notes |
| ----------- | ---- | ----- |
| `limit` | int | 1–100, default 10. |
| `sort_by` | enum | `worker_count` (default), `usage_day`, `usage_month`, `usage_total`. |
| `include_workers` | bool | Include per-worker detail. |

Only `image_generation` and `text_generation` carry usage data; other categories return an empty
list. Returns an empty list (not an error) if the upstream AI-Horde API is degraded - check
`/api/heartbeat`.

### Statistics

`GET /model_references/statistics/{category}` -> a `CategoryStatistics` object (counts overall /
NSFW / SFW, baseline distribution, download stats, tag/style distributions, category-specific
metrics).

| Query param | Type | Notes |
| ----------- | ---- | ----- |
| `group_text_models` | bool | Group `text_generation` variants by base name. |
| `limit` | int | Optional cap on returned models. |
| `offset` | int | ≥ 0, default 0. |

Cached (default 300 s); caching is skipped when grouping is enabled.

### Deletion risk

`GET /model_references/statistics/{category}/deletion-risk` -> a `CategoryDeletionRiskResponse`
combining local data with live usage.

| Query param | Type | Notes |
| ----------- | ---- | ----- |
| `group_text_models` | bool | Group text variants. |
| `include_backend_variations` | bool | `text_generation` only. |
| `preset` | str | One of `deletion_candidates`, `zero_usage`, `no_workers`, `missing_data`, `host_issues`, `critical`, `low_usage`. |
| `limit` / `offset` | int | Pagination. |

### `GET /model_references/v2/me/roles`

Requires `apikey`. Returns `{ "user_id", "username", "roles": [...], "is_approver", "is_requestor" }`.
Use it to decide which write/queue actions to offer in a client.

## Write operations

All writes require **PRIMARY mode**, **`canonical_format='v2'`**, and a requestor `apikey`. They
return **`202 Accepted`** with a `PendingChangeRecord` and enter the [pending queue](pending_queue_endpoints.md);
nothing changes in the live dataset until an approver applies it. A REPLICA or a mismatched
canonical format returns `503`. See [Submit Models via the API](../../guides/submit_models_via_the_api.md).

| Method & path | Body | Description |
| ------------- | ---- | ----------- |
| `POST /model_references/v2/{category}` | concrete `<CategoryRecord>` | **Typed create.** Name comes from the body. `409` if it already exists. |
| `PUT /model_references/v2/{category}/{model_name}` | concrete `<CategoryRecord>` | **Typed update.** Path `model_name` must equal the body `name` (`400` otherwise); `404` if missing. |
| `POST /model_references/v2/{category}/add` | `ModelRecordUnion` | **Generic create** - a single uniform endpoint covering any category. |
| `DELETE /model_references/v2/{category}/{model_name}` | - | Delete a model. |

Notes:

- The **typed** routes (`POST /{category}`, `PUT /{category}/{model_name}`) are preferred: they
  expose the category's concrete request schema, so a wrong-schema body is rejected with `422`.
- The **generic** `/add` and the generic `PUT /{category}/{model_name}` accept a discriminated
  union and exist as a uniform fallback (operation ids `create_v2_model`, `update_v2_model`,
  `delete_v2_model`).
- For `text_generation`, submit base model names only; backend-prefixed variants are generated
  automatically (submitting one returns `400`). `text_model_group` is auto-filled if omitted.

### Write status codes

| Code | When |
| ---- | ---- |
| `202` | Accepted and queued. |
| `400` | Path/body name mismatch, or a `text_generation` backend-prefixed name. |
| `401` | Missing/invalid `apikey`, or not a requestor. |
| `404` | Update/delete target does not exist. |
| `409` | Create target already exists. |
| `422` | Body failed validation, or unknown category. |
| `503` | REPLICA mode, or `canonical_format != 'v2'`. |

## Metadata

`GET /model_references/v2/metadata/...` returns per-category `last_updated` UNIX timestamps for
change detection (see [Consume the HTTP API -> Detect changes](../../guides/consume_the_http_api.md#detect-changes-without-re-downloading)).
These require a PRIMARY deployment whose canonical format is `v2`; otherwise they return `503`.

## See also

- [v2 Text Utilities](v2_text_utils.md) - the `text_generation` grouping/alias/family/naming toolkit.
- [Pending Queue endpoints](pending_queue_endpoints.md) - review and apply queued writes.
- [Canonical Format](../../concepts/canonical_format.md) - why writes route to one API version.
