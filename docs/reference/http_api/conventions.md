# HTTP API Conventions

Cross-cutting rules that apply to every endpoint in the [v2](v2_endpoints.md),
[v1](v1_endpoints.md), [text-utils](v2_text_utils.md), and [pending-queue](pending_queue_endpoints.md)
references. For the bigger picture of who calls the API and why, see
[The HTTP Service and Where It Fits](../../concepts/http_service.md).

## Base URL and the `/api` prefix

The public deployment is served at:

```
https://models.aihorde.net/api
```

Every path in these references is **relative to that base**. For example, the v2 category endpoint
documented as `/model_references/v2/image_generation` is reached at:

```
https://models.aihorde.net/api/model_references/v2/image_generation
```

The service sets FastAPI's `root_path="/api"`, so the interactive docs at
[`/api/docs`](https://models.aihorde.net/api/docs) and the OpenAPI schema at `/api/openapi.json`
already include the `/api` prefix in their server URL.

!!! note "Library default"
    When using the Python library as a REPLICA client, set
    `HORDE_MODEL_REFERENCE_PRIMARY_API_URL=https://models.aihorde.net/api` (note the `/api`). The
    `HTTPBackend` appends `/model_references/v2/{category}` to that base. This is the built-in
    default; you only need to set it to point at a different deployment.

## Interactive documentation

| URL | What it is |
| --- | ---------- |
| [`/api/docs`](https://models.aihorde.net/api/docs) | Swagger UI - browse and "Try it out" |
| `/api/redoc` | ReDoc - clean reference rendering |
| `/api/openapi.json` | Machine-readable OpenAPI 3.1 schema (for client codegen) |

Because every category exposes a **typed** create/update/read route, generated SDK clients see a
concrete model (e.g. `ImageGenerationModelRecord`) rather than a union - see
[v2 Endpoints -> Typed per-category routes](v2_endpoints.md#typed-per-category-routes).

## Versioning

Two parallel versions live under `/api/model_references/`:

- **`/v2`** - the current format. Prefer it for new work.
- **`/v1`** - the legacy GitHub-compatible format, kept for existing AI-Horde workers. Its outward
  request/response shape is frozen.

Both are readable everywhere. See [Canonical Format](../../concepts/canonical_format.md) for which
version accepts writes on a given deployment.

## Content type

All request and response bodies are JSON (`application/json`). Read endpoints that return stored
records do so as raw JSON passthrough - the documented `response_model` describes the schema, but the
bytes returned are the canonical stored record, unfiltered.

## Authentication

Read endpoints are **unauthenticated**.

Write endpoints and pending-queue operations require an AI-Horde API key supplied in the **`apikey`
HTTP header** (the same key you use elsewhere on the AI-Horde):

```bash
curl -H "apikey: $AI_HORDE_API_KEY" ...
```

The key is validated against the AI-Horde `/v2/find_user` endpoint, then checked against the
deployment's role allowlists:

- **requestor** - may propose changes (create/update/delete -> enqueue).
- **approver** - may review, approve/reject, and apply queued changes. Approvers are also requestors.

Call [`GET /api/model_references/v2/me/roles`](v2_endpoints.md#get-model_referencesv2meroles) to
see which roles your key has on a deployment. If an allowlist is not configured, the deployment
fails closed (all such requests are rejected).

## Capability discovery

Clients should not assume a deployment is writable. Probe it instead:

```bash
curl https://models.aihorde.net/api/replicate_mode
# {"replicate_mode": "PRIMARY", "canonical_format": "v2", "writable": true}
```

- `writable=false` -> the instance is a REPLICA; all writes return `503`.
- `canonical_format` -> which API version (`v2` or `legacy`) accepts writes; the other returns `503`
  for writes.

## Status codes

| Code | Meaning |
| ---- | ------- |
| `200 OK` | Successful read, or successful synchronous operation |
| `201 Created` | Resource created synchronously (legacy v1 direct writes) |
| `202 Accepted` | **Write accepted and queued for review** (the normal result of a v2 create/update/delete) |
| `204 No Content` | Successful delete with no body |
| `400 Bad Request` | Malformed request or a violated precondition (e.g. path/body name mismatch) |
| `401 Unauthorized` | Missing/invalid `apikey`, or the key lacks the required role |
| `404 Not Found` | Category or model does not exist |
| `409 Conflict` | Resource already exists (create), or a uniqueness conflict (alias/family) |
| `422 Unprocessable Entity` | Request body failed validation, or an unknown category was supplied |
| `500 Internal Server Error` | Unexpected server error |
| `503 Service Unavailable` | Writes attempted on a REPLICA, or against the non-canonical API version; or an upstream dependency is down |

## Error shape

Errors use FastAPI's standard envelope. A simple error is a string:

```json
{ "detail": "Model 'foo' not found in category 'image_generation'" }
```

A validation error (`422`) is a list of field-level problems:

```json
{
  "detail": [
    { "loc": ["body", "baseline"], "msg": "Field required", "type": "missing" }
  ]
}
```

This is described in the OpenAPI schema by the `ErrorResponse` / `HTTPValidationError` models.

## Pagination

List/search endpoints accept `limit` and `offset` query parameters. Search responses wrap results
in an envelope with explicit totals:

```json
{
  "results": [ ... ],
  "total": 137,
  "offset": 0,
  "limit": 50,
  "has_more": true
}
```

`has_more` tells a UI whether to offer a "next page". Typical bounds are `limit` 1–500 (default 50)
and `offset` ≥ 0; per-endpoint limits are noted in the [v2 reference](v2_endpoints.md).

## CORS

The service enables CORS using the deployment's `cors_allowed_origins` setting, so browser-based
clients on allow-listed origins can call the read API directly.

## Rate limiting

The application itself does not rate-limit; any limits are imposed by the deployment's edge/proxy.
Be a good citizen: cache read responses (they change infrequently) and prefer the library's
`HTTPBackend`, which caches with a TTL and falls back to GitHub on failure.
