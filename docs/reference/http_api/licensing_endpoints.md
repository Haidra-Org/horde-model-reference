# Licensing Endpoints

The v2 licensing API is rooted at `/api/model_references/v2/licensing`. Public reads require no authentication.

## Public reads

| Method and path | Description |
| --- | --- |
| `GET /licenses` | Paginated reusable definitions. Supports `include_deprecated`, `offset`, and `limit`. |
| `GET /licenses/{license_id}` | One definition with its canonical URL and permission defaults. |
| `GET /assets` | Unified models, custom nodes, software components, and other licensed assets. |
| `GET /assets/{asset_kind}/{asset_identifier}` | One directly managed non-model asset. |
| `GET /models/{category}/{model_name}` | Detailed licensing view for one model. |
| `GET /summary` | Counts by commercial-use status, redistribution status, and license identifier. |
| `GET /export` | Deterministic complete snapshot for offline consumers and Markdown generation. |

`GET /assets` supports `asset_kind`, `category`, `license_id`, `commercial_use`, `redistribution`, `name_contains`,
`offset`, and `limit`. Each unified asset view includes `definition_urls`, so a client can move from a conclusion to
the reusable license records without constructing paths.

The ordinary v2 category, single-model, pending, and search responses also include `licensing`. Pre-migration
records are represented explicitly as `NOASSERTION` and `unknown`; consumers do not need to infer meaning from a
missing field. Category and cross-category search endpoints accept `license_id`, `commercial_use`, and
`redistribution` filters.

## Direct editor CRUD

The following operations require PRIMARY mode, an `apikey` header, and membership in the independent licensing
editor allowlist:

| Method and path | Description |
| --- | --- |
| `POST /licenses` | Create a definition. |
| `PUT /licenses/{license_id}` | Replace a definition while preserving creation metadata. |
| `DELETE /licenses/{license_id}` | Delete only when no model, file, or auxiliary asset references it. |
| `POST /assets` | Create a non-model asset. |
| `PUT /assets/{asset_kind}/{asset_identifier}` | Replace a non-model asset. |
| `DELETE /assets/{asset_kind}/{asset_identifier}` | Delete a non-model asset. |

These writes are intentionally outside the model pending queue and append to the dedicated licensing audit stream.
Model licensing changes still use the normal model `PUT` route and pending review workflow.

`GET /model_references/v2/me/roles` exposes `is_license_editor` and includes `license_editor` in `roles` when
applicable. An empty editor allowlist rejects every direct licensing write.

Licensing records communicate reviewed data and evidence; they are not legal advice. Treat `unknown` and
`prohibited` as fail-closed, and surface obligations whenever a status is `allowed_with_conditions`.
