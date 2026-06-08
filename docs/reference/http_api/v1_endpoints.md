# v1 Endpoints (Legacy)

The v1 API, rooted at `/api/model_references/v1`, serves the **original GitHub-compatible format**.
It exists for backward compatibility with existing AI-Horde workers and tooling that already read
the legacy JSON shape; its outward request/response structure is intentionally frozen. **New
integrations should use [v2](v2_endpoints.md).**

All paths are relative to `https://models.aihorde.net/api`. See [Conventions](conventions.md) for
base URL, auth, and errors.

!!! info "Reads work everywhere; writes depend on canonical format"
    Any deployment can serve v1 **reads**. v1 **writes** are only enabled when the deployment's
    `canonical_format` is `legacy` (the public PRIMARY uses `v2`, so its v1 write endpoints return
    `503`). See [Canonical Format](../../concepts/canonical_format.md).

## Read operations

| Method & path | Description |
| ------------- | ----------- |
| `GET /model_references/v1/info` | API information message. |
| `GET /model_references/v1/model_categories` | List all category names. |
| `GET /model_references/v1/text_generation?include_group=false` | Text models in legacy form. With `include_group=true`, adds computed `text_model_group` / `text_model_group_summary` fields. |
| `GET /model_references/v1/{category}` | A category in legacy format. Accepts the `stable_diffusion` alias and a trailing `.json`. Returns `{}` for `lora`/`ti`; `404` for empty categories. |

Read responses are the legacy JSON shape (flat `{model_name: {...}}`), matching the original
`Haidra-Org/AI-Horde-*-model-reference` repositories.

## Write operations

Enabled only when `canonical_format='legacy'` **and** the deployment is PRIMARY. Each write requires
a requestor `apikey`. Unlike v2, direct legacy writes can apply synchronously (`201`/`200`) or be
queued (`202`) depending on deployment configuration.

| Method & path | Body | Description |
| ------------- | ---- | ----------- |
| `POST /model_references/v1/image_generation` | `LegacyStableDiffusionRecord` | Create an image model. `201`/`202`; `409` if it exists. |
| `PUT /model_references/v1/image_generation` | `LegacyStableDiffusionRecord` | Update an image model. |
| `POST /model_references/v1/text_generation` | `LegacyTextGenerationRecord` | Create a text model. |
| `PUT /model_references/v1/text_generation` | `LegacyTextGenerationRecord` | Update a text model. |
| `POST /model_references/v1/clip` | `LegacyClipRecord` | Create a CLIP model. |
| `POST /model_references/v1/controlnet` | `LegacyControlnetRecord` | Create a ControlNet model. |
| `POST /model_references/v1/blip` | `LegacyBlipRecord` | Create a BLIP model. |
| `POST /model_references/v1/esrgan` | `LegacyEsrganRecord` | Create an ESRGAN model. |
| `POST /model_references/v1/gfpgan` | `LegacyGfpganRecord` | Create a GFPGAN model. |
| `POST /model_references/v1/codeformer` | `LegacyCodeformerRecord` | Create a CodeFormer model. |
| `POST /model_references/v1/safety_checker` | `LegacySafetyCheckerRecord` | Create a safety-checker model. |
| `POST /model_references/v1/miscellaneous` | `LegacyMiscellaneousRecord` | Create a miscellaneous model. |
| `DELETE /model_references/v1/{category}/{model_name}` | - | Delete a model. `204` (applied) or `202` (queued); `404` if missing. |

When the deployment routes legacy writes through the review queue, these return `202` and a
`PendingChangeRecord` - see [Pending Queue endpoints](pending_queue_endpoints.md).

## Metadata

| Method & path | Description |
| ------------- | ----------- |
| `GET /model_references/v1/metadata/last_updated` | Canonical last-updated (legacy canonical format only). |
| `GET /model_references/v1/metadata/metadata` | Metadata for all categories. |
| `GET /model_references/v1/metadata/{category}` | Metadata for one category. |
| `GET /model_references/v1/metadata/{category}/last_updated` | Last-updated timestamp for one category. |

These require a PRIMARY deployment whose canonical format is `legacy`; otherwise `503`.

## Pending queue

The v1 prefix also mounts the full review workflow at `/model_references/v1/pending_queue/...`,
identical in shape to the [v2 pending-queue endpoints](pending_queue_endpoints.md). The active
prefix matches the deployment's canonical format.

## See also

- [v2 Endpoints](v2_endpoints.md) - the current format (use this for new work).
- [Legacy CSV Conversion](../legacy_csv_conversion.md) - how legacy text CSV maps to records.
- [Canonical Format](../../concepts/canonical_format.md) - which version accepts writes.
