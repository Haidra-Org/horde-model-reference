# v2 Text Utilities

`text_generation` models are organized into **groups** (variants of one base model), which can be
arranged into **families**, referenced by **aliases**, and named according to a **naming schema**.
These endpoints under `/api/model_references/v2/text_generation/...` manage that structure. Read
mutations (those that change model records) go through the [pending queue](pending_queue_endpoints.md);
schema/alias/family metadata changes are written directly on PRIMARY deployments.

All paths are relative to `https://models.aihorde.net/api`. See [Conventions](conventions.md) for
auth and errors. Writes here require a PRIMARY deployment and an `apikey`.

## Name parsing & composition

| Method & path | Auth | Description |
| ------------- | ---- | ----------- |
| `GET /model_references/v2/text_generation/parse_name?name=...` | - | Parse a model name into `{ original_name, base_name, size, variant, quant, version, suggested_group, extras }`. |
| `POST /model_references/v2/text_generation/compose_name` | - | Compose a name from parts and report collisions. Body: `ComposeNameRequest`. |
| `GET /model_references/v2/text_generation/distinct_baselines` | - | `{ "baselines": [...] }` - unique baseline values across text models. |

## Groups

| Method & path | Auth | Description |
| ------------- | ---- | ----------- |
| `GET /model_references/v2/text_generation/groups` | - | `{ "groups": [...] }` - all group names. |
| `GET /model_references/v2/text_generation/group/{group_name}` | - | Group members with parsed-name info, common fields, available sizes/variants/quants/versions, inferred or custom name format, health issues, related family. `404` if empty. |
| `GET /model_references/v2/text_generation/groups/summary` | - | Enriched overview of all groups (families, aliases, health flags). |
| `GET /model_references/v2/text_generation/groups/health` | - | Aggregate health issues across groups, with counts by issue type. |
| `PUT /model_references/v2/text_generation/group/{group_name}/common_fields` | requestor | Batch-update shared fields across all canonical members. Returns `202`; one `PendingChangeRecord` per member sharing a `batch_id`. |

## Naming schema

A group's naming schema defines how part fields (base, size, variant, quant, version) compose into
a model name. A group has an *inferred* schema unless a *custom* one is saved.

| Method & path | Auth | Description |
| ------------- | ---- | ----------- |
| `GET /model_references/v2/text_generation/group/{group_name}/name_schema` | - | The persisted custom schema, or the inferred one. |
| `PUT /model_references/v2/text_generation/group/{group_name}/name_schema` | requestor | Save a custom schema. Body: `GroupNameSchemaUpdateRequest`. |
| `DELETE /model_references/v2/text_generation/group/{group_name}/name_schema` | requestor | Delete the custom schema (revert to inferred). `204` / `404`. |
| `PUT /model_references/v2/text_generation/{model_name}/name_exception` | requestor | Set/clear a per-model "exempt from schema" flag. Body: `NameExceptionRequest`. Returns `202` (queued) with `pending_change_id`. |

## Aliases

Aliases are alternative names that resolve to a canonical group. An alias may belong to only one
group; claiming one held by another group returns `409`. These endpoints require an alias store
(`503` if unavailable).

| Method & path | Auth | Description |
| ------------- | ---- | ----------- |
| `GET /model_references/v2/text_generation/aliases` | - | All alias entries. |
| `GET /model_references/v2/text_generation/aliases/{canonical}` | - | Aliases for one canonical group. `404` if none. |
| `PUT /model_references/v2/text_generation/aliases/{canonical}` | requestor | Replace the full alias list. Body: `SetAliasesRequest`. `409` on collision. |
| `POST /model_references/v2/text_generation/aliases/{canonical}/add` | requestor | Add one alias. Body: `AddAliasRequest`. `409` on collision. |
| `POST /model_references/v2/text_generation/aliases/{canonical}/remove` | requestor | Remove one alias. Body: `RemoveAliasRequest`. |
| `DELETE /model_references/v2/text_generation/aliases/{canonical}` | requestor | Remove all aliases for the group. `204` / `404`. |

## Families

A family groups related groups together. A group may belong to only one family.

| Method & path | Auth | Description |
| ------------- | ---- | ----------- |
| `GET /model_references/v2/text_generation/families` | - | All families. |
| `GET /model_references/v2/text_generation/families/detect` | - | Auto-detect family suggestions. Params: `min_prefix_length` (2–20, default 3), `min_family_size` (2–50, default 2). Suggestions only - not persisted. |
| `GET /model_references/v2/text_generation/families/{family_name}` | - | One family. `404` if missing. |
| `PUT /model_references/v2/text_generation/families/{family_name}` | requestor | Create/replace a family. Body: `SetFamilyRequest`. `409` if a member belongs to another family. |
| `POST /model_references/v2/text_generation/families/{family_name}/add` | requestor | Add a group. Body: `AddFamilyMemberRequest`. `404`/`409`. |
| `POST /model_references/v2/text_generation/families/{family_name}/remove` | requestor | Remove a group. Body: `RemoveFamilyMemberRequest`. Family deleted if it becomes empty. |
| `DELETE /model_references/v2/text_generation/families/{family_name}` | requestor | Delete a family. `204` / `404`. |

## Notes

- "requestor" means a valid `apikey` on the deployment's requestor allowlist; these also require
  PRIMARY mode (`503` on REPLICA).
- Endpoints that mutate **model records** (e.g. `common_fields`, `name_exception`) enqueue
  `PendingChangeRecord`s and return `202`; they are applied via the
  [pending queue](pending_queue_endpoints.md). Endpoints that mutate only **structural metadata**
  (schemas, aliases, families) write directly and return `200`/`204`.
- Background on grouping and parsing: [Analytics Pipeline](../../concepts/analytics_pipeline.md).
