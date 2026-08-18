# Text Guidance Endpoints

Reusable prompting guidance for text models is served under
`/model_references/v2/text_generation/guidance`. All paths below are relative to
`https://models.aihorde.net/api`; see [Conventions](conventions.md) for the base URL, auth header, and
shared error semantics. Background on the data model is in
[Text Model Usage Guidance](../../concepts/text_guidance.md).

Reads are public and available on REPLICA deployments (served from the replica's hydrated copy of the PRIMARY
export). Every read sets a weak revision `ETag` of the form `W/"text-guidance-<revision>"`.

## Public reads

| Method and path | Response model | Description |
| --- | --- | --- |
| `GET /profiles` | `TextUsageProfilePage` | Profile index in `profile_id` order. |
| `GET /profiles/{profile_id}` | `TextPromptContract` or `TextUsageRecipe` | One profile, discriminated by `kind`. |
| `GET /assignments` | `TextGuidanceAssignmentPage` | Explicit exact-model assignments in model-name order. |
| `GET /model` | `ResolvedTextGuidance` | Resolved guidance for one exact model. |
| `GET /export` | `TextGuidanceCatalog` | Complete validated catalog. |

### `GET /profiles`

Query parameters:

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `include_deprecated` | bool | `false` | Include profiles flagged `deprecated`. |

Returns `{ items, total, metadata }`. Each item is a `TextUsageProfileSummary`: `profile_id`, `kind`,
`display_name`, `summary`, `aliases`, `deprecated`, and `assigned_model_count` (how many assignments name the
profile as primary or supplemental). `total` counts the returned items, not the whole catalog.

### `GET /profiles/{profile_id}`

Returns the full profile record. `404` when no profile has that identifier. Deprecated profiles are still
returned here.

### `GET /assignments`

Query parameters:

| Name | Type | Meaning |
| --- | --- | --- |
| `model_name` | string | Exact-match filter on the assignment's model name. |
| `profile_id` | string | Keep assignments naming this profile as primary or supplemental. |

Returns `{ items, total, metadata }` where each item is a `TextGuidanceAssignment`.

### `GET /model`

Query parameters:

| Name | Type | Meaning |
| --- | --- | --- |
| `name` | string, required, min length 1 | Exact canonical text model name. |

Returns `ResolvedTextGuidance`: `model_name`, the `summary` (`published`, `legacy_label`, or `undocumented`),
`primary_profile`, `supplemental_profiles`, `legacy_instruct_format`, and `catalog_metadata`.

Errors:

- `404` when the name is not a canonical text model. Names carrying a legacy backend prefix are excluded from
  the lookup and so also return `404`.
- `422` when `name` is missing or empty.

### `GET /export`

Returns the whole `TextGuidanceCatalog` (`metadata`, `profiles`, `assignments`). This is the payload REPLICA
managers hydrate from and the input the `generate-guidance-reference` CLI accepts via `--input`.

## Authenticated operations

Both routes require an `apikey` header belonging to a user on the pending-queue requestor allowlist and answer
`503` on a REPLICA, where there is no queue to submit to.

| Method and path | Auth | Success | Description |
| --- | --- | --- | --- |
| `POST /migration/preview` | requestor, PRIMARY | `200` | Build a draft change set from legacy `instruct_format` labels. |
| `POST /change-sets` | requestor, PRIMARY | `202` | Validate and enqueue a guidance change set. |

Shared auth failures: `401` when the header is missing or the key is invalid (or no requestor allowlist is
configured), `403` when the key authenticates but the user is not on the allowlist, `503` when the AI Horde
auth service is unreachable.

### `POST /migration/preview`

No request body. Reads canonical text models, groups those with a non-empty `instruct_format`, and returns
`GuidanceMigrationPreview`:

| Field | Meaning |
| --- | --- |
| `change_set` | A `TextGuidanceChangeSet` proposal, or `null` when nothing needs migrating. |
| `source_model_count` | Number of assignment changes in the proposal. |
| `format_count` | Number of distinct new prompt contracts in the proposal. |

Labels matching an existing profile's `display_name` or an alias (case-insensitive) reuse that profile rather
than producing a new one. Models that already have an assignment are skipped. New profile identifiers are
slugified from the label, with a numeric suffix on collision. Nothing is persisted: edit the returned change
set and submit it to `/change-sets`.

### `POST /change-sets`

Request body: `TextGuidanceChangeSet`.

```json
{
  "title": "Document the ChatML prompt contract",
  "profile_changes": [
    {
      "operation": "create",
      "profile_id": "chatml",
      "profile": {
        "profile_id": "chatml",
        "kind": "prompt_contract",
        "display_name": "ChatML",
        "summary": "Role-tagged chat serialization.",
        "interaction_modes": ["chat"]
      },
      "expected_before": null
    }
  ],
  "assignment_changes": [
    {
      "model_name": "koboldcpp/Example-7B",
      "assignment": {
        "model_name": "koboldcpp/Example-7B",
        "primary_profile_id": "chatml"
      },
      "expected_before": null
    }
  ]
}
```

On success the response is `202` with the created `PendingChangeRecord`. The record is stored with
`category=text_generation`, `operation=UPDATE`, `resource_kind=text_guidance`, a UUID `resource_id`, and
`model_name="guidance:<resource_id>"`. `notes` and `request_metadata.title` carry the change-set title;
`related_models` lists the models named by `assignment_changes`.

Errors:

- `422` when the body fails `TextGuidanceChangeSet` validation (including an empty change set), or when
  `recommended_settings` or an `ai_horde_examples[].parameters` entry uses a key that is not a valid text
  generation setting. The message lists the offending keys.
- `409` when an `expected_before` value no longer matches the live catalog (the profile or assignment
  changed after the proposal was reviewed).
- `422` when the dry run against the live catalog fails for any other reason: unknown or non-canonical
  models, missing or deprecated profiles, duplicate aliases, or a create for an existing profile. The
  message is the store's description of the problem.
- `503` when the deployment is not PRIMARY, or when the pending queue service is not configured.

The dry run means proposals with these defects never reach an approver.

## Applying a guidance change set

Approval and application go through the ordinary queue endpoints documented in
[Pending Queue Endpoints](pending_queue_endpoints.md). Applying a record whose `resource_kind` is
`text_guidance` writes the catalog instead of a model record, and
`GET /pending_queue/changes/{id}/diff` renders the whole current catalog against the prospective one.

## Guidance in other responses

The v2 `text_generation` category, single-model, and pending responses attach a read-only `guidance` summary
to each record. `GET /text_generation/group/{group_name}` attaches a per-member `guidance` and a
`guidance_coverage` map; see [v2 Text Utilities](v2_text_utils.md).
