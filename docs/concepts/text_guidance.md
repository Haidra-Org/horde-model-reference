# Text Model Usage Guidance

Prompting knowledge for text models is reusable data, held in its own catalog rather than embedded in each
model record. Most models sharing a prompt style share one authored description of it, so the catalog stores
that description once and points models at it.

The catalog lives in `text_guidance/catalog.json` and holds two collections:

| Collection | Type | Purpose |
| --- | --- | --- |
| `profiles` | `TextPromptContract` or `TextUsageRecipe` | Reusable authored guidance keyed by `profile_id`. |
| `assignments` | `TextGuidanceAssignment` | Which profiles apply to one exact canonical model name. |

`TextGuidanceCatalogMetadata` carries `schema_version`, a monotonic `revision`, and `updated_at`. The revision
is exposed on every read as a weak `ETag` (`W/"text-guidance-<revision>"`).

## Profiles

Both profile kinds share `profile_id` (lowercase URL-safe), `display_name`, `aliases`, `summary`, `examples`,
`recommended_settings`, `ai_horde_examples`, `sources`, `deprecated`, and lifecycle `metadata`. They also carry
two `GuidanceAudienceContent` blocks, `user` and `developer`, each with `overview`, `use_cases`, `tips`, and
`caveats`. Prose is CommonMark; raw HTML is rejected at validation.

**Prompt contract** (`kind: prompt_contract`) describes how a model wants a prompt serialized:
`interaction_modes` (at least one of `completion`, `instruction`, `chat`), `accepted_roles`, `role_markers`,
`stop_sequences`, and `templates`. A `RawPromptTemplate` is stored and displayed, never executed, and is tagged
with its `syntax` (`jinja2`, `handlebars`, `python_format`, `literal`, `other`). When the syntax is `other`, a
`syntax_name` is required.

**Usage recipe** (`kind: usage_recipe`) supplements a contract for one scenario, optionally naming a
`capability` (`tool_calling`, `structured_output`) and a free-text `scenario`.

Aliases are unique across the catalog, compared case-insensitively.

## Assignments and resolution

An assignment names one `primary_profile_id` (which must be an active prompt contract) and any number of
`supplemental_profile_ids` (which must be active usage recipes). A profile may appear only once per model.
Assignment keys are canonical text model names: names carrying a legacy backend prefix are rejected, as are
names absent from the canonical `text_generation` reference.

Resolution produces a `TextGuidanceSummary` with one of three statuses:

- `published`: an explicit assignment exists; the summary carries the profile identifiers.
- `legacy_label`: no assignment, but the model record has a non-empty `instruct_format`.
- `undocumented`: neither.

`ResolvedTextGuidance` adds the full primary contract, the ordered supplemental recipes, the raw
`legacy_instruct_format`, and the catalog metadata.

## Relationship to legacy `instruct_format`

`instruct_format` is a bare label on the model record with no authored content behind it. It stays in place and
still resolves as `legacy_label`, so nothing breaks before a model is documented. The migration preview route
reads those labels, groups models by label, and synthesizes a draft prompt contract per distinct label plus an
assignment per model that does not already have one. Labels matching an existing profile's `display_name` or
alias reuse that profile instead of creating a new one. Nothing is stored: the preview returns a
`TextGuidanceChangeSet` for a human to edit and submit.

## Change sets and the pending queue

The catalog is never edited in place over HTTP. Edits are packaged as a `TextGuidanceChangeSet` (a `title` plus
`profile_changes` and `assignment_changes`, at least one of them non-empty) and submitted to the shared review
queue. Each profile change names an `operation` (`create`, `update`, `deprecate`) and an `expected_before`
value; each assignment change carries the replacement assignment (or `null` to remove it) and its own
`expected_before`. Applying compares `expected_before` against the live record and raises
`GuidanceConflictError` when the record moved after review.

Queue records distinguish resources with `PendingResourceKind`. A guidance submission is stored with
`resource_kind=text_guidance`, a UUID `resource_id`, and a compatibility `model_name` of `guidance:<uuid>`.
On apply, `pending_queue/apply.py` routes those records to `TextGuidanceStore.apply_change_set` rather than
the model write path; beta materialization skips them. See
[Pending Queue Architecture](../reference/pending_queue.md).

Submission validates before enqueueing: unknown text generation settings keys in `recommended_settings` or
`ai_horde_examples[].parameters` are rejected with `422`, and the change set is dry-run against the store so
an unsatisfiable proposal never reaches an approver. Applying bumps the catalog `revision`, stamps profile and
assignment `metadata`, and writes `catalog.json` atomically.

## Where guidance appears in ordinary reads

`TextGenerationModelRecord` gained `context_window`, `interaction_modes`, and `capabilities` as durable
reviewed data stored on the record, plus a read-only `guidance` summary. The `guidance` field is never
persisted or accepted on write: the v2 service excludes it from stored payloads and attaches it to responses
from the catalog. Group responses under `/text_generation/group/{group_name}` carry a per-member `guidance`
summary and a `guidance_coverage` count by status.

## Replica hydration

PRIMARY owns a writable store. A REPLICA using `HTTPBackend` calls
`HTTPBackend.fetch_text_guidance_export()` the first time `ModelReferenceManager.text_guidance_store` is
accessed, validates the payload, and replaces its local read-only copy. A failed or malformed fetch is logged
and the packaged bootstrap catalog under `horde_model_reference/data/guidance/` is used instead. The refresh
is attempted once per manager instance.

## Settings

`TextGuidanceSettings` is nested under the main settings object:

| Setting | Environment variable | Default |
| --- | --- | --- |
| `relative_subdir` | `HORDE_MODEL_REFERENCE_TEXT_GUIDANCE__RELATIVE_SUBDIR` | `text_guidance` |
| `root_path_override` | `HORDE_MODEL_REFERENCE_TEXT_GUIDANCE__ROOT_PATH_OVERRIDE` | unset |

`HordeModelReferencePaths.text_guidance_path` resolves the override first, then the subdirectory under the
model reference base path.

## Generated Markdown

Markdown is a view, never the source of truth:

```bash
uv run generate-guidance-reference \
  --primary-api-url https://models.aihorde.net/api \
  --output review_artifacts/text-guidance.md
```

Use `--input` with a saved `/text_generation/guidance/export` response instead of `--primary-api-url` to
render offline. Output is deterministic (profiles ordered by identifier, assignments sorted) and begins with a
generated-file warning.

See [Text guidance endpoints](../reference/http_api/text_guidance_endpoints.md) for the HTTP surface.
