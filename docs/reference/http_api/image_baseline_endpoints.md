# Image Baseline Endpoints

The served image-generation baseline catalog is rooted at `/api/model_references/v2/image_generation/baselines`.
Baselines are a first-class resource, not a model category: the category enums are traversed as sets of model
records, and a baseline is not one.

Each record carries the family's `native_resolution` and `alternative_names`, the `capabilities` block (which kinds
of weights exist for the family at all), and the `horde_policy` block (kudos multipliers, batching, TTL, and the
resolution floor the AI-Horde applies). Capabilities describe the ecosystem; whether a worker's engine can run one
of them is a separate axis tracked by the worker.

## Public reads

| Method and path | Description |
| --- | --- |
| `GET /` | Every published baseline, in name order, with catalog metadata. |
| `GET /export` | The complete validated catalog, used by REPLICA hydration. |
| `GET /{baseline_name}` | One baseline, or 404. |

All three set a weak `ETag` of `W/"image-baselines-{revision}"` for inexpensive freshness checks.

## Change sets

| Method and path | Description |
| --- | --- |
| `POST /change-sets` | Enqueue one coherent proposal for review (202, `PendingChangeRecord`). |

A change set carries a `title` and a non-empty list of `upsert`/`delete` changes. Each change may carry
`expected_before`, the value the proposal was reviewed against; a mismatch is refused with 409. A delete is refused
while a canonical or live beta image model still names the baseline. The reference check is repeated when an
approved change is applied, so a model proposed after the baseline change cannot be stranded by it. Other validation
failures are 422.

Submissions require PRIMARY mode and an `apikey` header, and are applied through the shared pending queue like any
other reviewed change. Applying one registers the baseline in the running process, so the v2 model-create route
accepts models naming it without a package release.

A REPLICA fetches `GET /export` once when it first reads the store. `ModelReferenceManager.refresh_image_baselines()`
re-fetches it, which a long-lived replica needs in order to see a baseline published after it started.
