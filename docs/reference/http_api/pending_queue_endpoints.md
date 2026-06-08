# Pending Queue Endpoints

Every write in the model-reference API is a **proposal** that flows through a review queue before it
touches the live dataset: **propose -> approve -> apply**. These endpoints drive that workflow. For
the conceptual model (batch-ID semantics, partial application, dual audit logging) see the
[Pending Queue](../pending_queue.md) reference; for a worked example see
[Submit Models via the API](../../guides/submit_models_via_the_api.md).

The queue is mounted under **both** API versions; the active one matches the deployment's
`canonical_format`:

- v2: `/api/model_references/v2/pending_queue/...`
- v1: `/api/model_references/v1/pending_queue/...`

Paths below use the v2 prefix and are relative to `https://models.aihorde.net/api`. **All
pending-queue endpoints require an `apikey`** (approver, unless noted) and a PRIMARY deployment;
otherwise they return `401`/`503`.

## Inspect the queue

| Method & path | Description |
| ------------- | ----------- |
| `GET /pending_queue/my_changes` | **(requestor)** List *your own* submitted changes, scoped to the calling key's user id. Same shape as `/changes` (`PendingQueuePage`) with `statuses`, `categories`, `offset`, `limit` filters, but does **not** require the approver role - this is how a requestor tracks a proposal's `status` after the `202`. |
| `GET /pending_queue/changes` | **(approver)** List all queued changes. Filters: `statuses`, `categories`, `batch_id` (≥1), `model_name`, `requested_by`, `offset` (≥0), `limit` (1–500, default 50). Returns a `PendingQueuePage`. |
| `GET /pending_queue/changes/{change_id}` | **(approver)** One `PendingChangeRecord`. `404` if missing. |
| `GET /pending_queue/changes/{change_id}/diff` | `PendingChangeDiff` against current state (field-level). |
| `GET /pending_queue/changes/diff?change_ids=...` | Diffs for many changes (1–100). Returns `PendingChangeDiffPage`. |

(The `…/pending_queue/...` prefix above is short for `/model_references/v2/pending_queue/...`.)

## Approve, reject, purge

| Method & path | Description |
| ------------- | ----------- |
| `POST /pending_queue/batches` | Approve and/or reject changes as a titled batch. Body: `PendingBatchRequest` (`batch_title` 1–120 chars, `approved_ids`, `rejected_ids`, `reject_reason` ≤500 - required when rejecting). Returns `PendingBatchResult`. |
| `POST /pending_queue/purge` | Remove queued changes matching filters. Body: `PurgePendingChangesRequest`. Returns `PurgePendingChangesResponse`. |

## Apply approved changes

Applying writes the change to the live dataset and invalidates caches.

| Method & path | Description |
| ------------- | ----------- |
| `POST /pending_queue/changes/{change_id}/apply` | Apply one approved change. Body: `ApplyPendingChangeRequest`. Returns `ApplySingleChangeResponse` (may include batch-split info). |
| `POST /pending_queue/apply?change_ids=...` | Apply several approved changes. Params: `change_ids` (≥1), `job_id` (≤120), `allow_mixed_batch` (default false). Returns `ApplyPendingChangesResponse`. |
| `POST /pending_queue/apply_batch/{batch_id}?job_id=...` | Apply all approved changes in a batch. Returns `ApplyPendingChangesResponse`. |

`job_id` (optional on every apply variant) is a caller-supplied **reservation token**: it is recorded
on the change to make the apply idempotent and to stop two workers applying the same change at once.
Omit it and the server generates one; reuse the same value to safely retry an apply.

Applying part of a batch reassigns the remaining approved changes to a new `batch_id` (a *batch
split*); the response reports the split so a UI can follow the remainder.

## Audit views

Read-only history under `/pending_queue/audit`.

| Method & path | Description |
| ------------- | ----------- |
| `GET /pending_queue/audit/current` | Currently pending (unapproved) changes. Param: `domain_override`. Returns `PendingQueueAuditCurrentResponse`. |
| `GET /pending_queue/audit/batches` | Historical batches. Params: `cursor` (≥1, return older than this batch id), `limit` (1–50, default 10), `domain_override`. Returns `PendingQueueAuditBatchPage`. |
| `GET /pending_queue/audit/batches/{batch_id}` | One batch's detail. `404` if missing. |
| `GET /pending_queue/audit/batches/{batch_id}/net_changes` | Net effect of all changes in a batch (cached ~5 min). Returns `BatchNetChangeResponse`. |

## Status & error notes

| Code | When |
| ---- | ---- |
| `200` | Read / approve / apply succeeded. |
| `202` | (From the write endpoints that *feed* the queue.) Change accepted and queued. |
| `400` | Invalid request (e.g. rejecting without a reason, empty id list). |
| `401` | Missing/invalid `apikey`, or the caller lacks the approver role. |
| `404` | Change or batch not found. |
| `503` | REPLICA mode, or the canonical format does not expose queue writes. |

## See also

- [Pending Queue](../pending_queue.md) - concepts, batch semantics, audit design.
- [Audit Trail](../audit_trail.md) - the append-only operation log the queue writes to.
- [v2 write endpoints](v2_endpoints.md#write-operations) - the create/update/delete calls that enqueue.
