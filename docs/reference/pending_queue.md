# Pending Queue Architecture

## Purpose

The pending queue lets PRIMARY deployments gate write operations behind a two-person workflow. Instead of mutating the filesystem backend immediately, POST/PUT/DELETE requests are serialized into `PendingChangeRecord` objects that stay isolated from read APIs until an approver applies them. This keeps partially reviewed changes out of production payloads and gives operators a single place to audit model edits.

Key properties:

- Works for both canonical formats by routing writes through the authoritative API version (v2 by default, v1 when `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=LEGACY`).
- Persists staged changes under a dedicated directory so test runs, staging, and production never share queue data.
- Emits all audit trail events inside `PendingQueueService`, ensuring HTTP routers never double-log.

## Canonical Format Interaction

The environment variable `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT` determines which API version can write:

- `canonical_format = "v2"` (default): `/model_references/v2` POST/PUT/DELETE endpoints enqueue changes and `/model_references/v2/pending_queue/*` routes allow operators to inspect, approve, and apply them. The legacy `/v1` API becomes read-only.
- `canonical_format = "LEGACY"`: `/model_references/v1` CRUD routes switch to queue-first semantics while `/model_references/v2` is read-only. Applying a change calls `FileSystemBackend.update_model_legacy`/`delete_model_legacy` so legacy JSON artifacts stay authoritative.

Changing canonical format at runtime is strongly discouraged if pending entries exist. Each queue record stores the payload produced by whichever API enqueued it, so applying to the wrong canonical backend can fail validation.

## Router Registration

FastAPI exposes identical queue endpoints under both API versions so operators have a predictable surface area:

| Prefix                               | Purpose               | Notes                                                                                |
| ------------------------------------ | --------------------- | ------------------------------------------------------------------------------------ |
| `/model_references/v2/pending_queue` | V2 canonical mode     | Enabled when PRIMARY backend supports v2 writes and the queue service is configured. |
| `/model_references/v1/pending_queue` | Legacy canonical mode | Enabled when legacy writes are canonical (PRIMARY + `canonical_format="LEGACY"`).    |

Routers are included before category routes (`/{model_category_name}`) to avoid 422 collisions. Each endpoint enforces:

1. `authenticate_queue_approver` – Horde API key must belong to an approver.
2. `assert_canonical_write_enabled` – ensures PRIMARY mode and canonical format match the router’s API.
3. `require_pending_queue_service` – guarantees the queue is configured and storage is reachable.

## Configuration Checklist

> File paths follow `src/horde_model_reference/__init__.py` settings unless overridden.

| Setting                                                   | Description                                                                                                                        |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY`            | Queue is ignored in REPLICA mode.                                                                                                  |
| `HORDE_MODEL_REFERENCE_PENDING_QUEUE__ENABLED=true`       | Enables `PendingQueueService` construction.                                                                                        |
| `HORDE_MODEL_REFERENCE_PENDING_QUEUE__REQUESTOR_IDS`      | Allow-list of Horde user ids that can submit changes (JSON array).                                                                 |
| `HORDE_MODEL_REFERENCE_PENDING_QUEUE__APPROVER_IDS`       | Allow-list of ids that can approve/apply changes. Should superset requestors.                                                      |
| `HORDE_MODEL_REFERENCE_PENDING_QUEUE__ROOT_PATH_OVERRIDE` | Optional absolute path for queue storage. Defaults to `<cache_home>/pending_queue`. Required when multiple deployments share disk. |
| `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT`                  | Determines which API is writable (`v2` or `legacy`).                                                                               |

> **Development fallback:** If either allow-list is left empty, the service automatically falls back to the built-in `allowed_users` list defined in `src/horde_model_reference/service/shared.py`. This keeps local environments usable with the default approver IDs (`["1", "6572"]`) but production deployments **must** set `HORDE_MODEL_REFERENCE_PENDING_QUEUE__REQUESTOR_IDS` and `HORDE_MODEL_REFERENCE_PENDING_QUEUE__APPROVER_IDS` explicitly. Removing the fallback IDs or leaving them unset in production will cause every queue request to be rejected with `401 Invalid API key`.

Recommended production layout:

```ini
HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=v2
HORDE_MODEL_REFERENCE_PENDING_QUEUE__ENABLED=true
HORDE_MODEL_REFERENCE_PENDING_QUEUE__REQUESTOR_IDS=["12345","67890"]
HORDE_MODEL_REFERENCE_PENDING_QUEUE__APPROVER_IDS=["12345","67890","54321"]
HORDE_MODEL_REFERENCE_PENDING_QUEUE__ROOT_PATH_OVERRIDE=/var/lib/horde/pending_queue
```

### Storage Isolation

- Never reuse queue directories across environments. Tests override `pending_queue.root_path_override` (see `tests/conftest.py`) to keep fixtures isolated; replicate this pattern for staging vs production.
- Queue files are independent of audit trail segments. Keep both under separate directories to avoid mixing partially reviewed data with immutable logs.

## Dual Audit Logging Design

The pending queue system produces **two categories of audit events** that serve different purposes:

### 1. Queue Lifecycle Events (Category: `pending_queue`)

`PendingQueueService` writes audit events for every queue state transition:

| Action        | When                                | Audit Payload                                          |
| ------------- | ----------------------------------- | ------------------------------------------------------ |
| `enqueue`     | Change submitted                    | Request metadata, user ID, payload                     |
| `approve`     | Batch approved                      | Batch ID, approver ID, change IDs                      |
| `reject`      | Batch rejected                      | Batch ID, approver ID, reason, change IDs              |
| `apply`       | Change written to backend           | Change ID, batch ID, job ID                            |
| `batch_split` | Partial apply triggers reassignment | Original batch ID, new batch ID, reassigned change IDs |

**Purpose**: Tracks the approval workflow lifecycle. Enables reconstruction of queue state from audit logs alone via `PendingQueueAuditReader`.

**Domain**: Matches the canonical format (`LEGACY` or `V2`)
**Category**: Always `pending_queue`
**Model Name**: Change ID (stringified)
**Operation**: Always `UPDATE` (lifecycle transition)

## Batch ID Semantics

Batch IDs group approved changes together for coordinated application. The system maintains the following invariants:

### Single Open Batch Rule

At any point in time, there is **at most one "open" batch** - the batch containing all APPROVED (but not yet applied) changes. This ensures:

1. **Approval consolidation**: When an approver approves new changes, they join the existing open batch rather than creating a new one.
2. **Predictable batch IDs**: The batch ID for pending approvals equals `last_fully_applied_batch_id + 1`.
3. **Clear audit trail**: Each batch represents a cohesive set of changes approved together.

### Batch Lifecycle

| Event                          | Batch ID Behavior                                     |
| ------------------------------ | ----------------------------------------------------- |
| First approval (no open batch) | New batch ID allocated (`last_batch_id + 1`)          |
| Subsequent approvals           | Join existing open batch (same batch ID)              |
| Full batch apply               | Batch closes; next approval creates new batch         |
| **Partial batch apply**        | Remaining APPROVED changes reassigned to new batch ID |

### Partial Application and Batch Splits

When a batch is **partially applied** (some changes applied, others still APPROVED):

1. The applied changes retain their original batch ID with status `APPLIED`.
2. The remaining APPROVED changes are **reassigned to a new batch ID**.
3. A `batch_split` audit event is emitted recording the reassignment.
4. The new batch becomes the "open" batch for future approvals.

This ensures the partially-applied batch is "closed" and won't receive new approvals, maintaining batch integrity.

### Example Scenario

```text
1. Approve changes A, B, C -> All get batch_id=1
2. Approve change D       -> D gets batch_id=1 (joins existing batch)
3. Apply change A         -> A is now APPLIED, B/C/D still APPROVED
   └─ Partial apply detected: B, C, D reassigned to batch_id=2
4. Approve change E       -> E gets batch_id=2 (joins current open batch)
5. Apply all (B, C, D, E) -> Batch 2 fully applied
6. Approve change F       -> F gets batch_id=3 (new batch, none open)
```

### Implementation Details

- `PendingQueueStore.get_or_create_pending_batch_id()`: Returns existing open batch ID or allocates new one.
- `PendingQueueStore.get_approved_changes_in_batch(batch_id)`: Finds remaining APPROVED changes after partial apply.
- `PendingQueueService._handle_partial_batch_apply()`: Reassigns remaining changes and emits `batch_split` event.

### 2. Model Metadata Events (Category: model category)

When a pending change is **applied**, `FileSystemBackend.update_model()`/`delete_model()` automatically writes a separate audit event:

**Purpose**: Records the actual mutation to model metadata, independent of approval workflow.
**Category**: Target model category (e.g., `image_generation`, `text_generation`)
**Model Name**: The model being changed
**Operation**: `CREATE`, `UPDATE`, or `DELETE`
**Payload**: Snapshot or delta of model changes

### Why Two Categories?

- **Queue events** let operators audit who approved what and when, enabling workflow accountability.
- **Model events** preserve the authoritative history of model metadata changes, enabling state reconstruction via `scripts/audit_replay.py`.
- **Independence**: Queue state can be rebuilt from `pending_queue` events; model history can be rebuilt from category events. Neither requires the other for replay.

**Critical**: Audit logging happens exclusively within `PendingQueueService` and `FileSystemBackend`. HTTP routers never emit audit events directly, preventing double-logging.

## Request Lifecycle

1. **Requestor submits change** via `/model_references/vX/...` POST/PUT/DELETE. The router:
    - Authenticates the Horde API key against the requestor allow-list.
    - Validates create/update/delete constraints.
    - Calls `PendingQueueService.enqueue_change`, storing metadata such as `request_metadata.route` and `payload`.
    - Emits audit event: `action="enqueue"`, `category="pending_queue"`.
    - Returns HTTP 202 with the serialized `PendingChangeRecord`.
2. **Approver reviews queue** using `/pending_queue/changes`, `/changes/{id}`, and `/batches`. Batch requests accept `approved_ids`, `rejected_ids`, plus optional `reject_reason`.
    - Emits audit event: `action="approve"|"reject"`, `category="pending_queue"`.
3. **Apply operation** (automation or operator) calls `POST /pending_queue/changes/{id}/apply` for single changes or `POST /pending_queue/apply` with `{ "change_ids": [...], "job_id": "..." }` for ordered bulk operations. Application stops on first backend error and reports the failure in-line.
4. **Backend write + cache invalidation** happen inside `pending_queue/apply.py`. For v2 canonical deployments, the helper calls `backend.update_model`/`delete_model`. For legacy canonical deployments, it calls `backend.update_model_legacy`/`delete_model_legacy`. In both cases:
    - Emits audit event: `action="apply"`, `category="pending_queue"`.
    - Backend emits audit event: `category=<model_category>`, `operation=CREATE|UPDATE|DELETE`.
    - Filesystem backend triggers `mark_stale()` so cached JSON reloads on the next request.

## Authentication & Authorization Flow

- `authenticate_queue_requestor` and `authenticate_queue_approver` (in `src/horde_model_reference/service/shared.py`) call the AI Horde API (`v2/find_user`) and match user ids against the configured allow-lists. Requestors inherit approver access so they can promote their own changes if desired.
- The legacy v1 CRUD routers use the same helpers once canonical format switches to `"LEGACY"`, eliminating the bespoke `allowed_users` list.
- All queue endpoints return HTTP 401 when the header is missing/invalid, 503 when pending queue is disabled, and 400/404 for validation and existence errors.

## Operational Guidance

- **Job IDs:** Always supply a meaningful `job_id` (automation run id, incident ticket, etc.) when applying changes. This value is recorded in the queue record and audit payload for later correlation.
- **Monitoring:** Watch for unexpected growth of `pending_queue` files. A large backlog often means approvals are stalled or automation failed mid-apply; use the bulk apply endpoint to resume from the reported `failed_change_id`.
- **Mode switching:** Before changing `canonical_format`, ensure the pending queue is empty and cache directories are clean. Mixing payload styles can produce backend validation errors.
- **Disaster recovery:** If an apply job fails after writing to disk but before `mark_applied`, operators can manually verify the filesystem state and re-run the endpoint. The helper is idempotent regarding backend writes (`update_model` is an upsert).
- **Tooling:** The pending queue is operated exclusively via HTTP endpoints. Use the frontend UI or direct API calls with appropriate authorization headers. No separate CLI tools are provided for queue operations.

## File References

| Area                   | Files                                                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Settings & paths       | `src/horde_model_reference/__init__.py`, `src/horde_model_reference/path_consts.py`                                                                                                          |
| Pending queue service  | `src/horde_model_reference/pending_queue/{models.py,service.py,apply.py}`                                                                                                                    |
| Router logic           | `src/horde_model_reference/service/v1/routers/create_update.py`, `src/horde_model_reference/service/v2/routers/{references,pending_queue}.py`, `src/horde_model_reference/service/shared.py` |
| Tests                  | `tests/service/test_v2_api.py`, `tests/pending_queue/test_service.py`, `tests/pending_queue/test_apply.py`, fixtures in `tests/conftest.py`                                                  |
| Docs referencing queue | `docs/reference/model_reference_backend.md`, `docs/reference/primary_deployments.md`                                                                                                         |

## Related Documentation

- [Model Reference Backend](model_reference_backend.md)
- [Primary Deployment Guide](primary_deployments.md)
- [Audit Trail Documentation](audit_trail.md)

```text
Fast reference for on-call engineers:
1. Check `pending_queue/` directory size.
2. Use GET `/model_references/vX/pending_queue/changes?statuses=PENDING` to inspect backlog.
3. Approve via POST `/pending_queue/batches`.
4. Apply sequentially with POST `/pending_queue/apply` (provide `job_id`).
```
