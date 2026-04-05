# Audit Trail Best Practices

The audit trail captures every legacy CRUD mutation performed by the PRIMARY deployment. Events are streamed to disk as compact JSONL so downstream tooling (e.g. `scripts/audit_replay.py`) can reconstruct historical state.

This document collects best practices that keep the system maintainable, reduce friction for on-call engineers, and clarify operational expectations.

## Architecture Recap

- `AuditTrailWriter` is instantiated once by `ModelReferenceManager` when the backend supports writes. It persists events under `horde_model_reference_paths.audit_path` using the directory layout `audit/<domain>/<category>/audit-000001.jsonl`.
- Each event receives a monotonically increasing integer `event_id` recorded in `audit/index.json`. Writes acquire an in-process `RLock` and complete in O(1) time, so they must never block CRUD submissions.
- Rotation is size-based (default 5 MiB segments). Consumers should not rely on wall-clock boundaries; always treat segments as append-only logs.
- `AuditTrailReader` streams events lazily with filters covering domain, category, model names, event id and timestamp ranges.
- `AuditReplayer` composes reader output to rebuild effective category state, which powers the `scripts/audit_replay.py --output state` command.

## Audit Event Categories

The audit trail records two distinct categories of events:

### Model Metadata Events

Recorded by `FileSystemBackend` for all model CRUD operations:

- **Category**: Model category (e.g., `image_generation`, `text_generation`, `controlnet`)
- **Operations**: `CREATE`, `UPDATE`, `DELETE`
- **Payload**: Snapshot or delta of model metadata changes
- **Purpose**: Authoritative history of model data; enables state reconstruction via replay

### Pending Queue Lifecycle Events

Recorded by `PendingQueueService` when the pending queue is enabled:

- **Category**: `pending_queue`
- **Operations**: Always `UPDATE` (lifecycle transitions)
- **Actions**: `enqueue`, `approve`, `reject`, `apply`
- **Purpose**: Tracks approval workflow; enables queue state reconstruction
- **Model Name**: Change ID (stringified)

**See also**: [Pending Queue Architecture](pending_queue.md) for detailed coverage of dual audit logging design and how queue events interact with model events.

## Configuration

Set the following environment variables (all prefixed with `HORDE_MODEL_REFERENCE_`) to tailor audit storage and rotation:

| Variable | Description | Default |
| --- | --- | --- |
| `AUDIT_ENABLED` | Toggle audit writing entirely (PRIMARY mode only). | `true` |
| `AUDIT_MAX_SEGMENT_BYTES` | Maximum JSONL segment size before rotation. | `5 MiB` |
| `AUDIT_RELATIVE_SUBDIR` | Folder name under the cache home for audit logs. | `audit` |
| `AUDIT_ROOT_PATH_OVERRIDE` | Absolute path to store audit logs (bypasses relative subdir). | _unset_ |

Example: `HORDE_MODEL_REFERENCE_AUDIT__MAX_SEGMENT_BYTES=1048576` rotates each megabyte, while `HORDE_MODEL_REFERENCE_AUDIT__ROOT_PATH_OVERRIDE=/var/log/horde-audit` stores logs outside the cache root.

## Writing Events

1. **Single-writer discipline**: Only the PRIMARY backend process should append to audit logs. Redis-wrapped deployments continue to funnel all writes through the `FileSystemBackend`, so no extra work is required as long as the cache cluster does not perform writes itself.
2. **Propagate request context**: Always provide `logical_user_id` (immutable Horde user id) and reuse `request_id` for idempotency/debug correlation. If a new code path performs a write, ensure it forwards these values so events remain attributable.
3. **Payload accuracy**: Prefer `AuditPayload.from_create` / `.from_delete` / `.from_update` helpers. Avoid storing oversized blobs (e.g., binary files); stick to JSON-serializable dictionaries to keep replay deterministic.
4. **Error isolation**: Audit failures must never block CRUD paths. The backend already wraps `_append_legacy_audit_event` in a `try/except` that logs issues and continues. Maintain this pattern for any future emitters.

## Operating the Logs

- **Disk management**: The writer never truncates old segments. Operators should rely on log rotation tooling (e.g., compress and ship files older than _n_ days). Because segments are sequentially numbered, it is safe to archive whole files once they predate the desired retention window.
- **Integrity checks**: The replay CLI can spot malformed lines using `AuditTrailReader`'s validation. Periodically run `python scripts/audit_replay.py <category> --output events --pretty` and confirm there are no warnings in stdout/stderr.
- **Reconstructing state**: To verify that log replay matches the current JSON source of truth, compare `state` output with on-disk category files:

  ```bash
  python scripts/audit_replay.py image_generation --output state --pretty > /tmp/replayed.json
  diff -u <(jq -S . /tmp/replayed.json) <(jq -S . /path/to/legacy/image_generation.json)
  ```

- **Selective investigations**: Filter to one model or range of event ids to answer "who changed this" questions quickly:

  ```bash
  python scripts/audit_replay.py image_generation -m my_model --start-event-id 4500 --pretty
  ```

## Maintenance Guidance

- **Configuration knobs**: If deployments need larger or smaller segment sizes, adjust `DEFAULT_MAX_FILE_SIZE_BYTES` in `audit/writer.py` (or make it configurable via settings for multi-environment control). Keep the size under log shipping limits to avoid back-pressure.
- **Schema evolution**: When adding new fields to `AuditEvent`, prefer optional additions so older segments stay valid. Update `AuditTrailReader` and replay tests to cover new behavior.
- **Testing**: `tests/test_audit_trail.py` verifies the writer and FileSystem backend integration, while `tests/test_audit_replay.py` exercises reader filters and replay correctness. Extend these suites when modifying payload logic or adding new CLI modes.
- **Docs & onboarding**: Link this document from backend-focused guides so contributors learn how to add new audit emitters without accidental regressions.

## Known Friction Points & Mitigations

| Area | Friction | Suggested Mitigation |
| --- | --- | --- |
| Disk permissions | Audit root inherits the cache directory ownership, which can differ between local dev and containers. | Ensure `CACHE_HOME` is writable before starting PRIMARY workers; the writer will create missing directories but cannot fix permissions. |
| Large replays | Reading multiple gigabytes of logs via the CLI can take time. | Narrow the query using `--start-event-id/--end-event-id` or per-model filters, and pipe through `jq` or `rg` for incremental inspection. |
| Multi-process writers | Only a single process updates `audit/index.json`. Multiple PRIMARY writers would clobber event ids. | Deploy one write-capable API instance per shared storage location or switch to an external append-only store if true multi-writer support is required. |
| Retention | Repository lacks automated pruning. | Schedule OS-level jobs (systemd timer, cron, or logrotate) to archive/compress segments and delete files beyond policy. Document the schedule in ops runbooks. |

By following the practices above, the audit trail remains trustworthy, replayable, and easy to reason about when debugging production incidents.
