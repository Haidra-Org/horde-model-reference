# Model Reference Backend

## Overview

`ModelReferenceBackend` is the abstract base class that defines the interface all backend implementations must fulfill. It establishes the contract for fetching, caching, and managing model reference data from various sources (GitHub, filesystem, HTTP APIs, databases, etc.).

**Key responsibilities:**

- Define the interface for data fetching (sync and async)
- Specify cache refresh semantics
- Provide hooks for optional features (writes, health checks, statistics)
- Enable pluggable backend architecture

**Implementations:**

- `ReplicaBackendBase` - Abstract base with caching infrastructure
    - `HTTPBackend` - Fetches from PRIMARY API with GitHub fallback
    - `GitHubBackend` - Downloads from GitHub repositories
    - `FileSystemBackend` - Reads/writes local filesystem files
    - `RedisBackend` - Uses Redis for distributed caching (PRIMARY mode)

## Implementation Checklist

See [`ModelReferenceBackend`][horde_model_reference.backends.base.ModelReferenceBackend] for full details.

When creating a new backend implementation:

### Required Implementations

- `__init__()` - Initialize with appropriate mode (call `super().__init__(mode)`)
- `fetch_category()` - Sync data fetching
- `fetch_all_categories()` - Sync batch fetching
- `fetch_category_async()` - Async data fetching
- `fetch_all_categories_async()` - Async batch fetching
- `needs_refresh()` - Staleness detection _(auto-provided by `ReplicaBackendBase`)_
- `_mark_stale_impl()` - Backend-specific staleness marking _(auto-provided by `ReplicaBackendBase`)_
- `get_category_file_path()` - Return file path or None
- `get_all_category_file_paths()` - Return all file paths
- `get_legacy_json()` - Legacy format retrieval
- `get_legacy_json_string()` - Legacy format string retrieval

> **Note:** `ModelReferenceBackend` declares `needs_refresh()` and `_mark_stale_impl()` as abstract, but `ReplicaBackendBase` supplies both implementations. If you subclass `ReplicaBackendBase` (the recommended model), you only need to implement the fetching and file-path/legacy retrieval methods listed above.

### Optional Implementations

- `supports_writes()` + `update_model()` + `delete_model()` - If backend supports v2 writes
- `update_model_from_base_model()` - Automatically provided if `supports_writes()` returns `True`
- `supports_legacy_writes()` + `update_model_legacy()` + `delete_model_legacy()` - If legacy writes needed
- `update_model_legacy_from_base_model()` - Automatically provided if `supports_legacy_writes()` returns `True`
- `supports_cache_warming()` + `warm_cache()` + `warm_cache_async()` - If cache warming supported
- `supports_health_checks()` + `health_check()` - If health monitoring needed
- `supports_statistics()` + `get_statistics()` - If statistics tracking desired

## Best Practices

### 1. Extend ReplicaBackendBase for Caching

Don't implement `ModelReferenceBackend` directly. Use [`ReplicaBackendBase`][horde_model_reference.backends.replica_backend_base.ReplicaBackendBase] which provides:

- TTL-based caching
- File mtime validation
- Thread-safe locks
- Cache helper methods
- `_fetch_with_cache()` to remove boilerplate around cache lookups

The notable exception would be backends that are themselves caching layers (e.g. RedisBackend).

See the [ReplicaBackendBase documentation](replica_backend_base.md) for details.

### 2. Use `_fetch_with_cache()` When Possible

If your backend simply needs to "return cached data unless forced to refetch, otherwise fetch and store", call `_fetch_with_cache(category, fetch_fn, force_refresh=...)`. Provide a callable that performs the actual fetch and returns the parsed payload (or `None`). The helper checks `_get_from_cache()`, executes the callable on cache miss, stores the result via `_store_in_cache()`, and returns it. Use the more explicit patterns (locks, download + load, etc.) only when you need additional coordination around the fetch flow.

### 3. Honor force_refresh Parameter

Always respect the `force_refresh` parameter to bypass caches. See the [`fetch_category()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category] documentation for requirements.

### 4. Handle Errors Gracefully

Return `None` on errors, don't raise exceptions from fetch methods. This allows callers to handle missing data gracefully.

### 5. Use Async Properly

In async methods, use async I/O and concurrent operations with `asyncio.gather()`. See [`fetch_all_categories_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories_async] for implementation examples.

### 6. Implement Feature Detection

Always implement `supports_*()` methods before feature methods:

- [`supports_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_writes] before write operations
- [`supports_legacy_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_legacy_writes] before legacy operations
- [`supports_cache_warming()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_cache_warming] before cache warming
- [`supports_health_checks()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_health_checks] before health checks
- [`supports_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_statistics] before statistics

### 7. Document Your Backend

Include clear docstrings explaining:

- What data source the backend uses
- What modes it supports (PRIMARY/REPLICA)
- What optional features it provides
- Any special configuration requirements

## Important Design Constraints

These constraints are validated by the test suite:

### 1. Force Refresh Parameter

All fetch methods must support `force_refresh` to bypass backend-level caching. When `True`, perform a fresh fetch regardless of cache state.

### 2. Redownload Parameter for Legacy Methods

Legacy JSON methods should support the `redownload` parameter (analogous to `force_refresh`).

### 3. Async and Sync Cache Consistency

If your backend caches data, ensure async and sync methods share the same cache. [`ReplicaBackendBase`][horde_model_reference.backends.replica_backend_base.ReplicaBackendBase] handles this automatically.

### 4. Error Handling

Methods should handle errors gracefully and return `None` rather than raising exceptions.

### 5. Refresh Semantics

The [`needs_refresh()`][horde_model_reference.backends.base.ModelReferenceBackend.needs_refresh] method should indicate when **existing cached data** has become stale, NOT when initial fetch is needed. See the method documentation for details.

### 6. Statistics Tracking (Optional)

If implementing [`supports_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_statistics], track meaningful metrics like fetch counts, cache hits, fallback usage, error counts, and response times.

## Audit Trail and Replay

The PRIMARY filesystem backend emits append-only JSONL audit events whenever a legacy record is created, updated, or deleted. Logs are written under `horde_model_reference_paths.audit_path` using the structure `audit/<domain>/<category>/audit-000001.jsonl`. Each line is a serialized [`AuditEvent`][horde_model_reference.audit.events.AuditEvent] that includes the operation, model name, logical Horde user id, and payload snapshot or delta.

### Inspecting Events

Use the new `scripts/audit_replay.py` helper to stream events without writing ad-hoc parsers:

```bash
python scripts/audit_replay.py image_generation --domain legacy --start-event-id 10 --end-event-id 20 --pretty
```

Flags allow filtering by domain, category, specific model names, event id ranges, or timestamp ranges. The default output mode prints JSON lines for each matching event; pass `--output state` to reconstruct the final state of the selected category using the embedded [`AuditTrailReader`][horde_model_reference.audit.reader.AuditTrailReader] and [`AuditReplayer`][horde_model_reference.audit.replay.AuditReplayer].

Example to rebuild the current state for a subset of models:

```bash
python scripts/audit_replay.py image_generation --output state -m my_model -m other_model --pretty
```

These utilities operate entirely on the JSONL segments and do not require the service to be running, making them suitable for offline investigations or recovery workflows. Configure audit behavior via the `HORDE_MODEL_REFERENCE_AUDIT__*` environment variables (e.g. `AUDIT__MAX_SEGMENT_BYTES`, `AUDIT__ROOT_PATH_OVERRIDE`), and see [Audit Trail Best Practices](audit_trail.md) for operational tips.

## Pending Queue Apply Workflow

PRIMARY deployments can gate all v2 writes through the pending queue to ensure multi-person review before model metadata is promoted. The queue keeps staged edits out of read APIs until an approver applies the change, and all audit trail writes continue to flow through `PendingQueueService` rather than the HTTP routers.

For an operator-focused playbook (storage layout, canonical format behavior, router entry points, and troubleshooting) see [Pending Queue Architecture](pending_queue.md).

### Deployment Constraints and Storage Isolation

- Enable the workflow by setting `HORDE_MODEL_REFERENCE_PENDING_QUEUE__ENABLED=true` while `HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY`. REPLICA nodes ignore the queue entirely and always treat v2 APIs as read-only.
- Queue persistence defaults to `<cache_home>/pending_queue`, but production deployments should configure `HORDE_MODEL_REFERENCE_PENDING_QUEUE__ROOT_PATH_OVERRIDE` (or adjust `...RELATIVE_SUBDIR`) so each deployment, environment, or test run has a dedicated directory. This mirrors the test fixture override that prevents cross-talk between suites.
- Pending queue files are distinct from audit trail logs. Never co-locate `pending_queue` data under the audit path; the audit JSONL stream remains the only canonical record of applied operations.

### Auth Lists and Workflow Roles

- Requestors submit batches via the write APIs once their Horde user id appears in `HORDE_MODEL_REFERENCE_PENDING_QUEUE__REQUESTOR_IDS`. Approvers must include the requestor IDs and are configured with `...APPROVER_IDS` so approval permissions are a superset of submission permissions.
- Provide these list settings as JSON arrays (e.g. `["user_a","user_b"]`) when using environment variables. Use `__` (double underscore) to separate nesting levels from the field name when setting nested model fields via environment variables.
- Because PRIMARY mode is the authoritative source, always double-check that queue approvers can reach the deployment that owns the filesystem backend; REPLICA nodes cannot apply or approve changes.

### HTTP Apply Workflow

- The `pending_queue` router registers before category routes and exposes `GET /pending_queue/changes`, `GET /pending_queue/changes/{id}`, `POST /pending_queue/batches`, `POST /pending_queue/changes/{id}/apply`, and `POST /pending_queue/apply`.
- Every endpoint enforces `authenticate_queue_approver`, `assert_v2_write_enabled`, and `require_pending_queue_service`, ensuring only PRIMARY deployments with pending-queue enabled and authorized users can mutate state.
- `POST /pending_queue/changes/{id}/apply` performs a single apply by delegating to `apply_pending_change()`, which validates approval status, writes through the filesystem backend, marks the record as applied, and allows the backend to call `mark_stale()` so caches refresh on the next read.
- `POST /pending_queue/apply` accepts `{ "change_ids": [...], "job_id": "..." }`, processes IDs sequentially via `apply_pending_changes()`, and stops on the first backend failure. The response reports `applied_change_ids`, `failed_change_ids`, and serialized records so operators can retry without guessing intermediate state.
- Router responses rely on `.model_dump(..., exclude_none=True)` to prevent accidental audit duplication. All audit log writes remain in `PendingQueueService`, which already emits JSONL events alongside standard backend operations.

### Operational Guardrails

- Pending queue data never feeds read APIs until a change transitions to `applied`. If you observe pending data leaking, verify that cache directories differ per deployment and that only PRIMARY mode has writes enabled.
- The pending queue is operated via HTTP endpoints only. On-call engineers should use the frontend UI or directly call the HTTP endpoints with the same payload the UI would send. Always include `job_id` so audit investigations can pair queue actions with user intent.

## Testing Your Backend

When implementing a new backend, ensure you test:

### Core Functionality Tests

1. **Fetch operations:**
    - `fetch_category()` returns correct data
    - `fetch_category()` returns `None` for unavailable categories
    - `fetch_all_categories()` returns dict with all categories
    - Async variants behave identically to sync variants

2. **Cache behavior:**
    - First fetch populates cache
    - Second fetch uses cached data (verify with call counters)
    - `force_refresh=True` bypasses cache
    - TTL expiration triggers refetch
    - `mark_stale()` invalidates cache

3. **Helper methods (if using ReplicaBackendBase):**
    - `has_cached_data()` returns `False` before first fetch, `True` after
    - `should_fetch_data()` returns `True` when cache is invalid or stale
    - `needs_refresh()` returns `False` for initial state, `True` for stale data

### Write Operations Tests (if supported)

1. **Update operations:**
    - `update_model()` creates new model
    - `update_model()` updates existing model
    - Cache is invalidated after update
    - Callbacks are notified after update

2. **Delete operations:**
    - `delete_model()` removes existing model
    - `delete_model()` raises `KeyError` for non-existent model
    - Cache is invalidated after delete
    - Callbacks are notified after delete

### Semantic Correctness Tests

Test the semantic distinction for `needs_refresh()`:

```python
def test_needs_refresh_semantics(backend):
    category = MODEL_REFERENCE_CATEGORY.image_generation

    # Initially: no cache, needs_refresh should be False
    assert not backend.has_cached_data(category)
    assert not backend.needs_refresh(category)

    # After storing: has cache, needs_refresh should be False (fresh)
    backend._store_in_cache(category, {"test": "data"})
    assert backend.has_cached_data(category)
    assert not backend.needs_refresh(category)

    # After marking stale: has cache, needs_refresh should be True
    backend.mark_stale(category)
    assert backend.has_cached_data(category)
    assert backend.needs_refresh(category)
```

See `tests/test_replica_backend_base.py`, `tests/test_http_backend.py`, and `tests/test_redis_backend.py` for comprehensive examples.

## Summary

### Abstract Methods (Must Implement)

All backends must implement these methods from [`ModelReferenceBackend`][horde_model_reference.backends.base.ModelReferenceBackend]:

| Method                                                                                                                   | Purpose                            |
| ------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| [`fetch_category()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category]                           | Fetch single category data         |
| [`fetch_all_categories()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories]               | Fetch all categories data          |
| [`fetch_category_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category_async]               | Async single category fetch        |
| [`fetch_all_categories_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories_async]   | Async all categories fetch         |
| [`needs_refresh()`][horde_model_reference.backends.base.ModelReferenceBackend.needs_refresh]                             | Check if cached data is stale      |
| [`_mark_stale_impl()`][horde_model_reference.backends.base.ModelReferenceBackend._mark_stale_impl]                       | Backend-specific staleness marking |
| [`get_category_file_path()`][horde_model_reference.backends.base.ModelReferenceBackend.get_category_file_path]           | Get file path for category         |
| [`get_all_category_file_paths()`][horde_model_reference.backends.base.ModelReferenceBackend.get_all_category_file_paths] | Get all file paths                 |
| [`get_legacy_json()`][horde_model_reference.backends.base.ModelReferenceBackend.get_legacy_json]                         | Get legacy format dict             |
| [`get_legacy_json_string()`][horde_model_reference.backends.base.ModelReferenceBackend.get_legacy_json_string]           | Get legacy format string           |

> Inheriting from `ReplicaBackendBase` satisfies `needs_refresh()` and `_mark_stale_impl()` automatically, leaving only the fetch/file-path methods for you to implement.

### Optional Methods (Override If Needed)

| Feature           | Detection Method                                                                                               | Implementation Methods                                                                                                                                                                                             |
| ----------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Writes**        | [`supports_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_writes]               | [`update_model()`][horde_model_reference.backends.base.ModelReferenceBackend.update_model], [`delete_model()`][horde_model_reference.backends.base.ModelReferenceBackend.delete_model]                             |
| **Legacy Writes** | [`supports_legacy_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_legacy_writes] | [`update_model_legacy()`][horde_model_reference.backends.base.ModelReferenceBackend.update_model_legacy], [`delete_model_legacy()`][horde_model_reference.backends.base.ModelReferenceBackend.delete_model_legacy] |
| **Cache Warming** | [`supports_cache_warming()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_cache_warming] | [`warm_cache()`][horde_model_reference.backends.base.ModelReferenceBackend.warm_cache], [`warm_cache_async()`][horde_model_reference.backends.base.ModelReferenceBackend.warm_cache_async]                         |
| **Health Checks** | [`supports_health_checks()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_health_checks] | [`health_check()`][horde_model_reference.backends.base.ModelReferenceBackend.health_check]                                                                                                                         |
| **Statistics**    | [`supports_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_statistics]       | [`get_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.get_statistics]                                                                                                                     |

### Recommended Approach

1. **Extend [`ReplicaBackendBase`][horde_model_reference.backends.replica_backend_base.ReplicaBackendBase]** instead of `ModelReferenceBackend` directly
2. **Implement required abstract methods** using caching helpers like [`_fetch_with_cache()`][horde_model_reference.backends.replica_backend_base.ReplicaBackendBase._fetch_with_cache]
3. **Override optional methods** only if needed
4. **Follow implementation patterns** from existing backends:
    - [`HTTPBackend`][horde_model_reference.backends.http_backend.HTTPBackend]
    - [`FileSystemBackend`][horde_model_reference.backends.filesystem_backend.FileSystemBackend]
    - [`GitHubBackend`][horde_model_reference.backends.github_backend.GitHubBackend]
    - [`RedisBackend`][horde_model_reference.backends.redis_backend.RedisBackend]

See the [ReplicaBackendBase documentation](replica_backend_base.md) for details on the caching infrastructure.
