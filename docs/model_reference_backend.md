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
- `needs_refresh()` - Staleness detection
- `_mark_stale_impl()` - Backend-specific staleness marking
- `get_category_file_path()` - Return file path or None
- `get_all_category_file_paths()` - Return all file paths
- `get_legacy_json()` - Legacy format retrieval
- `get_legacy_json_string()` - Legacy format string retrieval

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

The notable exception would be backends that are themselves caching layers (e.g. RedisBackend).

See the [ReplicaBackendBase documentation](replica_backend_base.md) for details.

### 2. Honor force_refresh Parameter

Always respect the `force_refresh` parameter to bypass caches. See the [`fetch_category()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category] documentation for requirements.

### 3. Handle Errors Gracefully

Return `None` on errors, don't raise exceptions from fetch methods. This allows callers to handle missing data gracefully.

### 4. Use Async Properly

In async methods, use async I/O and concurrent operations with `asyncio.gather()`. See [`fetch_all_categories_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories_async] for implementation examples.

### 5. Implement Feature Detection

Always implement `supports_*()` methods before feature methods:

- [`supports_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_writes] before write operations
- [`supports_legacy_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_legacy_writes] before legacy operations
- [`supports_cache_warming()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_cache_warming] before cache warming
- [`supports_health_checks()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_health_checks] before health checks
- [`supports_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_statistics] before statistics

### 6. Document Your Backend

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

| Method | Purpose |
|--------|---------|
| [`fetch_category()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category] | Fetch single category data |
| [`fetch_all_categories()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories] | Fetch all categories data |
| [`fetch_category_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_category_async] | Async single category fetch |
| [`fetch_all_categories_async()`][horde_model_reference.backends.base.ModelReferenceBackend.fetch_all_categories_async] | Async all categories fetch |
| [`needs_refresh()`][horde_model_reference.backends.base.ModelReferenceBackend.needs_refresh] | Check if cached data is stale |
| [`_mark_stale_impl()`][horde_model_reference.backends.base.ModelReferenceBackend._mark_stale_impl] | Backend-specific staleness marking |
| [`get_category_file_path()`][horde_model_reference.backends.base.ModelReferenceBackend.get_category_file_path] | Get file path for category |
| [`get_all_category_file_paths()`][horde_model_reference.backends.base.ModelReferenceBackend.get_all_category_file_paths] | Get all file paths |
| [`get_legacy_json()`][horde_model_reference.backends.base.ModelReferenceBackend.get_legacy_json] | Get legacy format dict |
| [`get_legacy_json_string()`][horde_model_reference.backends.base.ModelReferenceBackend.get_legacy_json_string] | Get legacy format string |

### Optional Methods (Override If Needed)

| Feature | Detection Method | Implementation Methods |
|---------|------------------|----------------------|
| **Writes** | [`supports_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_writes] | [`update_model()`][horde_model_reference.backends.base.ModelReferenceBackend.update_model], [`delete_model()`][horde_model_reference.backends.base.ModelReferenceBackend.delete_model] |
| **Legacy Writes** | [`supports_legacy_writes()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_legacy_writes] | [`update_model_legacy()`][horde_model_reference.backends.base.ModelReferenceBackend.update_model_legacy], [`delete_model_legacy()`][horde_model_reference.backends.base.ModelReferenceBackend.delete_model_legacy] |
| **Cache Warming** | [`supports_cache_warming()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_cache_warming] | [`warm_cache()`][horde_model_reference.backends.base.ModelReferenceBackend.warm_cache], [`warm_cache_async()`][horde_model_reference.backends.base.ModelReferenceBackend.warm_cache_async] |
| **Health Checks** | [`supports_health_checks()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_health_checks] | [`health_check()`][horde_model_reference.backends.base.ModelReferenceBackend.health_check] |
| **Statistics** | [`supports_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.supports_statistics] | [`get_statistics()`][horde_model_reference.backends.base.ModelReferenceBackend.get_statistics] |

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
