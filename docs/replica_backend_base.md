# Replica Backend Base

## Overview

`ReplicaBackendBase` extends `ModelReferenceBackend` to provide a comprehensive caching layer for backend implementations. Despite its name, it provides caching infrastructure for both REPLICA and PRIMARY mode backends.

**What it provides:**

- TTL-based cache expiration
- File mtime validation for cache invalidation
- Dual cache system (v2/converted format + legacy format)
- Thread-safe locking (sync and async)
- Extensible validation hooks
- Helper methods for common caching patterns

**What you still implement:**

`ReplicaBackendBase` is still an abstract class. You must implement the abstract methods from `ModelReferenceBackend`:

- `fetch_category()` - Your data fetching logic
- `fetch_all_categories()` - Batch fetching logic
- `fetch_category_async()` - Async fetching logic
- `fetch_all_categories_async()` - Async batch fetching
- `get_category_file_path()` - Return file paths (if applicable)
- `get_all_category_file_paths()` - Return all file paths
- `get_legacy_json()` - Legacy format retrieval
- `get_legacy_json_string()` - Legacy format string retrieval

The caching infrastructure helps you implement these methods efficiently.

## Implementation Patterns for Abstract Methods

These patterns show how to implement the required abstract methods from `ModelReferenceBackend` using the caching infrastructure provided by `ReplicaBackendBase`.

### Pattern 1: Direct Fetch and Cache (HTTPBackend)

For backends that fetch data directly without storing to disk:

```python
def fetch_category(
    self,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Fetch from PRIMARY API with fallback."""
    # Check if we need to fetch
    if force_refresh or self.should_fetch_data(category):
        # Fetch data directly
        data = self._fetch_from_primary(category)
        
        # Fallback if needed
        if data is None and self._enable_github_fallback:
            data = self._github_backend.fetch_category(category, force_refresh=force_refresh)
        
        # Store directly in cache
        if data is not None:
            self._store_in_cache(category, data)
        
        return data
    
    # Return cached data
    return self._get_from_cache(category)
```

**Use when:** Your backend fetches data via network/API and doesn't need local file storage.

**Key points:**

- Check `should_fetch_data()` to determine if fetch is needed
- Fetch data using your backend-specific method
- Call `_store_in_cache()` directly with the fetched data
- Return cached data via `_get_from_cache()`

### Pattern 2: Read from Disk (FileSystemBackend)

For backends that read existing files from disk:

```python
def fetch_category(
    self,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Fetch model reference data from filesystem."""
    with self._lock:
        # Check if we need to fetch
        if force_refresh or self.should_fetch_data(category):
            file_path = self._get_file_path(category)
            
            if not file_path or not file_path.exists():
                self._store_in_cache(category, None)
                return None
            
            try:
                with open(file_path, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                
                self._store_in_cache(category, data)
                return data
            
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                self._invalidate_cache(category)
                return None
        
        # Return cached data
        return self._get_from_cache(category)
```

**Use when:** Your backend reads from local files that may be modified externally.

**Key points:**

- Lock with `self._lock` for file operations
- Check `should_fetch_data()` to determine if fetch is needed
- Handle missing files by caching `None`
- On read errors, call `_invalidate_cache()` to force retry next time
- Store successfully read data via `_store_in_cache()`
- Return cached data via `_get_from_cache()`

### Pattern 3: Download, Store to Disk, Load (GitHubBackend)

For backends that download data to disk then load it:

```python
def fetch_category(
    self,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    force_refresh: bool = False,
) -> dict[str, Any] | None:
    """Fetch model reference data for a specific category."""
    with self._lock:
        # Check if we need to fetch
        if force_refresh or self.should_fetch_data(category):
            # Download to disk
            self._download_and_convert_single(category, overwrite_existing=force_refresh)
            # Load from disk and cache (calls _store_in_cache internally)
            return self._load_converted_from_disk(category)
        
        # Return cached data
        return self._get_from_cache(category)
```

**Use when:** Your backend downloads files to disk for persistence or processing.

**Key points:**

- Lock with `self._lock` for file operations
- Check `should_fetch_data()` to determine if fetch is needed
- Download/write to disk using your backend-specific method
- Load from disk with a helper that calls `_store_in_cache()`
- Return cached data via `_get_from_cache()`

## ReplicaBackendBase-Specific Features

`ReplicaBackendBase` provides concrete implementations and infrastructure that is NOT part of `ModelReferenceBackend`:

### Caching Infrastructure Methods (Provided by ReplicaBackendBase)

These are **protected methods** for use in your subclass implementations:

- `should_fetch_data(category)` - Check if data needs fetching (initial or refresh)
- `_get_from_cache(category)` - Retrieve cached data if valid
- `_store_in_cache(category, data)` - Store fetched data in cache
- `_invalidate_cache(category)` - Mark cache as invalid
- `_get_legacy_from_cache(category)` - Retrieve cached legacy data
- `_store_legacy_in_cache(category, dict, string)` - Store legacy data in cache
- `_invalidate_legacy_cache(category)` - Mark legacy cache as invalid
- `has_cached_data(category)` - Simple existence check (no validation)
- `is_cache_valid(category)` - Check if cache exists and is valid
- `is_legacy_cache_valid(category)` - Check if legacy cache is valid

### Properties and State (Provided by ReplicaBackendBase)

- `lock` (RLock) - Thread-safe lock for synchronous operations
- `async_lock` (AsyncLock) - Thread-safe lock for async operations
- `cache_ttl_seconds` - Get/set cache TTL
- `_cache` - Internal cache storage (do not access directly)
- `_category_timestamps` - Internal timestamp tracking
- `_stale_categories` - Internal staleness tracking

### Abstract Method Implementations (Provided by ReplicaBackendBase)

These methods from `ModelReferenceBackend` have concrete implementations:

- `needs_refresh(category)` - Checks if cached data is stale
- `mark_stale(category)` - Marks category as requiring refresh

### Hooks You Can Override (Optional)

- `_get_file_path_for_validation(category)` - Return file path for mtime tracking
- `_get_legacy_file_path_for_validation(category)` - Return legacy file path for mtime tracking
- `_additional_cache_validation(category)` - Add custom validation logic

## Caching Architecture

### Dual Cache System

The backend maintains **two independent cache systems** to handle different data formats:

**V2/Converted Format Cache** - Primary cache used by `fetch_category()`:

- `_cache` - Stores converted data
- `_category_timestamps` - Records last update time
- `_last_known_mtimes` - Tracks file modification times
- `_stale_categories` - Tracks explicitly invalidated categories

**Legacy Format Cache** - Used by `get_legacy_json()` and `get_legacy_json_string()`:

- `_legacy_json_cache` / `_legacy_json_string_cache` - Stores legacy data
- `_legacy_cache_timestamps` - Records last update time
- `_legacy_last_known_mtimes` - Tracks legacy file modification times
- `_stale_legacy_categories` - Tracks explicitly invalidated legacy categories

Both caches operate independently with the same validation logic (TTL, mtime, staleness).

## Cache Validation Logic

The `should_fetch_data(category)` method determines if data needs fetching by checking:

1. **Cache existence** - Is there any cached data?
2. **TTL expiration** - Has the cache expired? (if `cache_ttl_seconds` set)
3. **File mtime** - Has the file been modified? (if file path hook implemented)
4. **Staleness flag** - Was it explicitly invalidated via `mark_stale()`?

If any check fails, `should_fetch_data()` returns `True`, indicating data should be fetched. This unified check handles both initial loads and refreshes automatically.

## Cache Helper Methods Reference

These are the concrete methods provided by `ReplicaBackendBase` to help you implement the abstract methods from `ModelReferenceBackend`.

### `should_fetch_data(category)` ⭐ **Primary Helper**

The primary method for determining whether to fetch data. Use this when implementing `fetch_category()`.

```python
# ✅ GitHubBackend pattern
if force_refresh or self.should_fetch_data(category):
    self._download_and_convert_single(category, overwrite_existing=force_refresh)
    return self._load_converted_from_disk(category)

return self._get_from_cache(category)
```

Returns `True` if:

- No cached data exists (initial fetch needed), OR
- Cached data exists but is stale (refresh needed)

This combines initial fetch detection and refresh detection into a single check.

### `_get_from_cache(category)` ⭐ **Cache Retrieval Helper**

Protected method that retrieves cached data if valid, handling all validation internally.

```python
# ✅ GitHubBackend pattern
cached_data = self._get_from_cache(category)
if cached_data is not None:
    return cached_data
```

Returns:

- The cached dict if data exists and is valid
- `None` if no data exists or cache is invalid

### `_store_in_cache(category, data)` ⭐ **Cache Storage Helper**

Protected method that stores fetched data and updates all cache metadata (timestamps, mtimes, staleness flags).

```python
# ✅ GitHubBackend pattern (inside _load_converted_from_disk)
with open(file_path) as f:
    data: dict[str, Any] = ujson.load(f)

self._store_in_cache(category, data)
return data
```

Automatically handles:

- Storing data in `_cache`
- Recording current timestamp
- Clearing staleness flags
- Updating file mtime tracking (via `_get_file_path_for_validation()` hook)

### `mark_stale(category)` - Concrete Implementation

Public API provided by `ReplicaBackendBase` that implements the abstract method from `ModelReferenceBackend`. Invalidates cached data and can be called externally.

```python
# Example: File changed externally
backend.mark_stale(MODEL_REFERENCE_CATEGORY.image_generation)
# Next call to should_fetch_data() will return True
```

### `_invalidate_cache(category)` - Protected method for subclasses

Marks cache as invalid without deleting data. Use in your backend when write operations or errors occur.

```python
# ✅ FileSystemBackend pattern - invalidate after write
def update_model(self, category, model_name, record_dict):
    with self._lock:
        # ... write data to disk ...
        self._invalidate_cache(category)  # Force reload on next fetch

# ✅ FileSystemBackend pattern - invalidate on read error
try:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    self._store_in_cache(category, data)
except Exception as e:
    logger.error(f"Failed to read {file_path}: {e}")
    self._invalidate_cache(category)  # Ensure retry on next access
    return None
```

Similarly, use `_invalidate_legacy_cache(category)` for legacy format cache.

### `needs_refresh(category)` - Concrete Implementation

Public API provided by `ReplicaBackendBase` that implements the abstract method from `ModelReferenceBackend`. Checks if **existing** cached data has become stale.

⚠️ **Important**: Returns `False` when no data has been cached yet. Use `should_fetch_data()` for fetch decisions.

```python
# ✅ GitHubBackend uses this to force redownload
needs_refresh = self.needs_refresh(category)
if needs_refresh:
    logger.debug(f"Category {category} needs refresh, proceeding to download")
    overwrite_existing = True
```

## Hook Methods for Subclasses

### `_get_file_path_for_validation(category)` - Override for mtime tracking

Return a file path to enable automatic mtime validation for v2/converted format files.

```python
# ✅ GitHubBackend implementation
@override
def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
    return horde_model_reference_paths.get_model_reference_file_path(
        category,
        base_path=self.base_path,  # Points to converted/v2 files
    )
```

When implemented, the base class automatically:

- Tracks file mtime on cache store
- Invalidates cache when mtime changes
- Enables multi-process synchronization

### `_get_legacy_file_path_for_validation(category)` - Override for legacy mtime tracking

Return a file path for legacy format files if your backend maintains separate legacy files.

```python
# ✅ GitHubBackend implementation
@override
def _get_legacy_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
    return self._references_paths_cache.get(category)  # Points to legacy files
```

### `_additional_cache_validation(category)` - Override for custom validation

Add custom validation logic beyond TTL and mtime checks. Return `False` to invalidate cache.

```python
@override
def _additional_cache_validation(self, category: MODEL_REFERENCE_CATEGORY) -> bool:
    # Example: Check external condition
    if self.some_condition_not_met():
        return False  # Cache invalid
    return True  # Cache valid
```

## TTL-Based Expiration

Configure cache time-to-live during backend initialization:

```python
# ✅ GitHubBackend initialization with TTL
backend = GitHubBackend(
    base_path=horde_model_reference_paths.base_path,
    cache_ttl_seconds=horde_model_reference_settings.cache_ttl_seconds,
    replicate_mode=ReplicateMode.REPLICA,
)

# Cache never expires (must be explicitly invalidated)
backend = GitHubBackend(cache_ttl_seconds=None)
```

**How TTL Works:**

1. When data is cached, current time is stored in `_category_timestamps`
2. On validation, elapsed time is compared: `time.time() - last_updated > cache_ttl_seconds`
3. If expired, `should_fetch_data()` returns True
4. Re-fetching updates the timestamp, resetting the TTL

## File Mtime Validation

The base class automatically tracks file modification times when you override the validation hooks. This enables cache invalidation when files change externally.

**How It Works:**

1. On first cache, file's mtime is stored in `_last_known_mtimes`
2. On validation, current mtime is compared to stored mtime
3. If changed, cache is invalidated and `should_fetch_data()` returns `True`
4. Updating cache updates the stored mtime

**Benefits:**

- Detects external file modifications
- Enables multi-process synchronization
- Works alongside TTL expiration

See the "Hook Methods" section above for implementation examples.

## Legacy Format Cache Methods

For backends that maintain separate legacy format files (like `GitHubBackend`), the base class provides parallel methods for legacy caching:

### `_get_legacy_from_cache(category)`

Retrieves cached legacy data (both dict and string format).

```python
# ✅ GitHubBackend pattern
legacy_dict, legacy_string = self._get_legacy_from_cache(category)
if legacy_dict is not None:
    return legacy_dict
```

Returns a tuple: `(dict | None, str | None)`

### `_store_legacy_in_cache(category, legacy_dict, legacy_string)`

Stores legacy format data (both dict and string representations).

```python
# ✅ GitHubBackend pattern (inside _load_legacy_json_from_disk)
with open(file_path, "rb") as f:
    content = f.read()

data: dict[str, Any] = ujson.loads(content)
content_str = content.decode("utf-8")

self._store_legacy_in_cache(category, data, content_str)
```

### `is_legacy_cache_valid(category)`

Checks if legacy cache is valid, using the same validation logic as the primary cache (TTL, mtime, staleness).

These methods operate independently from the primary cache system, allowing backends to maintain both legacy and converted formats simultaneously.

## Thread Safety

`ReplicaBackendBase` provides thread-safe locking for critical sections.

### When to Use Locks

**Use locks when:**

- Your backend performs file I/O operations that could conflict
- Multiple threads might fetch the same category simultaneously
- You need to maintain consistency during download/conversion

**Locks are optional when:**

- Your backend only does network I/O (HTTP requests)
- The underlying operations are already thread-safe
- You're willing to accept redundant fetches in race conditions

### Available Locks

- `self._lock` (RLock) - Use for synchronous operations
- `self._async_lock` (AsyncLock) - Use for asynchronous operations

### Example: File-based backend (GitHubBackend)

```python
def fetch_category(self, category, *, force_refresh=False):
    with self._lock:  # Lock needed for file operations
        if force_refresh or self.should_fetch_data(category):
            self._download_and_convert_single(category, overwrite_existing=force_refresh)
            return self._load_converted_from_disk(category)
        return self._get_from_cache(category)
```

### Example: Network-only backend (HTTPBackend)

```python
def fetch_category(self, category, *, force_refresh=False):
    # No lock - HTTP requests are safe, redundant fetches acceptable
    if force_refresh or self.should_fetch_data(category):
        data = self._fetch_from_primary(category)
        if data is not None:
            self._store_in_cache(category, data)
        return data
    return self._get_from_cache(category)
```

### Example: Async operations

```python
async def fetch_category_async(self, category, *, force_refresh=False):
    async with self.async_lock:  # Use async lock for async operations
        if force_refresh or self.should_fetch_data(category):
            await self._download_legacy_async(category, overwrite_existing=force_refresh)
            return self._load_converted_from_disk(category)
        return self._get_from_cache(category)
```

## Important Design Constraints

These constraints are validated by the test suite and should be followed when implementing backends:

### 1. Cache Lifecycle Semantics

**`needs_refresh()` vs. `should_fetch_data()`:**

- `needs_refresh()` - Returns `True` ONLY when **cached data exists** and has become stale
- `should_fetch_data()` - Returns `True` for BOTH initial fetch (no cache) AND refresh (stale cache)

**Initial state behavior (no cache exists):**

```python
# When no data has been cached yet:
backend.is_cache_valid(category)    # Returns False
backend.needs_refresh(category)     # Returns False (not True!)
backend.should_fetch_data(category) # Returns True
```

**After caching data (cache exists but is stale):**

```python
# After cache expires or file changes:
backend.is_cache_valid(category)    # Returns False
backend.needs_refresh(category)     # Returns True (data needs refresh)
backend.should_fetch_data(category) # Returns True
```

**Key takeaway:** Use `should_fetch_data()` for fetch decisions, not `needs_refresh()`. The latter is specifically for detecting when **existing** data has become stale.

### 2. TTL Behavior with None

When `cache_ttl_seconds=None`, the cache never expires based on time:

```python
backend = GitHubBackend(cache_ttl_seconds=None)
# Cache remains valid until explicitly invalidated via:
# - mark_stale(category)
# - _invalidate_cache(category)
# - File mtime changes (if validation hook implemented)
```

### 3. Force Refresh and Redownload Parameters

Backends should support bypassing cache when explicitly requested:

**`force_refresh` parameter** (on fetch methods):

```python
def fetch_category(
    self,
    category: MODEL_REFERENCE_CATEGORY,
    *,
    force_refresh: bool = False,  # Bypass cache when True
) -> dict[str, Any] | None:
    if force_refresh or self.should_fetch_data(category):
        # Always fetch when force_refresh=True
        ...
```

**`redownload` parameter** (on legacy JSON methods):

```python
def get_legacy_json(
    self,
    category: MODEL_REFERENCE_CATEGORY,
    redownload: bool = False,  # Bypass cache when True
) -> dict[str, Any] | None:
    if redownload or self.should_fetch_data(category):
        # Always fetch when redownload=True
        ...
```

Both parameters serve the same purpose - bypassing cache to force a fresh fetch.

### 4. Async and Sync Cache Sharing

Async operations populate the same cache as sync operations:

```python
# Both operations share the same cache
sync_data = backend.fetch_category(category)
async_data = await backend.fetch_category_async(category)

# If sync_data was cached, async fetch returns from same cache
# No need to maintain separate caches for async/sync
```

### 5. Error Handling and Cache Invalidation

When errors occur during fetch/read operations, invalidate the cache to force retry:

```python
try:
    with open(file_path) as f:
        data = json.load(f)
    self._store_in_cache(category, data)
    return data
except Exception as e:
    logger.error(f"Failed to read {file_path}: {e}")
    self._invalidate_cache(category)  # Force retry on next access
    return None
```

This ensures transient errors don't result in permanently cached `None` values.

## Best Practices

### 1. Choose the Appropriate Pattern

**For network/API backends without local storage** (like `HTTPBackend`):

```python
def fetch_category(self, category, *, force_refresh=False):
    if force_refresh or self.should_fetch_data(category):
        data = self._fetch_from_source(category)
        if data is not None:
            self._store_in_cache(category, data)
        return data
    return self._get_from_cache(category)
```

**For local file reading backends** (like `FileSystemBackend`):

```python
def fetch_category(self, category, *, force_refresh=False):
    with self._lock:
        if force_refresh or self.should_fetch_data(category):
            file_path = self._get_file_path(category)
            try:
                with open(file_path) as f:
                    data = json.load(f)
                self._store_in_cache(category, data)
                return data
            except Exception:
                self._invalidate_cache(category)  # Retry on next access
                return None
        return self._get_from_cache(category)
```

**For download-then-load backends** (like `GitHubBackend`):

```python
def fetch_category(self, category, *, force_refresh=False):
    with self._lock:
        if force_refresh or self.should_fetch_data(category):
            self._download_to_disk(category)
            return self._load_from_disk_and_cache(category)
        return self._get_from_cache(category)
```

Key elements for all patterns:

1. Check `should_fetch_data()` to decide if fetch is needed
2. Perform your backend-specific fetch/read/download
3. Store in cache (directly or via file load helper)
4. Handle errors appropriately (cache `None` or invalidate)
5. Return cached data via `_get_from_cache()`

### 2. Implement the Validation Hooks

Override the file path hooks to enable automatic mtime validation:

```python
@override
def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
    return horde_model_reference_paths.get_model_reference_file_path(
        category,
        base_path=self.base_path,
    )
```

### 3. Use Public APIs for Cache Control

```python
# ✅ GOOD: Use public API to mark stale
backend.mark_stale(category)

# ❌ BAD: Don't manipulate internal state directly
backend._stale_categories.add(category)  # Bypasses proper handling
```

### 4. Handle Both Initial and Refresh Cases

Use `should_fetch_data()` which handles both:

```python
# ✅ GOOD: Unified check
if force_refresh or self.should_fetch_data(category):
    fetch_data()

# ⚠️ AVOID: Manual separation (more complex, error-prone)
if not self.has_cached_data(category):
    initial_fetch()
elif self.needs_refresh(category):
    refresh_fetch()
```

### 5. Invalidate Cache After Write Operations

If your backend supports writes, invalidate cache after modifying data:

```python
# ✅ FileSystemBackend pattern
def update_model(self, category, model_name, record_dict):
    with self._lock:
        # Write to disk
        self._write_to_disk(category, updated_data)
        # Invalidate cache to force reload
        self._invalidate_cache(category)
        logger.info(f"Updated {model_name} in {category}")

def delete_model(self, category, model_name):
    with self._lock:
        # Delete from disk
        self._delete_from_disk(category, model_name)
        # Invalidate cache to force reload
        self._invalidate_cache(category)
        logger.info(f"Deleted {model_name} from {category}")
```

This ensures the next `fetch_category()` call reloads the modified data from disk.

## Summary

### Relationship with ModelReferenceBackend

`ReplicaBackendBase` extends `ModelReferenceBackend` and provides:

**Concrete implementations:**

- `needs_refresh(category)` - Checks if cached data is stale
- `mark_stale(category)` - Marks category as requiring refresh

**Protected helper methods for your implementations:**

- `should_fetch_data(category)` - Check if data needs fetching
- `_get_from_cache(category)` - Retrieve valid cached data
- `_store_in_cache(category, data)` - Store fetched data
- `_invalidate_cache(category)` - Mark cache as invalid

**You must still implement these abstract methods:**

- `fetch_category()` - Your data fetching logic with caching
- `fetch_all_categories()` - Batch fetching with caching
- `fetch_category_async()` - Async fetching with caching
- `fetch_all_categories_async()` - Async batch fetching with caching
- `get_category_file_path()` - Return file paths (if applicable)
- `get_all_category_file_paths()` - Return all file paths
- `get_legacy_json()` - Legacy format retrieval with caching
- `get_legacy_json_string()` - Legacy format string retrieval with caching

### Essential Helpers for Your Implementations

| Helper Method | Purpose | When to Use |
|---------------|---------|-------------|
| `should_fetch_data(category)` | Check if data needs fetching | In `fetch_category()` to decide whether to fetch |
| `_get_from_cache(category)` | Retrieve valid cached data | To return cached data after checking |
| `_store_in_cache(category, data)` | Store fetched data | After loading data from disk/network |
| `_invalidate_cache(category)` | Mark cache as invalid | On write operations or errors |

### Standard Implementation Patterns

#### Pattern A: Direct cache (HTTPBackend style)

```python
def fetch_category(self, category, *, force_refresh=False):
    """Fetch data and cache directly."""
    if force_refresh or self.should_fetch_data(category):
        data = self._fetch_from_api(category)
        if data is not None:
            self._store_in_cache(category, data)
        return data
    return self._get_from_cache(category)
```

#### Pattern B: Read from disk (FileSystemBackend style)

```python
def fetch_category(self, category, *, force_refresh=False):
    """Read from local filesystem and cache."""
    with self._lock:
        if force_refresh or self.should_fetch_data(category):
            file_path = self._get_file_path(category)
            try:
                with open(file_path) as f:
                    data = json.load(f)
                self._store_in_cache(category, data)
                return data
            except Exception:
                self._invalidate_cache(category)
                return None
        return self._get_from_cache(category)
```

#### Pattern C: Download and load (GitHubBackend style)

```python
def fetch_category(self, category, *, force_refresh=False):
    """Download to disk, then load and cache."""
    with self._lock:
        if force_refresh or self.should_fetch_data(category):
            self._download_and_convert_single(category, overwrite_existing=force_refresh)
            return self._load_converted_from_disk(category)  # Calls _store_in_cache internally
        return self._get_from_cache(category)
```

### What You Must Implement

You must implement all abstract methods from `ModelReferenceBackend`:

1. **`fetch_category()`** - Your data fetching logic (see patterns above)
2. **`fetch_all_categories()`** - Batch fetching (typically loops over fetch_category)
3. **`fetch_category_async()`** - Async version of fetch_category
4. **`fetch_all_categories_async()`** - Async batch fetching
5. **`get_category_file_path()`** - Return file path or None
6. **`get_all_category_file_paths()`** - Return all file paths
7. **`get_legacy_json()`** - Legacy format retrieval (see patterns in implementations)
8. **`get_legacy_json_string()`** - Legacy format string retrieval

### Optional Overrides (Provided by ReplicaBackendBase)

**File path validation hook** (enables mtime tracking - only needed if your backend stores files):

```python
@override
def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
    return self.base_path / f"{category.value}.json"
```

If you don't override this (returns `None` by default), the base class skips mtime validation and relies on TTL and explicit staleness marking.

**Examples:**

- `HTTPBackend` - Doesn't override (no local files, relies on TTL only)
- `GitHubBackend` - Overrides to enable mtime tracking of downloaded files

That's it! The base class handles all cache validation, TTL expiration, mtime tracking, and staleness management automatically.
