# Design Decisions & Known Limitations

This page explains user-visible design trade-offs so you know what to expect when using the library.

## Singleton Manager

`ModelReferenceManager` is a singleton. The first instantiation locks in all configuration (backend, prefetch strategy, etc.). Subsequent calls with the same parameters return the existing instance; calls with _different_ parameters raise `RuntimeError`.

**Why:** The manager owns caches, backend connections, and download state. Multiple instances with conflicting settings would lead to race conditions, duplicate downloads, and inconsistent cached data.

**What to do:** Initialize once at application startup. Retrieve the instance elsewhere with `ModelReferenceManager()` (no args) or `ModelReferenceManager.get_instance()`.

## or_none Methods

Methods named `*_or_none` return `None` on failure instead of raising it. The name means "the return type includes `None`, so you must handle that case."

| Method                             | Returns                                 | On failure            |
| ---------------------------------- | --------------------------------------- | --------------------- |
| `get_model_reference(cat)`         | `dict[str, GenericModelRecord]`         | Raises `RuntimeError` |
| `get_model_reference_or_none(cat)` | `dict[str, GenericModelRecord] \| None` | Returns `None`        |

**Why:** Some consumers (workers) want guaranteed data and prefer exceptions. Others (dashboards) want graceful degradation. Both patterns are supported.

## Environment-Driven Configuration

Settings are read from environment variables (prefix `HORDE_MODEL_REFERENCE_`) at import time via the Pydantic Settings singleton. This means:

- Changing an env var after import has no effect on the already-created settings object
- The settings object validates combinations on creation (e.g., warns if REPLICA mode has Redis enabled)

**Why:** Import-time configuration is standard for Pydantic Settings and avoids the complexity of runtime reconfiguration with an already-initialized singleton manager.

## Async / Sync Separation

The library provides parallel sync and async method sets (`get_model_reference` / `get_model_reference_async`). You should not mix them in the same execution context.

**Why:** The backends use different HTTP client implementations (`httpx.Client` vs `httpx.AsyncClient`). Calling sync methods from within an async context (or vice versa) can block the event loop or cause `RuntimeError` from nested event loop usage.

## Return Type Precision

`get_model_reference()` returns `dict[str, GenericModelRecord]` even though the actual records are specialized subclasses (e.g., `ImageGenerationModelRecord`). This is because the method accepts any category.

**Workarounds:**

- Use the query API, which is fully typed per category: `manager.query("image_generation")` returns records typed as `ImageGenerationModelRecord`
- Use `isinstance()` checks for type narrowing
