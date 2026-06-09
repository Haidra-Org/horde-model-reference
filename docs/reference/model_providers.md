# Model Providers

## Overview

The provider system lets consumers register third-party sources of model records and read canonical
data, a specific provider, or all sources together. It is the read-side counterpart to the canonical
write loop: providers contribute records, consumers select them via a `source` argument, and the
library never writes back to a provider.

**Key pieces:**

- [`ModelProvider`][horde_model_reference.providers.base.ModelProvider] - the abstract base class a
  third-party source implements.
- [`StaticModelProvider`][horde_model_reference.providers.static_provider.StaticModelProvider] - a
  ready-to-use, in-memory provider for the common case (records you already have).
- [`ModelProviderRegistry`][horde_model_reference.providers.registry.ModelProviderRegistry] - the
  thread-safe `source_id -> provider` registry owned by each `ModelReferenceManager`.
- `source_consts` - the source-selection constants and the `SourceSelector` type.

For a task-oriented walkthrough see [Registering & Consuming Providers](../tutorials/registering_providers.md).

## Source selection

Every read API accepts a `source` argument of type `SourceSelector`
(defined in `horde_model_reference.source_consts`):

| Selector                | Meaning                                                              |
| ----------------------- | ------------------------------------------------------------------- |
| `"horde"` (default)     | Canonical horde data only. Existing callers are unaffected.         |
| `"any"`                 | Canonical data merged with **all** registered providers.            |
| `"civitai"`             | A single provider id.                                               |
| `["horde", "civitai"]`  | An explicit, ordered set. Earlier ids win name collisions.          |

`"horde"` (`HORDE_SOURCE_ID`) and `"any"` (`ANY_SOURCE`) are the
`RESERVED_SOURCE_IDS`; providers cannot
register under them. Naming an unregistered source id explicitly raises `ValueError`.

## Collision and de-duplication rules

When more than one source is selected:

1. Records are gathered **canonical-first**, then each provider in selector order.
2. They are de-duplicated by model name; the **first occurrence wins** (canonical, or the earlier id in
   an explicit sequence).
3. Duplicates are retained for inspection - the query builder exposes
   [`duplicate_names()`][horde_model_reference.query.ModelQuery.duplicate_names],
   `has_duplicate_names()`, `where_source()`, `to_list_with_source()`, `sources()`, and
   `group_by_source()`.

A provider that raises during fetch is logged and skipped (error isolation); a provider returning
`None` for a category contributes nothing. Both cases leave that source absent from the merged
result. A consumer can distinguish them with
[`source_status()`][horde_model_reference.query.ModelQuery.source_status] and
[`failed_sources()`][horde_model_reference.query.ModelQuery.failed_sources] on the query builder,
which report each selected source as `"ok"`, `"empty"`, or `"error"`.

## The `ModelProvider` interface

Subclasses must implement `source_id`, `provided_categories()`, and `fetch_category()`. The async
variant defaults to running `fetch_category` in a worker thread.

See the [`ModelProvider`][horde_model_reference.providers.base.ModelProvider] API reference for details, and the
[Registering & Consuming Providers tutorial](../tutorials/registering_providers.md) for
examples of how to implement, register, and consume providers, plus the collision and error-handling
behaviors in action.

## `StaticModelProvider`

The lowest-friction implementation: construct it with already-built records, or use `from_raw` to
validate plain dicts against each category's record type (from
[`MODEL_RECORD_TYPE_LOOKUP`][horde_model_reference.model_reference_records.MODEL_RECORD_TYPE_LOOKUP],
falling back to `GenericModelRecord`). The mapping key is injected as each record's `name`.

See the [`StaticModelProvider`][horde_model_reference.providers.static_provider.StaticModelProvider] API reference for details.

## `ModelProviderRegistry`

See the [`ModelProviderRegistry`][horde_model_reference.providers.registry.ModelProviderRegistry] API reference for details.

## Manager integration

`ModelReferenceManager` owns a registry and exposes:

- `register_provider(provider, *, replace=False)` - register a provider (raises on duplicate/reserved id unless `replace`).
- `unregister_provider(source_id)` - remove a provider; returns whether one existed.
- `list_providers()` - registered source ids in registration order.
- `get_provider(source_id)` - the provider instance, or `None`.
- `provider_registry` - the underlying [`ModelProviderRegistry`][horde_model_reference.providers.registry.ModelProviderRegistry].

The `source=` argument on `query()`, `get_model_reference()`, and `get_model()` (plus their `_or_none`
and async variants) drives selection and merging.

## Constraints

- Providers are read-only: `supports_writes()` always returns `False`.
- The library does not cache provider output; `cache_ttl_seconds()` is advisory metadata.
- Records returned by a provider are trusted as-is - the library does not re-validate provider output
  against any schema (use `StaticModelProvider.from_raw` if you want validation at construction time).
