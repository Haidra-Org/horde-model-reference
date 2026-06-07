# Registering & Consuming Providers

A **model provider** lets a third party (for example, a model catalog like Civitai) contribute model
records *without* going through the canonical horde write loop. Providers are **read-only** from the
library's perspective: you supply records, consumers read them, and the library never writes back or
re-validates against a server.

Every source has a stable **source id**. The canonical horde data is `"horde"`; the special selector
`"any"` means "canonical plus every registered provider". You choose any other id for your provider
(e.g. `"civitai"`).

## Quick start: contribute records in a few lines

If you already have your records as plain dictionaries, [`StaticModelProvider.from_raw`][horde_model_reference.providers.static_provider.StaticModelProvider.from_raw]
validates them against each category's record type and serves them:

```python
from horde_model_reference import (
    ModelReferenceManager,
    MODEL_REFERENCE_CATEGORY,
    StaticModelProvider,
)

provider = StaticModelProvider.from_raw(
    "civitai",  # your source id
    {
        MODEL_REFERENCE_CATEGORY.image_generation: {
            "my_cool_model": {
                "baseline": "stable_diffusion_xl",
                "nsfw": False,
                "description": "A community model",
            },
        },
    },
)

manager = ModelReferenceManager()
manager.register_provider(provider)
```

The mapping key (`"my_cool_model"`) is injected as the record's `name`, so you don't repeat it. If a
raw record fails validation, `from_raw` raises a `pydantic.ValidationError` immediately - validation is
your responsibility, and this is the hook for it.

If you already hold *built* record objects (instances of `GenericModelRecord` or a subclass), pass them
to the constructor directly instead of `from_raw`:

```python
provider = StaticModelProvider(
    "civitai",
    {MODEL_REFERENCE_CATEGORY.image_generation: {record.name: record}},
)
```

## Consuming provider records

Once registered, providers are visible through the normal read API via the `source=` argument. The
default is always canonical-only, so existing code is unaffected.

```python
from horde_model_reference import ANY_SOURCE  # == "any"

# Canonical only (default)
manager.query("image_generation").to_list()

# Just one provider
manager.query("image_generation", source="civitai").to_list()

# Canonical merged with every provider
manager.query("image_generation", source=ANY_SOURCE).to_list()

# Explicit, ordered set - earlier ids win name collisions
manager.query("image_generation", source=["horde", "civitai"]).to_list()
```

The same `source=` selector works on `get_model_reference()` and `get_model()`:

```python
merged = manager.get_model_reference("image_generation", source=ANY_SOURCE)
```

### Collisions and provenance

When several sources are merged, records are de-duplicated by name with the **canonical (or
earlier-listed) source winning**. Collisions are never silently lost - you can inspect them:

```python
q = manager.query("image_generation", source=ANY_SOURCE)

q.has_duplicate_names()      # did any name come from more than one source?
q.duplicate_names()          # {name: [source_id, ...]} for colliding names
q.where_source("civitai")    # keep only one source's contributions
q.to_list_with_source()      # [(record, source_id), ...]
q.group_by_source()          # {source_id: [record, ...]}
```

### Error isolation

If a provider raises while fetching (network error, bad data), the manager logs it and **skips that
provider** - other sources still contribute. Prefer returning `None` from `fetch_category` over raising
when you simply have no data for a category.

## Introspecting registered providers

```python
manager.list_providers()              # ["civitai", ...] in registration order
manager.get_provider("civitai")       # the provider instance, or None
manager.unregister_provider("civitai")  # True if one was removed
```

## Advanced: a live / remote provider

For data that must be fetched lazily (an HTTP API, a database), subclass
[`ModelProvider`][horde_model_reference.providers.base.ModelProvider] and implement the three abstract
members. The async path has a thread-pool default; override `fetch_category_async` only if you have a
native async data path.

```python
from horde_model_reference import ModelProvider, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import GenericModelRecord


class CivitaiProvider(ModelProvider):
    @property
    def source_id(self) -> str:
        return "civitai"

    def provided_categories(self) -> set[MODEL_REFERENCE_CATEGORY | str]:
        return {MODEL_REFERENCE_CATEGORY.image_generation}

    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY | str,
        *,
        force_refresh: bool = False,
    ) -> dict[str, GenericModelRecord] | None:
        if category != MODEL_REFERENCE_CATEGORY.image_generation:
            return None
        raw = self._download(force_refresh=force_refresh)  # your code
        return {
            name: ImageGenerationModelRecord.model_validate({**fields, "name": name})
            for name, fields in raw.items()
        }
```

## Custom record types

Providers may carry fields the built-in records don't have. Subclass `GenericModelRecord` (or a
specialized record) and register it for a category with
[`register_record_type`][horde_model_reference.model_reference_records.register_record_type]; both
`from_raw` and the query layer will then build and return your type for that category:

```python
from horde_model_reference.model_reference_records import GenericModelRecord, register_record_type
from horde_model_reference import MODEL_REFERENCE_CATEGORY


@register_record_type(MODEL_REFERENCE_CATEGORY.miscellaneous)
class MyRecord(GenericModelRecord):
    extra_field: str = "default"
```

## Notes & caveats

- Providers are **read-only**; there is no provider write path (`supports_writes()` is always `False`).
- The library does **not** cache provider output; `cache_ttl_seconds()` is advisory metadata only.
- Reserved source ids `"horde"` and `"any"` are rejected at registration / construction time.

## Next steps

- [Querying Models](querying_models.md) -- The full query builder reference
- [Model Providers (reference)](../reference/model_providers.md) -- ABC contract, registry, and source-selection rules
```
