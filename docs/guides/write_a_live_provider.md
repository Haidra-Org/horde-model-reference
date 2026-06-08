# Write a live (remote) provider

**Goal:** expose a third-party catalog (e.g. a model site's API) as a read-only
source that consumers can read alongside the canonical horde data.

If your records are static, use
[`StaticModelProvider`](../tutorials/registering_providers.md#quick-start-contribute-records-in-a-few-lines)
instead - this guide is for sources you fetch at runtime.

## Implement the three required members

Subclass [`ModelProvider`][horde_model_reference.providers.base.ModelProvider]
and implement `source_id`, `provided_categories()`, and `fetch_category()`.
Below, `fetch_category` calls a remote API and caches the result with a small
TTL so repeated queries don't hammer the upstream:

```python
import time

import httpx

from horde_model_reference import ModelProvider, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    GenericModelRecord,
    get_record_type_for_category,
)


class CivitaiProvider(ModelProvider):
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, dict[str, GenericModelRecord]]] = {}

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
        if not self.serves_category(category):
            return None

        key = str(category)
        cached = self._cache.get(key)
        if cached and not force_refresh and (time.monotonic() - cached[0]) < self._ttl:
            return cached[1]

        record_type = get_record_type_for_category(category)
        resp = httpx.get("https://example.invalid/api/models", timeout=10)
        resp.raise_for_status()

        records: dict[str, GenericModelRecord] = {}
        for name, fields in resp.json().items():
            # Validate upstream data against the category's record type.
            records[name] = record_type.model_validate({"name": name, **fields})

        self._cache[key] = (time.monotonic(), records)
        return records
```

Notes:

- **Validate everything.** Run upstream payloads through the category's record
  type (`get_record_type_for_category`) so consumers always get well-formed
  records.
- **The library does not cache provider output.** Cache inside your provider if
  the upstream is expensive - `force_refresh=True` is the hint to bypass it.
- **Override `fetch_category_async()`** for a natively async upstream; the base
  class otherwise runs `fetch_category` in a thread.

## Register and consume

```python
from horde_model_reference import ModelReferenceManager, ANY_SOURCE

manager = ModelReferenceManager()
manager.register_provider(CivitaiProvider())

# Canonical only (default) - provider not consulted:
canonical = manager.query("image_generation").to_list()

# Canonical + every provider; canonical wins name collisions:
combined = manager.query("image_generation", source=ANY_SOURCE).to_list()

# Just your source:
mine = manager.query("image_generation", source="civitai").to_list()
```

Provider failures are isolated: if `fetch_category` raises, the error is logged
and other sources still return.

## See also

- [Registering & Consuming Providers](../tutorials/registering_providers.md) - the full tutorial, including custom record types
- [Model Providers](../reference/model_providers.md) - interface reference, source-selection and collision rules
- [`examples/04_register_provider.py`](https://github.com/Haidra-Org/horde-model-reference/blob/main/examples/04_register_provider.py) - runnable static-provider version
