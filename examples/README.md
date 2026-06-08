# Examples

Small, self-contained, runnable scripts for the most common ways to use
`horde-model-reference`. Each file stands alone - read it top to bottom, then
run it.

```bash
uv run examples/01_quickstart.py
```

(Or `python examples/01_quickstart.py` from an environment with the package installed.)

| Script | What it shows |
| ------ | ------------- |
| [`01_quickstart.py`](01_quickstart.py) | Create the manager, list a category, look up one model |
| [`02_query_recipes.py`](02_query_recipes.py) | Filter, sort, paginate, and aggregate with the fluent query API |
| [`03_working_with_records.py`](03_working_with_records.py) | Type narrowing, classification, download URLs, serialization |
| [`04_register_provider.py`](04_register_provider.py) | Register a read-only third-party provider and read it via `source=` |

> **Note:** The first data access fetches model references over the network
> (the PRIMARY server at `aihorde.net`, with a GitHub fallback) and caches the
> result in memory. Subsequent calls in the same process are served from cache.

For narrative walkthroughs of these topics, see the
[Getting Started](../docs/tutorials/getting_started.md),
[Querying Models](../docs/tutorials/querying_models.md), and
[Registering & Consuming Providers](../docs/tutorials/registering_providers.md)
tutorials.
