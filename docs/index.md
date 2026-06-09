# Horde Model Reference

The Horde Model Reference is the authoritative source for AI model metadata in the [AI-Horde](https://aihorde.net) ecosystem. It provides validated metadata: download URLs, checksums, baselines, NSFW flags, capabilities: for image generation, text generation, and utility models used by workers, clients, and services.

New to the project? The [Glossary](reference/glossary.md) defines the recurring terms (PRIMARY/REPLICA, canonical data, GitHub fallback, backends, providers) in plain language.

## Where do I start?

This project is three things in one: a Python library, a public HTTP API, and a FastAPI service you can run yourself. Each serves the same data, but is optimized for different use cases:

| You want to... | Use | Start here |
| --- | --- | --- |
| Read model metadata from Python (worker, client, script) | The **library** (`pip install horde-model-reference`), runs in-process | [Getting Started](tutorials/getting_started.md) |
| Read metadata from a non-Python app, or without adding a dependency | The **HTTP API** on the public PRIMARY server (`https://models.aihorde.net/api`) | [Using the HTTP API](tutorials/using_the_http_api.md) |
| Surface your own / third-party models alongside the canonical set | A read-only **provider** (library feature) | [Registering & Consuming Providers](tutorials/registering_providers.md) |
| Host a replica of the canonical dataset for local reads and redundancy | Run the **FastAPI service** in REPLICA mode | [API Deployments](reference/api_deployments.md) |
| Host the canonical dataset for others (uncommon, for AI-Horde forks or custom deployments) | Run the **FastAPI service** in PRIMARY mode | [Primary Deployments](reference/api_deployments.md) |

**Library or HTTP API?** If you are already in Python, prefer the library: it caches in-process,
returns typed records, and automatically falls back to GitHub if the PRIMARY server is down. Reach
for the HTTP API when you are not in Python, want zero install footprint, or are integrating against
a central server you do not operate. Both read the same data.

As a *reader* you never choose PRIMARY vs REPLICA: consumers are REPLICA (read-only) by default,
with no configuration required. See the [Glossary](reference/glossary.md#replicate-mode-replica-primary).

## Getting Started

*I want to learn by doing.*

- [Getting Started](tutorials/getting_started.md): Install, run your first query, understand the singleton pattern and prefetch strategies
- [Querying Models](tutorials/querying_models.md): Filter, sort, and aggregate models with the fluent query API
- [Using the HTTP API](tutorials/using_the_http_api.md): Call the live service: list, fetch, search, and rank models over HTTP
- [Registering & Consuming Providers](tutorials/registering_providers.md): Contribute and read records from third-party sources
- [Working with Records](tutorials/working_with_records.md): Record types, fields, type narrowing, serialization
- [Configuration & Troubleshooting](tutorials/configuration_and_troubleshooting.md): Environment variables, debugging, common issues

Prefer working code? The [`examples/`](https://github.com/Haidra-Org/horde-model-reference/tree/main/examples) directory has runnable scripts for each topic.

## How-To Guides

*I have a specific task to accomplish.*

- [Filter Models for a Worker](guides/filter_models_for_a_worker.md): Narrow the reference to the models your node should serve
- [Read Resiliently](guides/offline_and_resilient_reads.md): Caching, GitHub fallback, and graceful degradation when offline
- [Consume the HTTP API](guides/consume_the_http_api.md): Integrate the read API into a worker or client; detect changes cheaply
- [Submit Models via the API](guides/submit_models_via_the_api.md): The authenticated propose / approve / apply write workflow
- [Write a Live Provider](guides/write_a_live_provider.md): Expose a remote catalog as a read-only source

## Concepts

*I want to understand how the system works.*

- [Architecture Overview](concepts/architecture_overview.md): Three usage modes, backbone modules, subsystem map
- [The HTTP Service](concepts/http_service.md): What the FastAPI service is for, who calls it, and where it fits in the AI-Horde ecosystem
- [Request Lifecycle](concepts/request_lifecycle.md): How the FastAPI service processes read and write requests end-to-end
- [Sync System](concepts/sync_system.md): How PRIMARY changes propagate to legacy GitHub repositories
- [Analytics Pipeline](concepts/analytics_pipeline.md): Statistics, audit analysis, text model grouping, and cache hydration
- [Integrations](concepts/integrations.md): Live Horde API data fetching, caching, and merging with model references
- [Canonical Format](concepts/canonical_format.md): API versioning (v1/v2) and canonical format configuration
- [Design Decisions](concepts/design_decisions.md): Trade-offs and known limitations explained

## Reference

*I need to look something up.*

- [Glossary](reference/glossary.md): Plain-language definitions of PRIMARY/REPLICA, canonical data, backends, providers, and more
- [Model Reference Backend](reference/model_reference_backend.md): Backend ABC, capability checks, implementation checklist
- [Model Providers](reference/model_providers.md): Provider ABC, registry, source selection and collision rules
- [Model Reference Records](reference/model_reference_records.md): Record hierarchy, validation, registration pattern
- [HTTP API](reference/http_api/conventions.md): Full REST reference: conventions, v2 & v1 endpoints, text utilities, pending queue
- [Primary Deployments](reference/api_deployments.md): Backend selection, Redis multi-worker setup, cache warming
- [Replica Backend Base](reference/replica_backend_base.md): TTL caching and staleness tracking infrastructure
- [Audit Trail](reference/audit_trail.md): Append-only operation logging and state replay
- [Pending Queue](reference/pending_queue.md): Propose / approve / apply change workflow
- [Legacy CSV Conversion](reference/legacy_csv_conversion.md): Converting from legacy text generation CSV format

## Code Reference

Auto-generated API documentation from source code docstrings. Browse the package reference in the **Code Reference** nav section. For the **HTTP API**, see the [HTTP API reference](reference/http_api/conventions.md) or visit the live [`/api/docs`](https://models.aihorde.net/api/docs) Swagger UI.
