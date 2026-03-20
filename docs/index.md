# Horde Model Reference

The Horde Model Reference is the authoritative source for AI model metadata in the [AI-Horde](https://aihorde.net) ecosystem. It provides validated metadata -- download URLs, checksums, baselines, NSFW flags, capabilities -- for image generation, text generation, and utility models used by workers, clients, and services.

## Getting Started

*I want to learn by doing.*

- [Getting Started](tutorials/getting_started.md) -- Install, run your first query, understand the singleton pattern and prefetch strategies
- [Querying Models](tutorials/querying_models.md) -- Filter, sort, and aggregate models with the fluent query API
- [Working with Records](tutorials/working_with_records.md) -- Record types, fields, type narrowing, serialization
- [Configuration & Troubleshooting](tutorials/configuration_and_troubleshooting.md) -- Environment variables, debugging, common issues

## Concepts

*I want to understand how the system works.*

- [Architecture Overview](concepts/architecture_overview.md) -- Three usage modes, backbone modules, subsystem map
- [Request Lifecycle](concepts/request_lifecycle.md) -- How the FastAPI service processes read and write requests end-to-end
- [Sync System](concepts/sync_system.md) -- How PRIMARY changes propagate to legacy GitHub repositories
- [Analytics Pipeline](concepts/analytics_pipeline.md) -- Statistics, audit analysis, text model grouping, and cache hydration
- [Integrations](concepts/integrations.md) -- Live Horde API data fetching, caching, and merging with model references
- [Canonical Format](concepts/canonical_format.md) -- API versioning (v1/v2) and canonical format configuration
- [Design Decisions](concepts/design_decisions.md) -- Trade-offs and known limitations explained

## Reference

*I need to look something up.*

- [Model Reference Backend](reference/model_reference_backend.md) -- Backend ABC, capability checks, implementation checklist
- [Model Reference Records](reference/model_reference_records.md) -- Record hierarchy, validation, registration pattern
- [Primary Deployments](reference/primary_deployments.md) -- Backend selection, Redis multi-worker setup, cache warming
- [Replica Backend Base](reference/replica_backend_base.md) -- TTL caching and staleness tracking infrastructure
- [Audit Trail](reference/audit_trail.md) -- Append-only operation logging and state replay
- [Pending Queue](reference/pending_queue.md) -- Propose / approve / apply change workflow
- [Legacy CSV Conversion](reference/legacy_csv_conversion.md) -- Converting from legacy text generation CSV format

## Code Reference

Auto-generated API documentation from source code docstrings. Browse the package reference in the **Code Reference** nav section, or run the FastAPI service and visit `/docs` for interactive Swagger UI documentation.
