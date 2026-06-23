# Glossary

Plain-language definitions of the terms that show up across these docs. Each
entry links to the page where the concept is covered in depth.

## Replicate mode (REPLICA / PRIMARY)

Every instance of the library runs in one of two modes, set by the
`HORDE_MODEL_REFERENCE_REPLICATE_MODE` environment variable:

- **REPLICA** (the default) - a *read-only consumer*. It fetches model
  reference data from somewhere else and serves it locally. This is what you
  are using when you `import horde_model_reference` in a worker, client, or
  script. **As a consumer you do not need to configure anything** - REPLICA is
  the default.
- **PRIMARY** - the *authoritative source*. It owns the data on its own
  filesystem and can accept writes (create/update/delete). You only run PRIMARY
  if you are hosting the canonical dataset for others.

See [Architecture Overview](../concepts/architecture_overview.md#backend-selection)
for how the mode selects a backend, and
[Primary Deployments](api_deployments.md) for running a PRIMARY server.

## PRIMARY server

The running service (in PRIMARY mode) that hosts the official model reference
data over HTTP. The public one is `models.aihorde.net`. A REPLICA fetches from a
PRIMARY server via its `PRIMARY_API_URL` setting. "PRIMARY server" is just a
PRIMARY-mode instance viewed from the outside.

## GitHub fallback

Before this library existed, model references lived in two GitHub repositories
([image](https://github.com/Haidra-Org/AI-Horde-image-model-reference),
[text](https://github.com/Haidra-Org/AI-Horde-text-model-reference)) in a
"legacy" format. Those repos still exist, so a REPLICA can read them directly.
**GitHub fallback** means: if the PRIMARY server is unreachable, the library
automatically downloads from GitHub instead (and converts the legacy format on
the fly). It is on by default (`ENABLE_GITHUB_FALLBACK=True`) and makes reads
resilient to a PRIMARY outage. See
[Read Resiliently](../guides/offline_and_resilient_reads.md).

## Canonical data (the `"horde"` source)

The official horde model dataset - as opposed to records contributed by a
third-party [provider](#provider-source). In the query API the canonical data
has the source id `"horde"`, which is the default `source=` for every read. When
data from multiple sources is merged, **canonical records win name collisions**.
See [Model Providers](model_providers.md).

> **Don't confuse this with [canonical *format*](#canonical-format).** "Canonical
> data" is *whose* data (horde vs. a provider). "Canonical format" is *which API
> version* is authoritative for writes.

## Canonical format

A PRIMARY-only setting (`CANONICAL_FORMAT`, default `v2`) that decides which API
version - v1 (legacy) or v2 - is the authoritative target for write operations.
It does not affect reads, and it is irrelevant to REPLICA consumers. See
[Canonical Format and API Versioning](../concepts/canonical_format.md).

## Backend

The pluggable component the manager uses to actually fetch (and, in PRIMARY mode,
write) data - e.g. `HTTPBackend`, `GitHubBackend`, `FileSystemBackend`. The
manager **selects one automatically** from your mode and configuration; you
rarely interact with it directly. See
[Model Reference Backend](model_reference_backend.md).

## Legacy format

The original GitHub-repo data shape (a flat JSON dict per category, MD5
checksums, etc.). The library converts legacy data into the modern record schema
automatically when reading from GitHub. Text models use a CSV legacy format -
see [Legacy CSV Conversion](legacy_csv_conversion.md).

## Category

A kind of model, e.g. `image_generation`, `text_generation`, `controlnet`,
`clip`. Categories are the `MODEL_REFERENCE_CATEGORY` enum and the unit you pass
to `get_model_reference(category)` / `query(category)`. Each maps to one JSON
file (e.g. `image_generation` -> `stable_diffusion.json`).

## Record

One model's metadata, as a typed Pydantic object (`GenericModelRecord` or a
category-specific subclass like `ImageGenerationModelRecord`). See
[Working with Records](../tutorials/working_with_records.md).

## Baseline

For image models, the base architecture a model is built on - e.g.
`stable_diffusion_1`, `stable_diffusion_xl`, `flux_1`. Exposed as the `baseline`
field and the `KNOWN_IMAGE_GENERATION_BASELINE` enum. See
[Working with Records](../tutorials/working_with_records.md#baselines).

## Provider / source

A **provider** is a read-only, third-party source of records you register to
sit alongside the canonical horde data. Each provider has a unique **source id**
(e.g. `"civitai"`, `"pending"`). Consumers opt in per call with the `source=`
argument (`"horde"`, `"any"`, a single id, or an ordered list; earlier ids win
name collisions). See
[Registering & Consuming Providers](../tutorials/registering_providers.md) and
[Model Providers](model_providers.md).

## Prefetch strategy

Controls *when* the manager loads data: `LAZY` (on first access, the default),
`SYNC` (at construction, blocking), `ASYNC` (background), `DEFERRED` (you
trigger it), or `NONE`. See
[Getting Started](../tutorials/getting_started.md#prefetch-strategy).

## Singleton

`ModelReferenceManager` is a singleton: the first construction locks in its
configuration and every later `ModelReferenceManager()` returns that same
instance. See [Getting Started](../tutorials/getting_started.md#the-singleton-pattern).
