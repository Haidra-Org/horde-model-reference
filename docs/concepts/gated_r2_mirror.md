# The gated R2 model mirror

Generation checkpoints are too large for the project to host, but the *auxiliary* models the ecosystem depends
on (controlnets and their annotators, CLIP/BLIP feature extractors, ESRGAN/GFPGAN/CodeFormer post-processors,
the safety checker, and miscellaneous helpers) are small enough to mirror on Cloudflare R2, where egress is
free. This page explains the scheme that lets us host them cheaply **without** becoming a free public download
service.

## Why a mirror, not a replacement

A model record's `DownloadRecord` still points at its original host. The R2 copy is a *preferred mirror*, not the
source of truth: a download tries the mirror first and falls back to the origin URL on **any** failure: an
ineligible key, an object not yet uploaded, or a Cloudflare outage. This keeps two important properties:

- **No single point of failure.** If the mirror is down or a file is not mirrored, the ecosystem still works.
- **No hard dependency for outsiders.** A standalone `hordelib` user, an anonymous worker, or anyone without an
  eligible key simply downloads from origin, exactly as before. The mirror only ever *adds* a fast, free path.

## Content addressing

Objects are stored by content: the key is `by-hash/<sha256>`. The download engine already knows a file's
declared sha256, so it derives the mirror URL itself (`gateway_url_for`) with no change to the record schema,
and the same file shared by several records is stored once. A record whose hash is still the `"FIXME"` sentinel
cannot be addressed, so it is served only from origin until its real hash is
[backfilled](../guides/mirror_models_to_r2.md).

## How a download decides

`download_engine.download_record_files` is given an apikey by its caller (the worker via `hordelib`) and uses
either the explicit gateway base URL passed by that caller or `HORDE_MODEL_REFERENCE_R2__GATEWAY_URL` from
settings. For each file it builds an ordered list of candidates and uses the first that succeeds:

1. `GET <gateway>/by-hash/<sha256>` with the apikey (only when a gateway, an apikey, and a real hash all exist).
2. the record's origin `file_url`.

A definitive rejection from the mirror (HTTP 401/403/404/410) does not consume the retry budget; the engine
moves straight to origin. A *transient* mirror failure (a 5xx or a timeout) is given only a single attempt
before falling through, rather than the full retry-with-backoff budget the origin keeps: the mirror is an
accelerator, so a degraded mirror must hand off promptly instead of making the download slower than having no
mirror at all. The sha256 is verified after **every** download regardless of source, so a tampered or stale
mirror object is rejected and re-fetched from origin; a file already on disk is likewise re-validated against
its declared hash, so a corrupt object left by a past failure is not trusted forever.

## ControlNet annotators

The ControlNet *annotator* checkpoints (the `comfyui_controlnet_aux` detector weights: HED, LeReS depth, OpenPose,
UniFormer segmentation, M-LSD) are not model-reference records, but they download the same way and benefit from
the same mirror. They are made known in the torch-free
[`annotator_catalog`][horde_model_reference.annotator_catalog], which lists each file with the exact
repo / subfolder / filename the node expects on disk and its HuggingFace origin URL. The worker pre-fetches these
through this engine (gateway then origin) *before* the detectors run, so the node finds them present and skips its
own download; the [upload tool](../guides/mirror_models_to_r2.md) hosts them like any other file. MiDaS (the
`normal` type) is excluded: it loads via HuggingFace `transformers`, a separate mechanism.

## The gate

The mirror bucket is private. In front of it sits a Cloudflare Worker (its own repository,
`horde-r2-model-gateway`) that, for each request:

1. reads the apikey (an `apikey` header or `Authorization: Bearer`),
2. applies a coarse **per-key rate limit** (see below) before any further work,
3. resolves it against the AI Horde `/v2/find_user` endpoint (with a request timeout), caching the resolved
   **user** in KV (keyed by a hash of the key, with separate positive/negative TTLs) so the fleet does not
   hammer the API,
4. applies the **eligibility policy** to that user *on every request*, and
5. streams the object from its R2 binding, honouring HTTP `Range` for resumable downloads.

Caching the user rather than the allow/deny decision means a maintainer can change the policy vars and have it
take effect on the next request, instead of waiting out a cached decision. A *definitive* invalid key (the
Horde API returns 404) is negatively cached; a *transient* Horde-API failure (timeout or 5xx) is **not** cached
and returns a 403 the client treats as "fall back to origin", so a Horde outage degrades to plain origin
downloads rather than locking anyone out or serving ungated.

The Worker reads, but never modifies, the existing Horde API, so the scheme needs **no changes to the AI Horde
server**.

## Rate limiting and key safety

Two coarse safeguards bound abuse and misconfiguration:

- **Per-second rate limit.** The Worker applies a generous per-key limit (Cloudflare's Rate Limiting binding,
  keyed on a hash of the apikey) *before* any KV or Horde-API work, so a single key cannot be scripted as a bulk
  scraper or hammer the gate in a retry loop. This is a coarse anti-abuse limit, not a quota: R2 egress is free
  and the objects are public weights, and a throttled request returns `429`, which the client treats as a
  transient mirror failure and simply satisfies from the origin host. Tune the limit in `wrangler.toml`; raise
  it for large fleets that share one key.
- **Monthly caps (a Durable Object).** Because the per-second limit must stay generous (a fleet shares one key),
  month-scoped caps bound the longer game. A **per-(key, file) served cap** stops a key slow-dripping re-pulls
  of a file it already has (a legitimate worker fetches each file roughly once, then caches it to disk). A
  higher **per-key miss cap** catches valid-key hash-spray attempts: arbitrary `by-hash` lookups cost Worker,
  KV, DO and R2 lookup operations even though they serve no bytes. A **global op-budget kill-switch** caps total
  served ops per month, bounding the blast radius of *any* attack shape, including a many-key botnet that
  per-key limits structurally cannot catch; over the ceiling the gate stops serving (`503`, still an origin
  fallback for the client) until the month rolls over. The counters live in a Durable Object rather than KV on
  purpose: a per-request KV counter would, ironically, become the most expensive line under the very abuse it
  caps, whereas a DO increments its own state with no per-write billing. An over-cap request, like every other
  denial, is one the client satisfies from origin, so the caps never block a legitimate download. See
  `scripts/estimate-cost.mjs` in the gateway repo for the cost model these defend, and the README for tuning.
- **The key is only ever sent over `https`.** The worker's apikey must be sent to the gateway (that is how the
  gate validates it), so the download client refuses to attach the key to a non-`https` gateway URL: a
  plaintext gateway disables the mirror (origins are used) rather than leaking the key in transit. The gateway
  URL should point at the official deployment. It is configured through
  `HORDE_MODEL_REFERENCE_R2__GATEWAY_URL`; operators should point it only at a trusted host.

## The eligibility policy

The policy is a configurable OR of independently-enabled grant paths, set entirely through the Worker's
`POLICY_*` vars:

- `trusted` (default on): the AI Horde "trusted" flag.
- worker-owner: the account owns at least one registered worker.
- kudos threshold.
- explicit user-id allowlist.
- "any valid key" (a dev escape hatch; off by default).

With nothing enabled the policy denies (fail-closed). The worker-owner and kudos paths exist to avoid a
**bootstrap deadlock**: a brand-new, not-yet-trusted worker still needs the safety checker and annotators in
order to run and start earning trust. Tune the policy by editing vars and redeploying; client code never
changes.

## Redistribution is opt-in

Hosting a third-party weight is redistribution, which not every upstream license permits. The
[upload tool](../guides/mirror_models_to_r2.md) therefore mirrors **nothing** by default: a model is uploaded
only once a maintainer has added it **by name** to the redistributable allowlist. The opt-in is deliberately
per-model, not per-category, so a model added to the reference later is never mirrored until someone has
reviewed its licence; the licence reviewed under is recorded on the allowlist entry and stamped onto the
uploaded object.
