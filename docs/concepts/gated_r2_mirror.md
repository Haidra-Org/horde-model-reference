# The gated R2 model mirror

The scheme mirrors redistribution-approved auxiliary weights to a private Cloudflare R2 bucket while keeping
each model record's original URL authoritative. Generation checkpoints and anything unreviewed or license-blocked
remain origin-only. This makes the mirror an accelerator, not a new source of truth or public hosting service.

## Client routing

Objects are deduplicated at `by-hash/<sha256>`. A successful sync atomically publishes a small inventory at the
gateway's public `GET /v1/manifest` endpoint. Clients cache it briefly and attempt R2 only when all of these are
true:

1. the record declares a real SHA-256 rather than `FIXME`;
2. the published inventory contains that hash;
3. a safe HTTPS gateway and a non-anonymous Horde key are configured.

An unavailable or invalid inventory fails closed to the origin. This avoids an authenticated R2 miss for every
generation checkpoint or not-yet-mirrored auxiliary file.

Before downloading an eligible object, a client exchanges its long-lived Horde key once at `POST /v1/session`.
The gateway validates eligibility and returns a short-lived HMAC-signed token usable only at this mirror. The
token is cached and sent as `Authorization: Bearer`; the Horde key is not attached to every ranged model request.
Legacy direct-key access remains temporarily available for rolling upgrades. Authenticated requests never follow
HTTP redirects, gateway URLs containing credentials/query/fragment are rejected, and non-HTTPS URLs are accepted
only for local development.

The gateway gets one prompt attempt. Any rejection, timeout, or error falls through to the origin, which retains
the normal retry budget. A partial created by the gateway is deleted before origin hand-off; it is never resumed
against a different source. Every completed source is checked against the declared SHA-256.

## Gate and cost controls

The Worker keeps the R2 bucket private. A session exchange resolves `/v2/find_user`, caches the user in KV under
a hash of the API key, and evaluates the deployment policy. The default policy permits trusted users; optional
worker-owner, kudos, and explicit-user paths avoid worker bootstrap deadlocks. Transient Horde API failures are
not cached and simply cause origin fallback.

Object requests are controlled in this order:

1. validate the mirror token (or legacy key) and policy;
2. reject a key already blocked for excessive misses;
3. increment/check the global and per-key/file monthly SQLite Durable Object counters;
4. only then perform an R2 `HEAD` or `GET`;
5. record an R2 miss and block repeated hash spraying when its separate threshold is crossed.

Consequently `GLOBAL_MONTHLY_OP_BUDGET=0` is a real immediate no-R2 kill-switch, and HEAD/miss traffic cannot
bypass the global cost bound. Cloudflare's per-key rate-limit binding provides an additional burst guard. Every
denial remains an origin fallback rather than a failed model download.

The inventory intentionally reveals only content hashes and redistribution notices, not bucket access. Knowing
a hash does not bypass authentication, eligibility, per-key limits, or monthly caps. See the gateway repository's
deployment guide and cost estimator before enabling production traffic.

## Redistribution policy

Hosting third-party weights is redistribution. The sync uses a category-aware, optionally file-scoped policy
whose approvals require license evidence, reviewer identity, and review date. Explicit blocks document known
restrictions; missing decisions are origin-only. The policy, uploaded object metadata, and published attribution
inventory form an auditable chain, but maintainers remain responsible for confirming the legal conclusions.

See [Mirror models to R2](../guides/mirror_models_to_r2.md) for the review and apply workflow.
