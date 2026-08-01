# Mirror models to R2 (maintainers)

The mirror is an optional, gated accelerator for hostable auxiliary models. Origin URLs remain authoritative.
Install the `r2` dependency group and provide `HORDE_MODEL_REFERENCE_R2__UPLOAD_BUCKET`,
`UPLOAD_ENDPOINT_URL`, `UPLOAD_ACCESS_KEY_ID`, and `UPLOAD_SECRET_ACCESS_KEY` for bucket-aware commands.

## Review candidates

Start with the generated reconciliation report:

```bash
uv run python -m scripts.r2_sync.sync candidates
```

This writes `build/r2_sync/candidates.json`, listing every canonical category/model/file as `approved`,
`blocked`, or `unreviewed`, plus policy entries that no longer match a canonical model. It does not inspect R2
or download model bytes. Exact `--category` and repeatable `--model` filters make focused reviews practical.

Edit `scripts/r2_sync/redistribution_policy.json` to make a decision. An approval must include:

- the exact model-reference `category` and `name`;
- `decision: "approved"`;
- a license expression, review evidence, reviewer, and review date;
- optional exact `files` scope and required attribution text.

Use `decision: "blocked"` plus a reason when redistribution is disallowed or evidence is insufficient. There
is no command-line policy bypass, category-wide approval, or bare-name approval. A new reference therefore
stays origin-only until reviewed, and identical names in different categories cannot grant each other access.

The old `redistributable_allowlist.json` is retained only as migration/audit input and is not an upload policy.
The canonical `controlnet_annotator` API category is treated exactly like every other category; the sync no
longer overlays a second built-in catalog that could hide API drift.

## Plan and reconcile

```bash
uv run python -m scripts.r2_sync.sync plan --cache-dir build/r2_sync/cache -v
uv run python -m scripts.r2_sync.sync reconcile --cache-dir build/r2_sync/cache -v
```

`plan` verifies approved declarations and reports uploads without writing. `reconcile` additionally writes the
inventory that the successful state would publish. With R2 read credentials, both distinguish verified existing
objects from uploads; without them, they deliberately over-report uploads. Omit `--cache-dir` to forbid origin
fetches and use local model roots only.

Filters are intentionally rejected by `apply`: the published inventory must describe the complete approved set,
not make every unselected object disappear from clients. Use filters to investigate, then apply unfiltered.

Existing objects count as present only when their content-addressed key, size, and stored SHA-256 metadata agree.
The uploader always re-hashes actual bytes without trusting checksum sidecar timestamps. Missing or inconsistent
metadata is repaired by an apply.

## Apply atomically

```bash
uv run python -m scripts.r2_sync.sync apply --cache-dir build/r2_sync/cache -v
```

Approved bytes are verified and uploaded to `by-hash/<sha256>` with source, identity, size, license, and review
metadata. The run always writes candidate and SHA-256 backfill reports. If any approved file is missing or has a
hash mismatch, it exits nonzero and does **not** publish a new inventory. Otherwise it uploads the complete
`manifests/current.json` last, so clients see either the previous valid set or the new valid set.

Files whose reference hash is `FIXME` can be uploaded after verification and appear in the backfill report, but
clients remain origin-only until the canonical reference receives that real hash. Apply the report through the
normal reference review path.
