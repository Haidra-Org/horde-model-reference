# Mirror models to R2 (maintainers)

This guide is for ecosystem maintainers. It covers uploading hostable model files to the gated Cloudflare R2
mirror with the `scripts.r2_sync` tool. For *why* the mirror exists and how clients use it, see
[The gated R2 model mirror](../concepts/gated_r2_mirror.md).

## Prerequisites

- The `r2` dependency group: `uv sync --group r2` (provides `boto3`).
- R2 credentials, supplied as settings (env prefix `HORDE_MODEL_REFERENCE_R2__`):

  ```bash
  export HORDE_MODEL_REFERENCE_R2__UPLOAD_BUCKET=horde-model-mirror
  export HORDE_MODEL_REFERENCE_R2__UPLOAD_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
  export HORDE_MODEL_REFERENCE_R2__UPLOAD_ACCESS_KEY_ID=...
  export HORDE_MODEL_REFERENCE_R2__UPLOAD_SECRET_ACCESS_KEY=...
  ```

## 1. Opt models in

Nothing is mirrored until you opt it in **by name** (hosting a weight is redistribution). There is deliberately
no category-wide opt-in: that would silently clear every model later added to the category, whatever its
licence, with nobody reviewing it. Opting a model in is exactly where you record that you reviewed its licence.
Edit `scripts/r2_sync/redistributable_allowlist.json`, listing each model you have confirmed is licence-clear:

```json
{
  "models": [
    "a-terse-entry-by-name",
    { "name": "a-reviewed-model", "license": "Apache-2.0", "note": "upstream README permits redistribution" }
  ]
}
```

An entry is either a bare name string or an object with `name` plus the optional `license` / `note` provenance
fields. When present, the `license` and `note` are stamped onto every object the model uploads, so the bucket
carries the audit trail for why each hosted file is allowed to be there.

### ControlNet annotators

The tool also mirrors the **controlnet-annotator** checkpoints (the `comfyui_controlnet_aux` weights the worker
pre-fetches), read from `horde_model_reference.annotator_catalog`. These are opted in **by HuggingFace repo**
(one repo is one licence-review unit); every horde-exposed annotator currently comes from `lllyasviel/Annotators`,
so a single entry clears them all:

```json
{ "models": ["lllyasviel/Annotators"] }
```

Pass `--no-annotators` to mirror only the model-reference categories, or `--annotator-ckpts-dir` to point at a
local annotator checkpoint directory (it defaults to `<weights-root>/controlnet/annotators`). The catalog ships
each file's hash unset, so the first run computes and backfills them via the same report (paste them into the
catalog).

## 2. Dry-run

The default run uploads nothing; it reports what *would* happen. Run it with R2 credentials present so
"already present" is answered truthfully against the live bucket:

```bash
uv run python -m scripts.r2_sync.sync --dry-run --verbose
```

Without R2 credentials a dry-run cannot see the bucket, so it treats every object as absent and over-reports
uploads (the tool warns loudly when this happens). Note also that a dry-run is not necessarily free: with
`--cache-dir` set it will fetch bytes from origin hosts to hash any still-`FIXME` record (that is the only way
to learn the content address). Omit `--cache-dir` to operate on local files only and never download.

You will see per-file outcomes and totals: `upload`, `already_present`, `skipped_not_allowlisted`,
`missing_bytes`, `hash_mismatch`.

## 3. Apply

```bash
uv run python -m scripts.r2_sync.sync --apply --backfill-report build/r2_backfill.json
```

For each opted-in file the tool:

1. uses your local model copy if present (resolved from the weights root; pass `--weights-root` /
   `--extra-root` to point at it), otherwise downloads it from the origin host (pass `--cache-dir` to cache
   those fetches),
2. verifies/computes its sha256,
3. skips it if the content-addressed object already exists (idempotent), otherwise uploads it to
   `by-hash/<sha256>` with provenance metadata.

## 4. Backfill `FIXME` hashes

Many records still declare `sha256sum: "FIXME"`. The tool computes the real hash for every file it processes and
writes the corrections to the `--backfill-report` JSON. A CI workflow consumes that report to open a pull
request correcting the canonical reference, so the gaps self-heal over time. (Until a hash is backfilled, that
file is served only from origin.)

## Exit code

The tool exits non-zero when any file could not be processed (`missing_bytes` or `hash_mismatch`), so it is safe
to gate a CI job on it.
