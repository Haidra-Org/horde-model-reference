# Licensing as First-Class Data

Licensing is part of the v2 data contract, not prose attached to a model. The normalized design separates three
records so consumers can answer common policy questions without parsing Markdown:

| Record | Storage | Purpose |
| --- | --- | --- |
| `LicenseDefinition` | `licensing/licenses.json` | One reusable license identity, canonical URL, permission defaults, obligations, and restrictions. |
| `ModelLicensing` | Each canonical v2 model record | The reviewed conclusion for that model, including evidence, attribution, and optional exact-file overrides. |
| `LicensedAsset` | `licensing/assets.json` | Licensing for non-model assets such as ComfyUI and custom nodes. |

`license_ids` act as foreign keys into the definition catalog. The service rejects assignments to missing or
deprecated definitions and refuses to delete a referenced definition. Definitions and non-model assets carry
revision and editor metadata; direct mutations also append an event to `licensing/audit.jsonl`.

## Permission conclusions

`commercial_use` and `redistribution` are separate fields with four possible values:

- `allowed`
- `allowed_with_conditions`
- `prohibited`
- `unknown`

`allowed_with_conditions` means consumers must inspect `obligations`, `restrictions`, `attribution`, and evidence;
it is not equivalent to unconditional permission. `unknown` is deliberately fail-closed. Unaudited legacy records
are returned as `license_expression: NOASSERTION`, with both permission fields set to `unknown`.

A model-level assignment is the default for every declared download. `licensing.files` can override that conclusion
for a file, keyed by the exact `config.download[].file_name`. Record validation rejects dangling override keys.

```json
{
  "licensing": {
    "license_expression": "Apache-2.0",
    "license_ids": ["Apache-2.0"],
    "commercial_use": "allowed",
    "redistribution": "allowed_with_conditions",
    "obligations": ["include_license"],
    "evidence": [{"source": "https://example.org/model/LICENSE"}],
    "reviewed_by": "maintainer-id",
    "reviewed_at": "2026-08-01",
    "files": {}
  }
}
```

## Write and rollout model

Model changes continue through the pending queue, so licensing changes receive the same review and critical-diff
treatment as other model metadata. Definitions and non-model assets use direct PRIMARY-only CRUD guarded by the
independent `licensing.editor_ids` allowlist. This keeps licensing authority separate from queue approval authority.

During migration, a model create that omits `licensing` receives an explicit `NOASSERTION` record. After the PRIMARY
has been backfilled, set `HORDE_MODEL_REFERENCE_LICENSING__REQUIRE_EXPLICIT_MODEL_ASSIGNMENTS=true` to reject such
creates. Configure direct editors as a JSON list with
`HORDE_MODEL_REFERENCE_LICENSING__EDITOR_IDS='["123", "456"]'`.

HTTP-backed replicas refresh the normalized definition and auxiliary-asset snapshot from PRIMARY on first access.
The packaged JSON snapshot remains available when PRIMARY cannot be reached; model assignments remain embedded in
the ordinary v2 records and follow the existing backend cache path.

## Migration and generated documentation

The one-time backfill reads only the canonical PRIMARY v2 API. It never reads from or writes to the legacy Git
repositories:

```bash
uv run python scripts/migrate_model_licensing.py \
  --primary-api-url https://models.aihorde.net/api \
  --output review_artifacts/model-licensing-backfill.json
```

Review the manifest, then repeat with `--submit --apikey ...` to queue the generated model updates. Every model not
covered by reviewed migration evidence receives an explicit unknown conclusion.

Markdown is a generated view, never the source of truth:

```bash
uv run generate-license-reference \
  --primary-api-url https://models.aihorde.net/api \
  --output-directory review_artifacts/generated-licenses
```

The generator can also consume a saved `/licensing/export` response with `--input`. Output is deterministic and
starts with a generated-file warning.

See [Licensing endpoints](../reference/http_api/licensing_endpoints.md) for the consumer API.
