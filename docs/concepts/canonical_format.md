# Canonical Format and API Versioning

## Overview

The Model Reference Service provides two API versions (v1 and v2) for managing model records. The **canonical format** setting determines which API version is the authoritative source of truth for data modifications.

This is controlled by the `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT` environment variable.

> **"Canonical format" is not the same as "canonical data."** This page is about
> *which API version* (v1/v2) owns writes - a PRIMARY-only concern that does not
> affect reads. "Canonical data" (in the query/provider docs) means *whose* data
> you're reading - the horde dataset vs. a third-party provider. See the
> [Glossary](../reference/glossary.md#canonical-format).

## Configuration

### Environment Variable

```bash
# V2 format (default) - v2 API has write access
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=v2

# Legacy format - v1 API has write access
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=LEGACY
```

### Interaction with Replicate Mode

| Setting  | Replicate Mode | V1 API     | V2 API     |
| -------- | -------------- | ---------- | ---------- |
| `legacy` | PRIMARY        | Read/Write | Read-Only  |
| `legacy` | REPLICA        | Read-Only  | Read-Only  |
| `v2`     | PRIMARY        | Read-Only  | Read/Write |
| `v2`     | REPLICA        | Read-Only  | Read-Only  |

**Key Points:**

- Only PRIMARY mode instances can have write access
- REPLICA mode instances are always read-only regardless of canonical format
- The canonical format determines which API has write access on PRIMARY

## Backend Info Endpoint

The `/replicate_mode` endpoint returns comprehensive backend configuration information:

```json
{
    "replicate_mode": "PRIMARY",
    "canonical_format": "LEGACY",
    "writable": true
}
```

### Response Fields

| Field              | Type              | Description                                  |
| ------------------ | ----------------- | -------------------------------------------- |
| `replicate_mode`   | `ReplicateMode`   | Either `PRIMARY` or `REPLICA`                |
| `canonical_format` | `CanonicalFormat` | Either `LEGACY` or `V2`                      |
| `writable`         | `boolean`         | Whether the backend accepts write operations |

### Usage in Frontend Clients

Clients should query this endpoint at startup to determine the correct API to use for CRUD operations:

```typescript
detectBackendCapabilities(): Observable<BackendCapabilities> {
  return this.defaultService.replicateModeReplicateModeGet().pipe(
    map((info: BackendInfo) => ({
      writable: info.writable,
      mode: info.replicate_mode === 'PRIMARY' ? 'PRIMARY' : 'REPLICA',
      canonicalFormat: info.canonical_format === 'LEGACY' ? 'legacy' : 'v2',
    })),
  );
}
```

## API Version Differences

### V1 (Legacy) API

- **Path pattern:** `/model_references/v1/{category}`
- **Model format:** Category-specific record types (e.g., `LegacyStableDiffusionRecord`, `LegacyBlipRecord`)
- **Endpoints:** Category-specific create/update/delete methods
- **Write access:** When `canonical_format=LEGACY` and `replicate_mode=PRIMARY`

### V2 API

- **Path pattern:** `/model_references/v2/{category}`
- **Model format:** Unified `ModelRecordUnion` with discriminated unions
- **Endpoints:** Generic CRUD methods accepting any model type
- **Write access:** When `canonical_format=v2` and `replicate_mode=PRIMARY`

## Frontend Routing Logic

When implementing CRUD operations in the frontend, route to the appropriate API based on the canonical format:

```typescript
createModel(category: string, modelData: LegacyRecordUnion): Observable<PendingChangeRecord> {
  const canonicalFormat = this.backendCapabilities().canonicalFormat;

  if (canonicalFormat === 'legacy') {
    // Use category-specific v1 endpoint
    return this.v1Service.createLegacyBlipModel(modelData);
  } else {
    // Use generic v2 endpoint
    return this.v2Service.createV2Model(category, modelData);
  }
}
```

## Data Synchronization

When the canonical format changes, data must be synchronized between formats:

1. **Legacy -> V2:** Export v1 data and import into v2 format
2. **V2 -> Legacy:** Export v2 data and import into legacy format

The canonical format should not be changed while the system is in production without proper migration planning.

## Validation Behavior

### V1 API Validation

- Uses category-specific Pydantic models
- Only validates fields relevant to that category
- Clearer error messages for category-specific field issues

### V2 API Validation

- Uses a discriminated union (`ModelRecordUnion`)
- The `record_type` field determines which model type applies
- Validation errors may mention fields from other model types if the union is not configured correctly

**Important:** If you receive validation errors mentioning fields from the wrong model type (e.g., ImageGeneration fields when creating a BLIP model), ensure you're using the correct API for your backend's canonical format.

## Best Practices

1. **Always query `/replicate_mode` at startup** to determine backend capabilities
2. **Route CRUD operations based on canonical format** - don't assume which API to use
3. **Handle backward compatibility** - older backends may return only `ReplicateMode` instead of full `BackendInfo`
4. **Display appropriate UI** - disable write operations when `writable=false`
5. **Log canonical format** - include it in diagnostics for debugging API mismatches

## Troubleshooting

### "Validation Error" with wrong field names

**Symptom:** Creating a BLIP model fails with errors about `baseline`, `parameters`, or `controlnet_style` fields.

**Cause:** Frontend is using V2 API when backend is configured for legacy format.

**Solution:** Query `/replicate_mode` and ensure the frontend routes to the V1 API when `canonical_format=LEGACY`.

### Write operations return 503

**Symptom:** All create/update/delete operations fail with "Service Unavailable".

**Cause:**

- Backend is in REPLICA mode, OR
- Using the wrong API version for the canonical format

**Solution:**

1. Check `/replicate_mode` response
2. Ensure you're using the API matching the canonical format
3. Only PRIMARY mode backends support writes

### Inconsistent data between V1 and V2 reads

**Symptom:** Reading the same model via V1 and V2 returns different data.

**Cause:** Data was written via one API but the canonical source is the other.

**Solution:** Always write via the canonical format's API. Both APIs read from the same underlying data, so reads should be consistent.

## Environment Variable Reference

| Variable                                 | Values               | Default   | Description                              |
| ---------------------------------------- | -------------------- | --------- | ---------------------------------------- |
| `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT` | `LEGACY`, `v2`       | `v2`      | Which API version is the source of truth |
| `HORDE_MODEL_REFERENCE_REPLICATE_MODE`   | `PRIMARY`, `REPLICA` | `REPLICA` | Whether this instance accepts writes     |

## Related Documentation

- [Glossary](../reference/glossary.md) - Plain-language definitions of replicate mode, canonical data vs. canonical format, and more
- [Model Reference Backend](../reference/model_reference_backend.md) - Backend implementation details
- [Pending Queue Architecture](../reference/pending_queue.md) - Write approval workflow
- [Audit Trail](../reference/audit_trail.md) - Operation logging
