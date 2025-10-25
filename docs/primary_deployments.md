# Primary Deployments

## Transitioning to a new model reference system

Historically, models have been managed via GitHub repositories ([image models](https://github.com/Haidra-Org/AI-Horde-image-model-reference), [text models](https://github.com/Haidra-Org/AI-Horde-text-model-reference)). This approach has had limitations which are mitigated by github actions and manual review, but a more robust solution is needed for scaling to new model categories and more frequent updates.

The nature of the horde is such that we have many (many) third-party integrations which have hardcoded references to these github repositories. To avoid breaking these integrations, we are introducing a new model reference system which supports both the legacy github format and a new v2 format, while also providing a REST API for model reference access. Further, until we completely deprecate the github repositories, the new system will keep the github repositories in sync with the new system. You can see more details about that in the [sync readme in the scripts folder](../scripts/SYNC_README.md).

Adopting the v1 API is not recommended for new integrations, as it will eventually be deprecated. However, existing integrations and drop-in replace their references to github with calls to the v1 API without any other changes. Legacy filenames (`stable_diffusion.json` for image, `db.json` for text) are supported and the returned data is in the same format, order, etc, as the github repositories.

## Service Architecture Overview

### PRIMARY vs REPLICA Mode

The package has two operational modes:

#### **PRIMARY Mode** (Server)

> You only need this if you are deploying your own Horde

- Authoritative source for model references
- Supports CRUD operations (Create, Read, Update, Delete)
- Can use Redis for distributed caching across multiple workers
- Optionally seeds initial data from GitHub legacy repositories
- Serves REPLICA clients via REST API

#### **REPLICA Mode** (Client)

- Fetches model references from PRIMARY API or GitHub
- Read-only access
- Local file-based caching with TTL
- Automatic GitHub fallback if PRIMARY is unavailable
- Used by workers, clients, and integrations

**Configuration:**

```bash
# REPLICA mode (default) - for workers/clients
export HORDE_MODEL_REFERENCE_REPLICATE_MODE=REPLICA
export HORDE_MODEL_REFERENCE_PRIMARY_API_URL=https://aihorde.net/api

# PRIMARY mode - for server deployment
export HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY
export HORDE_MODEL_REFERENCE_REDIS_USE_REDIS=true  # for multi-worker
```

### ⚠️ Backend Architecture

> **REVIEW MARKER**: The backend system is complex; consider if this needs more/less detail.

The package uses a pluggable backend system:

| Backend | Mode | Purpose | Use Case |
|---------|------|---------|----------|
| **FileSystemBackend** | PRIMARY | Direct file I/O | Single-worker PRIMARY server |
| **RedisBackend** | PRIMARY | Distributed cache wrapper | Multi-worker PRIMARY server |
| **GitHubBackend** | REPLICA | GitHub downloads | REPLICA without PRIMARY API |
| **HTTPBackend** | REPLICA | PRIMARY API + fallback | REPLICA with PRIMARY API (recommended) |

**Backend Selection (Automatic):**

```python
# Determined by environment variables:
# PRIMARY + Redis → RedisBackend(FileSystemBackend)
# PRIMARY + No Redis → FileSystemBackend
# REPLICA + primary_api_url → HTTPBackend
# REPLICA + No primary_api_url → GitHubBackend
```

### ⚠️ Canonical Format Architecture

> **REVIEW MARKER**: The dual-format system is potentially confusing; consider rewording.

The package supports two file formats:

- **`legacy`**: Original GitHub repository format (flat dictionary, single JSON file per category)
- **`v2`**: New standardized format (enhanced metadata, schema versioning)

The `CANONICAL_FORMAT` setting determines which format is authoritative:

```bash
# v2 format (default/recommended)
export HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=v2
# - v2 API has CRUD operations
# - v1 API is read-only (serves converted data)

# legacy format (for backward compatibility)
export HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy
# - v1 API has CRUD operations
# - v2 API is read-only (serves converted data)
```

**Both formats can be read by both API versions** - this enables gradual migration.

### Model Categories

The package manages multiple model categories:

```python
from horde_model_reference import MODEL_REFERENCE_CATEGORY

print(list(MODEL_REFERENCE_CATEGORY))
# Output:
# - image_generation: Stable Diffusion, FLUX, etc.
# - text_generation: LLMs (LLaMA, GPT, etc.)
# - clip: Text-image embedding models
# - controlnet: Image control models (canny, depth, etc.)
# - blip: Image captioning models
# - esrgan: Image upscaling models
# - gfpgan: Face restoration models
# - codeformer: Face restoration models
# - safety_checker: NSFW detection models
# - video_generation: Video generation models (future)
# - audio_generation: Audio generation models (future)
# - miscellaneous: Other models
```
