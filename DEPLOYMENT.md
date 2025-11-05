# Deployment Guide

Deploy a Horde Model Reference PRIMARY server using Docker or Python directly.

## Table of Contents

- [Deployment Guide](#deployment-guide)
    - [Table of Contents](#table-of-contents)
    - [Prerequisites](#prerequisites)
    - [First-Time Setup](#first-time-setup)
        - [Option 1: Auto-seed from GitHub (recommended)](#option-1-auto-seed-from-github-recommended)
        - [Option 2: Provide files manually](#option-2-provide-files-manually)
    - [Non-Docker Deployment](#non-docker-deployment)
        - [Single-Worker Setup](#single-worker-setup)
    - [Docker Deployment](#docker-deployment)
        - [Single-Worker Docker Setup](#single-worker-docker-setup)
    - [Optional: GitHub Sync Service](#optional-github-sync-service)
    - [Verification](#verification)
    - [Configuration Reference](#configuration-reference)
    - [Troubleshooting](#troubleshooting)
    - [Production Notes](#production-notes)

---

## Prerequisites

**All deployments:**

- Model reference files OR enable GitHub seeding (see [First-Time Setup](#first-time-setup))

**Docker deployments:**

- [Docker](https://docs.docker.com/get-docker/) 20.10+
- [Docker Compose](https://docs.docker.com/compose/install/) 1.29+

**Non-Docker deployments:**

- Python 3.10+
- Git

**Multi-worker (production) deployments:**

- Redis server

---

## First-Time Setup

If this is your first deployment and you don't have model reference files:

### Option 1: Auto-seed from GitHub (recommended)

Add to your `.env` file or docker-compose environment:

```bash
HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED=true
```

The server will download and convert model references on first startup. **Set back to `false` after initial startup.**

### Option 2: Provide files manually

Place JSON files in `{AIWORKER_CACHE_HOME}/horde_model_reference/`:

- `stable_diffusion.json` - Image generation models
- `text_generation.json` - Text generation models
- Other category files as needed

## Non-Docker Deployment

### Single-Worker Setup

**1. Install the project:**

```bash
git clone https://github.com/Haidra-org/horde-model-reference.git
cd horde-model-reference

# Using uv (recommended)
uv sync .

# OR using pip
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install .
```

**2. Configure environment:**

Create a `.env` file:

```bash
HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY
HORDE_MODEL_REFERENCE_MAKE_FOLDERS=true
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy
```

Or use `.env.primary.example` as a template:

```bash
cp .env.primary.example .env
# Edit .env as needed
```

**3. Start the server:**

```bash
# Activate venv first (if using pip)
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Start server
fastapi dev src/horde_model_reference/service/app.py --port 19800
```

**Alternative: Set environment variables inline:**

```bash
# Linux/macOS
export HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY
export HORDE_MODEL_REFERENCE_MAKE_FOLDERS=true
export HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy
fastapi dev src/horde_model_reference/service/app.py --port 19800

# Windows PowerShell
$env:HORDE_MODEL_REFERENCE_REPLICATE_MODE="PRIMARY"
$env:HORDE_MODEL_REFERENCE_MAKE_FOLDERS="true"
$env:HORDE_MODEL_REFERENCE_CANONICAL_FORMAT="legacy"
fastapi dev src/horde_model_reference/service/app.py --port 19800
```

---

## Docker Deployment

### Single-Worker Docker Setup

For development, testing, or low-traffic production use.

**1. (Optional) Configure environment variables:**

Create a `.env.primary` file if you need custom settings:

```bash
cp .env.primary.example .env.primary
# Edit .env.primary as needed
```

Note: The default docker-compose.yml loads `.env.primary` automatically.

**2. Start the server:**

```bash
docker-compose up -d
```

**3. Verify:**

```bash
curl http://localhost:19800/api/heartbeat
# Should return: {"status":"ok"}
```

**Common commands:**

```bash
# View logs
docker-compose logs -f horde-model-reference

# Stop server
docker-compose down

# Restart after config changes
docker-compose restart
```

---

## Optional: GitHub Sync Service

Automatically sync model references to GitHub legacy repositories. See [SYNC_README.md](scripts/sync/README.md) for setup instructions.

---

## Verification

**Check server health:**

```bash
curl http://localhost:19800/api/heartbeat
# Expected: {"status":"ok"}
```

**View API documentation:**

Open <http://localhost:19800/docs> in your browser

**Test model references:**

```bash
curl http://localhost:19800/api/model_references/v2/image_generation
```

**Verify Redis (multi-worker only):**

```bash
redis-cli ping
# Expected: PONG
```

---

## Configuration Reference

See `.env.example` for all available options, or `.env.primary.example` for PRIMARY-specific configuration. Common settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `HORDE_MODEL_REFERENCE_REPLICATE_MODE` | `REPLICA` | Set to `PRIMARY` for server mode |
| `HORDE_MODEL_REFERENCE_MAKE_FOLDERS` | `false` | Auto-create directories |
| `HORDE_MODEL_REFERENCE_CANONICAL_FORMAT` | `legacy` | Use legacy format (pre-v2 transition) |
| `HORDE_MODEL_REFERENCE_REDIS_USE_REDIS` | `false` | Enable Redis (multi-worker) |
| `HORDE_MODEL_REFERENCE_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS` | `60` | Cache lifetime |
| `HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED` | `false` | Auto-seed on first start |

---

## Troubleshooting

**Server won't start:**

```bash
# Check logs
docker-compose logs horde-model-reference  # Docker
# or check terminal output for non-Docker

# Common fixes:
# - Port conflict: Add PORT=9000 to .env
# - Missing files: Enable HORDE_MODEL_REFERENCE_GITHUB_SEED_ENABLED=true
```

**Redis connection failures (multi-worker):**

```bash
# Verify Redis
redis-cli ping  # Should return: PONG

# Check Docker network (if using Docker)
docker-compose -f docker-compose.redis.yml ps

# Update Redis URL in .env if needed
```

**Performance issues:**

- Single-worker too slow? Switch to multi-worker with Redis
- Multi-worker not faster? Verify `HORDE_MODEL_REFERENCE_REDIS_USE_REDIS=true` is set
- Increase `HORDE_MODEL_REFERENCE_CACHE_TTL_SECONDS` for longer caching

---

## Production Notes

**Security:**

- Use a reverse proxy (nginx/Caddy) for HTTPS
- Keep Redis port 6379 internal-only
- Use Redis password: `redis://:password@redis:6379/0`

**Monitoring:**

- Set up health checks to `/api/heartbeat`
- Monitor Redis memory and connections
- Aggregate logs (ELK, Loki, etc.)

**Backup:**

- Model reference files: `{AIWORKER_CACHE_HOME}/horde_model_reference/`
- Redis data (optional - it's cache only)

**Scaling:**

- Use shared Redis for all instances
- Share model files via NFS/S3
- Use load balancer for traffic distribution

**Support:**

- [GitHub Issues](https://github.com/Haidra-Org/horde-model-reference/issues)
- [Discussions](https://github.com/Haidra-Org/horde-model-reference/discussions)
- [Discord](https://discord.gg/3DxrhksKzn)
