# Deployment Guide

Deploy a Horde Model Reference PRIMARY server using Docker or Python directly.

## Table of Contents

- [Deployment Guide](#deployment-guide)
    - [Table of Contents](#table-of-contents)
    - [Prerequisites](#prerequisites)
    - [Docker Deployment](#docker-deployment)
        - [Single-Worker Docker Setup](#single-worker-docker-setup)
        - [Multi-Worker Docker Setup (Production)](#multi-worker-docker-setup-production)
            - [Optional: GitHub Sync Service\*](#optional-github-sync-service)
    - [Non-Docker Deployment](#non-docker-deployment)
        - [Single-Worker Setup](#single-worker-setup)
        - [Multi-Worker Setup (Production)](#multi-worker-setup-production)
    - [First-Time Setup](#first-time-setup)
        - [Option 1: Auto-seed from GitHub (recommended)](#option-1-auto-seed-from-github-recommended)
        - [Option 2: Provide files manually](#option-2-provide-files-manually)
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

## Docker Deployment

### Single-Worker Docker Setup

For development, testing, or low-traffic production use.

**1. (Optional) Configure environment variables:**

Create a `.env` file if you need custom settings:

```bash
cp .env.example .env
# Edit .env as needed
```

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

### Multi-Worker Docker Setup (Production)

For high-traffic production deployments with distributed caching.

**1. Configure environment (if needed):**

```bash
cp .env.example .env
# Edit .env to customize Redis URL, worker count, etc.
```

**2. Start with Redis:**

```bash
docker-compose -f docker-compose.redis.yml up -d
```

**3. Verify Redis connection:**

```bash
docker-compose -f docker-compose.redis.yml logs horde-model-reference | grep -i redis
# Should show successful Redis connection
```

See `docker-compose.redis.yml` for worker count and Redis configuration options.

#### Optional: GitHub Sync Service*

Automatically sync model references to GitHub legacy repositories. See [SYNC_README.md](SYNC_README.md) for setup instructions.

```bash
# After configuring .env.sync
docker-compose --profile sync up -d
```

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
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy # assuming the v2 transition hasn't happened yet
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
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy # assuming the v2 transition hasn't happened yet
fastapi dev src/horde_model_reference/service/app.py --port 19800

# Windows PowerShell
$env:HORDE_MODEL_REFERENCE_REPLICATE_MODE="PRIMARY"
$env:HORDE_MODEL_REFERENCE_MAKE_FOLDERS="true"
$env:HORDE_MODEL_REFERENCE_CANONICAL_FORMAT="legacy" # assuming the v2 transition hasn't happened yet
fastapi dev src/horde_model_reference/service/app.py --port 19800
```

---

### Multi-Worker Setup (Production)

**1. Install Redis:**

```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis-server

# macOS
brew install redis
brew services start redis

# Windows: Use WSL or download from https://github.com/microsoftarchive/redis/releases
```

**2. Configure environment:**

Create a `.env` file:

```bash
HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY
HORDE_MODEL_REFERENCE_CANONICAL_FORMAT=legacy # assuming the v2 transition hasn't happened yet
HORDE_MODEL_REFERENCE_MAKE_FOLDERS=true
HORDE_MODEL_REFERENCE_REDIS_USE_REDIS=true
HORDE_MODEL_REFERENCE_REDIS_URL=redis://localhost:6379/0
```

**3. Start the server:**

```bash
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

fastapi run src/horde_model_reference/service/app.py --host 0.0.0.0 --port 19800 --workers 4
```

**Note:** Use `fastapi dev` (auto-reload) for development, `fastapi run` for production.

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

See `.env.example` for all options. Common settings:

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
