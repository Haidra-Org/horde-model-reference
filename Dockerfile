# syntax=docker/dockerfile:1

# Multi-stage build for horde-model-reference PRIMARY server
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables for Python and uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Install dependencies using cache and bind mounts for optimal performance
ARG EXTRA_DEPS=""
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    if [ -n "$EXTRA_DEPS" ]; then \
        uv sync --frozen --no-dev --no-editable --no-install-project --extra service --extra "$EXTRA_DEPS"; \
    else \
        uv sync --frozen --no-dev --no-editable --no-install-project --extra service; \
    fi

# Copy the project source code
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Final stage
FROM python:3.12-slim AS final

# Add image labels for metadata
LABEL org.opencontainers.image.title="Horde Model Reference" \
      org.opencontainers.image.description="Model reference service for AI Horde" \
      org.opencontainers.image.source="https://github.com/Haidra-Org/horde-model-reference"

# Install runtime dependencies
# Note: git is required for the GitHub sync service
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and data directory
RUN useradd --system --no-log-init -m -u 1000 horde && \
    mkdir -p /data/horde_model_reference && \
    chown -R horde:horde /data

# Set working directory
WORKDIR /app

# Copy the virtualenv from the builder and ensure it is owned by the runtime user
# NOTE: using --chown requires a recent Docker; fallback below included in case of incompatibility
COPY --from=builder --chown=horde:horde /app/.venv /app/.venv

# Copy source with ownership set to horde
COPY --chown=horde:horde src ./src

# Copy scripts directory (needed for GitHub sync service)
COPY --chown=horde:horde scripts ./scripts

# Ensure logs dir exists and that /app is owned and writable by horde
RUN mkdir -p /app/logs && \
    chown -R horde:horde /app && \
    chmod -R u+rwX /app

# Switch to non-root user
USER horde

# Create data directory volumes
VOLUME ["/data"]

# Expose port
EXPOSE 19800

# Environment variables
ENV HORDE_MODEL_REFERENCE_REPLICATE_MODE=PRIMARY \
    AIWORKER_CACHE_HOME=/data \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:19800/api/heartbeat || exit 1

# Start the FastAPI application using JSON array format for proper signal handling
CMD ["/app/.venv/bin/fastapi", "run", "src/horde_model_reference/service/app.py", "--host", "0.0.0.0", "--port", "19800"]
