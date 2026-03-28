"""FastAPI application factory with lifespan management and CORS configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haidra_core.service_base import ContainsMessage, ContainsStatus
from loguru import logger

import horde_model_reference.service.statistics.routers.deletion_risk as ref_deletion_risk
import horde_model_reference.service.statistics.routers.statistics as ref_statistics
import horde_model_reference.service.v1.routers.metadata as v1_metadata
import horde_model_reference.service.v1.routers.pending_queue as v1_pending_queue
import horde_model_reference.service.v1.routers.pending_queue_audit as v1_pending_queue_audit
import horde_model_reference.service.v1.routers.references as v1_references
import horde_model_reference.service.v2.routers.metadata as v2_metadata
import horde_model_reference.service.v2.routers.pending_queue as v2_pending_queue
import horde_model_reference.service.v2.routers.pending_queue_audit as v2_pending_queue_audit
import horde_model_reference.service.v2.routers.references as v2_references
import horde_model_reference.service.v2.routers.search as v2_search
import horde_model_reference.service.v2.routers.user as v2_user
from horde_model_reference import BackendInfo, ReplicateMode, horde_model_reference_settings
from horde_model_reference.service.shared import statistics_prefix, v1_prefix, v2_prefix


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application lifespan events.

    Starts background cache hydration on startup and stops it on shutdown.
    """
    # Startup
    if horde_model_reference_settings.cache_hydration_enabled:
        from horde_model_reference.analytics.cache_hydrator import get_cache_hydrator

        hydrator = get_cache_hydrator()
        logger.info("Starting cache hydration on application startup...")
        await hydrator.start()

    yield

    # Shutdown
    from horde_model_reference.service.shared import httpx_client

    await httpx_client.aclose()

    if horde_model_reference_settings.cache_hydration_enabled:
        from horde_model_reference.analytics.cache_hydrator import get_cache_hydrator

        hydrator = get_cache_hydrator()
        logger.info("Stopping cache hydration on application shutdown...")
        await hydrator.stop()


app = FastAPI(root_path="/api", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,  # ty:ignore[invalid-argument-type] - This is idiomatic usage of CORSMiddleware in FastAPI, which expects these arguments.
    allow_origins=horde_model_reference_settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(v2_search.router, prefix=v2_prefix, tags=["v2", "search"])
app.include_router(v2_pending_queue.router, prefix=v2_prefix, tags=["v2", "pending_queue"])
app.include_router(v2_pending_queue_audit.router, prefix=v2_prefix, tags=["v2", "pending_queue", "audit"])
app.include_router(v2_user.router, prefix=v2_prefix, tags=["v2", "user"])
app.include_router(v2_references.router, prefix=v2_prefix, tags=["v2"])
app.include_router(ref_statistics.router, prefix=statistics_prefix, tags=["v2", "statistics"])
app.include_router(ref_deletion_risk.router, prefix=statistics_prefix, tags=["v2", "deletion-risk"])
app.include_router(v2_metadata.router, prefix=f"{v2_prefix}/metadata", tags=["v2", "metadata"])
app.include_router(v1_pending_queue.router, prefix=v1_prefix, tags=["v1", "pending_queue"])
app.include_router(v1_pending_queue_audit.router, prefix=v1_prefix, tags=["v1", "pending_queue", "audit"])
app.include_router(v1_references.router, prefix=v1_prefix)
app.include_router(v1_metadata.router, prefix=f"{v1_prefix}/metadata", tags=["v1", "metadata"])


@app.get("/")
async def read_root() -> ContainsMessage:
    """Root endpoint for the Horde Model Reference API1."""
    return ContainsMessage(
        message="Welcome to the Horde Model Reference API Check `/api/docs` for documentation.",
    )


@app.get("/heartbeat")
async def heartbeat() -> ContainsStatus:
    """Heartbeat endpoint to check the service status."""
    return ContainsStatus(status="ok")


@app.get("/replicate_mode")
async def replicate_mode() -> BackendInfo:
    """Get backend configuration and capabilities.

    Returns information about the backend's replication mode, canonical format,
    and whether write operations are supported.

    Clients should use this endpoint on startup to determine:
    - Whether the backend supports write operations (writable=True)
    - Which API version to use for CRUD operations (based on canonical_format)

    Note: For backward compatibility, this endpoint path is retained but now
    returns a richer BackendInfo response instead of just the ReplicateMode.
    """
    from horde_model_reference import horde_model_reference_settings

    # Map the string setting to the enum
    canonical_format = horde_model_reference_settings.canonical_format

    return BackendInfo(
        replicate_mode=horde_model_reference_settings.replicate_mode,
        canonical_format=canonical_format,
        writable=horde_model_reference_settings.replicate_mode == ReplicateMode.PRIMARY,
    )
