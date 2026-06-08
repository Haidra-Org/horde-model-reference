"""FastAPI application factory with lifespan management and CORS configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haidra_core.service_base import ContainsMessage
from loguru import logger
from pydantic import BaseModel

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
import horde_model_reference.service.v2.routers.text_utils as v2_text_utils
import horde_model_reference.service.v2.routers.user as v2_user
from horde_model_reference import BackendInfo, ReplicateMode, horde_model_reference_settings
from horde_model_reference.http_retry import horde_api_circuit_breaker
from horde_model_reference.service.shared import statistics_prefix, v1_prefix, v2_prefix


class AIHordeStatus(BaseModel):
    """Status of the external AI Horde API connection."""

    degraded: bool
    consecutive_failures: int
    seconds_until_retry: float | None


class HeartbeatResponse(BaseModel):
    """Enhanced heartbeat response with external service status."""

    status: str
    ai_horde: AIHordeStatus


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


try:
    _SERVICE_VERSION = version("horde_model_reference")
except PackageNotFoundError:  # pragma: no cover - only when running from a non-installed checkout
    _SERVICE_VERSION = "0.0.0+unknown"


_API_DESCRIPTION = """
The **Horde Model Reference API** is the authoritative source of AI model metadata for the
[AI-Horde](https://aihorde.net) ecosystem. It serves the curated lists of image, text, and
utility models (CLIP, ControlNet, ESRGAN, …) that workers download and that clients display.

### Who uses this API

- **Workers & clients** read model references - either directly over HTTP or via the
  `horde-model-reference` Python library running in REPLICA mode (which calls this same API,
  falling back to GitHub if the PRIMARY is unreachable).
- **The AI-Horde backend** runs this service in PRIMARY mode at
  [`models.aihorde.net`](https://models.aihorde.net/api/docs) as the canonical source.

### Two API versions

- **v2** (`/model_references/v2`) - the current format, with search, per-model retrieval,
  statistics, and the full text-model grouping toolkit. Prefer this for new integrations.
- **v1** (`/model_references/v1`) - the legacy GitHub-compatible format, retained unchanged for
  backward compatibility with existing AI-Horde workers.

Both versions are readable regardless of deployment configuration. **Reads are open; writes are
not.** Write operations require a PRIMARY deployment and a valid `apikey`, and they are not
applied immediately - they enter a [pending queue](https://models.aihorde.net/api/docs) for
two-person review (propose -> approve -> apply).

### Discovering capabilities

Call [`GET /replicate_mode`](#operations-default-replicate_mode_replicate_mode_get) on startup to
learn whether an instance is writable and which canonical format it serves.

Full documentation, tutorials, and guides: <https://github.com/Haidra-Org/horde-model-reference>
"""


_OPENAPI_TAGS = [
    {
        "name": "v2",
        "description": "Current model-reference format: reads, CRUD, per-model retrieval, and metadata.",
    },
    {
        "name": "v1",
        "description": "Legacy GitHub-compatible format, retained unchanged for existing AI-Horde workers.",
    },
    {
        "name": "search",
        "description": "Filter, sort, and paginate models within a category or across all categories.",
    },
    {
        "name": "statistics",
        "description": "Aggregated per-category counts, baseline/tag distributions, and download statistics.",
    },
    {
        "name": "deletion-risk",
        "description": "Live-usage-informed risk analysis identifying models that are candidates for removal.",
    },
    {
        "name": "text_utils",
        "description": (
            "Text-generation grouping toolkit: name parsing/composition, groups, aliases, "
            "families, and naming schemas."
        ),
    },
    {
        "name": "pending_queue",
        "description": "Propose -> approve -> apply workflow for model changes on PRIMARY deployments.",
    },
    {
        "name": "audit",
        "description": "Read-only history of pending-queue batches and their net effect.",
    },
    {
        "name": "metadata",
        "description": "Per-category last-updated timestamps for change detection by REPLICA clients.",
    },
    {
        "name": "user",
        "description": "Authenticated user identity and pending-queue roles (requestor/approver).",
    },
]


app = FastAPI(
    root_path="/api",
    lifespan=lifespan,
    title="Horde Model Reference API",
    summary="Authoritative AI model metadata for the AI-Horde ecosystem.",
    description=_API_DESCRIPTION,
    version=_SERVICE_VERSION,
    openapi_tags=_OPENAPI_TAGS,
    contact={
        "name": "Haidra-Org / AI-Horde",
        "url": "https://github.com/Haidra-Org/horde-model-reference",
    },
    license_info={
        "name": "AGPL-3.0",
        "url": "https://www.gnu.org/licenses/agpl-3.0.en.html",
    },
    servers=[
        {"url": "https://models.aihorde.net/api", "description": "Public PRIMARY deployment"},
        {"url": "http://localhost:19800/api", "description": "Local development server"},
    ],
)

app.add_middleware(
    CORSMiddleware,  # ty:ignore[invalid-argument-type] - This is idiomatic usage of CORSMiddleware in FastAPI, which expects these arguments.
    allow_origins=horde_model_reference_settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(v2_text_utils.router, prefix=v2_prefix, tags=["v2", "text_utils"])
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


@app.get("/", summary="API landing message", tags=["default"])
async def read_root() -> ContainsMessage:
    """Return a welcome message pointing to the interactive documentation."""
    return ContainsMessage(
        message="Welcome to the Horde Model Reference API. See `/api/docs` for interactive documentation.",
    )


@app.get("/heartbeat", summary="Service health check", tags=["default"])
async def heartbeat() -> HeartbeatResponse:
    """Heartbeat endpoint to check the service status.

    Returns overall service status and the state of the external AI Horde API
    connection. When the AI Horde API is unreachable, ``ai_horde.degraded`` is
    ``True`` and ``ai_horde.seconds_until_retry`` indicates when the next probe
    request will be attempted.
    """
    cb_status = horde_api_circuit_breaker.get_status_dict()
    return HeartbeatResponse(
        status="ok",
        ai_horde=AIHordeStatus(
            degraded=cb_status["degraded"],
            consecutive_failures=cb_status["consecutive_failures"],
            seconds_until_retry=cb_status["seconds_until_retry"],
        ),
    )


@app.get("/replicate_mode", summary="Backend capabilities probe", tags=["default"])
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
