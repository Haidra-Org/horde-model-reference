from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from haidra_core.service_base import ContainsMessage, ContainsStatus

import horde_model_reference.service.v1.routers.metadata as v1_metadata
import horde_model_reference.service.v1.routers.references as v1_references
import horde_model_reference.service.v2.routers.metadata as v2_metadata
import horde_model_reference.service.v2.routers.references as v2_references
from horde_model_reference import ReplicateMode
from horde_model_reference.service.shared import v1_prefix, v2_prefix

app = FastAPI(root_path="/api")

origins = [
    "http://localhost:51457",
    "http://localhost:4200",
    "http://localhost:9877",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(v2_references.router, prefix=v2_prefix, tags=["v2"])
app.include_router(v2_metadata.router, prefix=f"{v2_prefix}/metadata", tags=["v2", "metadata"])
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
async def replicate_mode() -> ReplicateMode:
    """Endpoint to get the current replication mode."""
    from horde_model_reference import horde_model_reference_settings

    return horde_model_reference_settings.replicate_mode
