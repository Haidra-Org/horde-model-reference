from fastapi import FastAPI
from haidra_core.service_base import ContainsMessage, ContainsStatus

import horde_model_reference.service.v1.routers.references as v1_references
import horde_model_reference.service.v2.routers.references as v2_references
from horde_model_reference import ReplicateMode
from horde_model_reference.service.shared import v1_prefix, v2_prefix

app = FastAPI(root_path="/api")


app.include_router(v1_references.router, prefix=v1_prefix, tags=["model_reference_v1"])
app.include_router(v2_references.router, prefix=v2_prefix, tags=["model_reference_v2"])


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
