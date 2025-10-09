from fastapi import FastAPI
from haidra_core.service_base import ContainsMessage, ContainsStatus

import horde_model_reference.service.v1.routers.references as v1_references
import horde_model_reference.service.v2.routers.references as v2_references

app = FastAPI(root_path="/api")

app.include_router(v1_references.router, prefix="/model_references/v1", tags=["model_reference_v1"])
app.include_router(v2_references.router, prefix="/model_references/v2", tags=["model_reference_v2"])


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
