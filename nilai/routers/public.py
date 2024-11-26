# Fast API and serving
from fastapi import APIRouter

# Internal libraries
from nilai.model import HealthCheckResponse
from nilai.state import state

router = APIRouter()


# Health Check Endpoint
@router.get("/v1/health", tags=["Health"])
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", uptime=state.uptime)
