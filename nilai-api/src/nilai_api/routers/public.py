# Fast API and serving
from fastapi import APIRouter

# Internal libraries
from nilai_common import HealthCheckResponse
from nilai_api.state import state

router = APIRouter()


# Health Check Endpoint
@router.get("/v1/health", tags=["Health"])
async def health_check() -> HealthCheckResponse:
    """
    Perform a system health check.

    - **Returns**: Current system health status and uptime

    ### Health Check Details
    - Provides a quick verification of system operational status
    - Reports current system uptime

    ### Status Indicators
    - `status`: Indicates system operational condition
      - `"ok"`: System is functioning normally
    - `uptime`: Duration the system has been running

    ### Example
    ```python
    # Retrieve system health status
    health = await health_check()
    # Expect: HealthCheckResponse(status='ok', uptime=3600)
    ```
    """
    return HealthCheckResponse(status="ok", uptime=state.uptime)
