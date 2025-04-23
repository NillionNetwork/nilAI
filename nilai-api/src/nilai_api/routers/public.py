# Fast API and serving
from fastapi import APIRouter
from nilai_api.state import state

# Internal libraries
from nilai_common import HealthCheckResponse, AttestationReport
from nilai_api.attestation import verify_attestation_report

router = APIRouter()


@router.get("/v1/public_key", tags=["Public"])
async def get_public_key() -> str:
    """
    Get the public key of the API.
    """
    return state.b64_public_key


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


@router.post("/attestation/verify", tags=["Attestation"])
async def post_attestation(attestation_report: AttestationReport) -> bool:
    """
    Verify a cryptographic attestation report.

    - **attestation_report**: Attestation report to verify
    - **Returns**: True if the attestation report is valid, False otherwise
    """
    return await verify_attestation_report(attestation_report)
