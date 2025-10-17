from fastapi import HTTPException
import httpx
from nilai_common import AttestationReport
from nilai_common.logger import setup_logger

logger = setup_logger(__name__)

ATTESTATION_URL = "http://nilcc-attester/v2/report"


async def get_attestation_report() -> AttestationReport:
    """Get the attestation report"""

    try:
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.get(ATTESTATION_URL)
            response_json = response.json()
            return AttestationReport(
                gpu_attestation=response_json["report"],
                cpu_attestation=response_json["gpu_token"],
                verifying_key="",  # Added later by the API
            )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=str("Error getting attestation report" + str(e)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str("Error getting attestation report" + str(e))
        )
