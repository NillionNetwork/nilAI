from fastapi import HTTPException
import httpx
from nilai_common import Nonce, AttestationReport, SETTINGS
from nilai_common.logger import setup_logger

logger = setup_logger(__name__)


async def get_attestation_report(
    nonce: Nonce | None,
) -> AttestationReport:
    """Get the attestation report for the given nonce"""

    try:
        attestation_url = f"http://{SETTINGS.attestation_host}:{SETTINGS.attestation_port}/attestation/report"
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.get(attestation_url, params=nonce)
            report = AttestationReport(**response.json())
            return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def verify_attestation_report(attestation_report: AttestationReport) -> bool:
    """Verify the attestation report"""
    try:
        attestation_url = f"http://{SETTINGS.attestation_host}:{SETTINGS.attestation_port}/attestation/verify"
        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.get(
                attestation_url, params=attestation_report.model_dump()
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
