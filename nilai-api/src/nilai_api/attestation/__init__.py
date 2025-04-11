from fastapi import HTTPException
import httpx
from nilai_common import SETTINGS, Nonce, AttestationReport
from nilai_common.logger import setup_logger

logger = setup_logger(__name__)


async def get_attestation_report(
    nonce: Nonce | None,
) -> AttestationReport:
    """Get the attestation report for the given nonce"""

    try:
        attestation_url = f"http://{SETTINGS['attestation_host']}:{SETTINGS['attestation_port']}/attestation/report"

        logger.info(f"Getting attestation report for nonce: {nonce}")
        logger.info(f"Attestation URL: {attestation_url}")

        async with httpx.AsyncClient() as client:
            response: httpx.Response = await client.get(attestation_url, params=nonce)
            report = AttestationReport(**response.json())
            logger.info(f"Attestation report: {report}")
            return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
