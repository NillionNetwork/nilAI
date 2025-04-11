# Fast API and serving
import logging
from fastapi import APIRouter

# Internal libraries
from nilai_attestation.attestation import (
    get_attestation_report,
    verify_attestation_report,
)
from nilai_common import (
    AttestationReport,
    Nonce,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/attestation/report", tags=["Attestation"])
async def get_attestation(nonce: Nonce | None = None) -> AttestationReport:
    """
    Generate a cryptographic attestation report.

    - **nonce**: Optional nonce for the attestation (query parameter)
    - **Returns**: Attestation details for service verification

    ### Attestation Details
    - `cpu_attestation`: CPU environment verification
    - `gpu_attestation`: GPU environment verification

    ### Security Note
    Provides cryptographic proof of the service's integrity and environment.
    """
    return get_attestation_report(nonce)


@router.post("/attestation/verify", tags=["Attestation"])
async def post_attestation(attestation_report: AttestationReport) -> bool:
    """
    Verify a cryptographic attestation report.

    - **attestation_report**: Attestation report to verify
    - **Returns**: True if the attestation report is valid, False otherwise
    """
    return verify_attestation_report(attestation_report)
