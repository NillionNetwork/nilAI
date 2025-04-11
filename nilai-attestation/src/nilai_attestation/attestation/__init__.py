from functools import lru_cache
from nilai_common import AttestationReport, Nonce

from nilai_attestation.attestation.sev.sev import sev
from nilai_attestation.attestation.nvtrust.nv_attester import nv_attest
from nilai_common.logger import setup_logger

logger = setup_logger(__name__)


@lru_cache(maxsize=1)
def load_sev_library() -> bool:
    """Load the SEV library"""
    return sev.init()


def get_attestation_report(nonce: Nonce | None = None) -> AttestationReport:
    """Get the attestation report for the given nonce"""

    # Since Nonce is an Annotated[str], we can use it directly
    attestation_nonce: Nonce = "0" * 64 if nonce is None else nonce

    logger.info(f"Nonce: {attestation_nonce}")

    load_sev_library()
    return AttestationReport(
        nonce=attestation_nonce,
        verifying_key="",
        cpu_attestation=sev.get_quote(),
        gpu_attestation=nv_attest(attestation_nonce),
    )


if __name__ == "__main__":
    nonce = "0" * 64
    report = get_attestation_report(nonce)
    print(report)
