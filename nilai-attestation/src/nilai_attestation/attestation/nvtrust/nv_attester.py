#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Attester: Generate an attestation token from local evidence
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
from nv_attestation_sdk import attestation  # type: ignore
import subprocess
from functools import lru_cache
import base64
from nilai_common import Nonce, NVAttestationToken

from nilai_common.logger import setup_logger

logger = setup_logger(__name__)

NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"
OCSP_URL = "https://ocsp.ndis.nvidia.com/"
RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"


@lru_cache(maxsize=1)
def is_nvidia_gpu_available() -> bool:
    """Check if an NVIDIA GPU with compute capability is available in the system and cache the result.

    Returns:
        bool: True if an NVIDIA GPU is available and compute capability is ON, False otherwise.
    """
    try:
        # Run the command and capture its output
        result = subprocess.run(
            ["nvidia-smi", "conf-compute", "-f"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,  # ensures stdout/stderr are strings not bytes
        )

        output = result.stdout.strip()
        if "ON" in output:
            return True
        else:
            return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def nv_attest(nonce: Nonce, name: str = "thisNode1") -> NVAttestationToken:
    """Generate an attestation token from local evidence.

    Args:
        nonce: The nonce to be used for the attestation

    Returns:
        NVAttestationToken: The attestation token response
    """
    # Create and configure the attestation client.
    client = attestation.Attestation()
    client.set_name(name)
    client.set_nonce(nonce)

    logger.info("Checking if NVIDIA GPU is available")
    evidence_list = []
    if is_nvidia_gpu_available():
        logger.info("NVIDIA GPU is available")
        # Configure the remote verifier.
        client.add_verifier(
            attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
        )
        # Collect evidence and perform attestation.
        evidence_list = client.get_evidence()

    else:
        logger.info("NVIDIA GPU is not available")
        client.add_verifier(
            attestation.Devices.GPU,
            attestation.Environment.LOCAL,
            "",
            "",
            OCSP_URL,
            RIM_URL,
        )
        logger.info(f"Using local verifier {client.get_verifiers()}")

        evidence_list = client.get_evidence(options={"no_gpu_mode": True})

    logger.info(f"Evidence list: {evidence_list}")

    # Attestation result
    attestation_result = client.attest(evidence_list)
    logger.info(f"Attestation result: {attestation_result}")
    # Retrieve the attestation token and return it wrapped in our model
    token: str = client.get_token()

    b64_token: NVAttestationToken = base64.b64encode(token.encode("utf-8")).decode(
        "utf-8"
    )
    logger.info(f"Token: {b64_token}")
    return b64_token
