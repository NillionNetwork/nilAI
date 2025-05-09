#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Verifier: Validate an attestation token against a remote policy
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
from nilai_common.api_model import AttestationReport
from nv_attestation_sdk import attestation  # type: ignore
import json
import base64
from nilai_common.logger import setup_logger

logger = setup_logger(__name__)

NRAS_URL = "https://nras.attestation.nvidia.com/v3/attest/gpu"


POLICY = {
    "version": "3.0",
    "authorization-rules": {
        "type": "JWT",
        "overall-claims": {"x-nvidia-overall-att-result": True, "x-nvidia-ver": "2.0"},
        "detached-claims": {
            "measres": "success",
            "x-nvidia-gpu-arch-check": True,
            "x-nvidia-gpu-attestation-report-cert-chain-validated": True,
            "x-nvidia-gpu-attestation-report-parsed": True,
            "x-nvidia-gpu-attestation-report-nonce-match": True,
            "x-nvidia-gpu-attestation-report-signature-verified": True,
            "x-nvidia-gpu-driver-rim-fetched": True,
            "x-nvidia-gpu-driver-rim-schema-validated": True,
            "x-nvidia-gpu-driver-rim-cert-validated": True,
            "x-nvidia-gpu-driver-rim-signature-verified": True,
            "x-nvidia-gpu-driver-rim-measurements-available": True,
            "x-nvidia-gpu-vbios-rim-fetched": True,
            "x-nvidia-gpu-vbios-rim-schema-validated": True,
            "x-nvidia-gpu-vbios-rim-cert-validated": True,
            "x-nvidia-gpu-vbios-rim-signature-verified": True,
            "x-nvidia-gpu-vbios-rim-measurements-available": True,
            "x-nvidia-gpu-vbios-index-no-conflict": True,
        },
    },
}


def verify_attestation(
    attestation_report: AttestationReport, name: str = "thisNode1"
) -> bool:
    """Verify an NVIDIA attestation token against a policy.

    Args:
        token: The attestation token to verify
        policy_path: Optional path to the policy file. If not provided, uses default policy.

    Returns:
        bool: True if the token is valid according to the policy, False otherwise.
    """

    # Create an attestation client instance for token verification.
    logger.info(f"Attestation report: {attestation_report}")
    client = attestation.Attestation()
    client.set_name(name)
    client.set_nonce(attestation_report.nonce)
    client.add_verifier(
        attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
    )

    token = base64.b64decode(attestation_report.gpu_attestation).decode("utf-8")
    logger.info(f"Token: {token}")
    try:
        validation_result = client.validate_token(json.dumps(POLICY), token)
        logger.info(f"Token validation result: {validation_result}")

        return validation_result

    except Exception as e:
        logger.error(f"Failed to verify attestation token: {e}")
        return False
