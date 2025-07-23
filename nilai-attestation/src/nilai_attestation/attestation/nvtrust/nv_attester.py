#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Attester: Generate an attestation token from local evidence
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
from nilai_attestation.attestation.nvtrust import is_nvidia_gpu_available, get_client
import base64
from nilai_common import Nonce, NVAttestationToken
import logging

logger = logging.getLogger(__name__)


def nv_attest(nonce: Nonce) -> NVAttestationToken:
    """Generate an attestation token from local evidence.

    Args:
        nonce: The nonce to be used for the attestation

    Returns:
        NVAttestationToken: The attestation token response
    """
    client = get_client()
    client.set_nonce(nonce)

    evidence_list = []

    # Collect evidence and perform attestation.
    options = {}
    if not is_nvidia_gpu_available():
        options["no_gpu_mode"] = True

    evidence_list = client.get_evidence(options=options)
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
