#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Attester: Generate an attestation token from local evidence
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
from nv_attestation_sdk import attestation  # type: ignore

import subprocess
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

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


@lru_cache(maxsize=1)
def get_client() -> attestation.Attestation:
    """Create and configure the attestation client with appropriate verifiers.

    This function initializes an attestation client and configures it based on the availability
    of an NVIDIA GPU. If a GPU is available, a remote verifier is added. Otherwise, a local
    verifier is configured.

    Returns:
        attestation.Attestation: A configured attestation client instance.
    """
    # Create and configure the attestation client.
    client = attestation.Attestation()
    client.set_name("nilai-attestation-module")
    logger.info("Checking if NVIDIA GPU is available")

    if is_nvidia_gpu_available():
        logger.info("NVIDIA GPU is available")
        # Configure the remote verifier.
        # WARNING: The next statement happens at a global level. It shall only be done once.
        client.add_verifier(
            attestation.Devices.GPU, attestation.Environment.REMOTE, NRAS_URL, ""
        )
    else:
        logger.info("NVIDIA GPU is not available")
        # WARNING: The next statement happens at a global level. It shall only be done once.
        client.add_verifier(
            attestation.Devices.GPU,
            attestation.Environment.LOCAL,
            "",
            "",
            OCSP_URL,
            RIM_URL,
        )
    return client
