import logging
import time
from asyncio import Semaphore
from typing import Dict, Optional

from nilai_api import config
from nilai_api.crypto import generate_key_pair
from nilai_api.sev.sev import sev
from nilai_common import ModelServiceDiscovery
from nilai_common.api_model import ModelEndpoint
from verifier.cc_admin import collect_gpu_evidence, attest
import secrets
import json
import base64

logger = logging.getLogger("uvicorn.error")


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        self.sem = Semaphore(2)

        self.discovery_service = ModelServiceDiscovery(
            host=config.ETCD_HOST, port=config.ETCD_PORT
        )
        self._uptime = time.time()
        self._cpu_quote = None
        self._gpu_quote = "<No GPU>"

    @property
    def cpu_attestation(self) -> str:
        if self._cpu_quote is None:
            try:
                sev.init()
                self._cpu_quote = sev.get_quote()
            except RuntimeError:
                self._cpu_quote = "<Non TEE CPU>"
        return self._cpu_quote

    @property
    def gpu_attestation(self) -> str:
        # Check if GPU is available
        try:
            nonce = secrets.token_bytes(32).hex()
            arguments_as_dictionary = {
                "nonce": nonce,
                "verbose": False,
                "test_no_gpu": False,
                "rim_root_cert": None,
                "rim_service_url": None,
                "ocsp_service_url": None,
                "ocsp_attestation_settings": "default",
                "allow_hold_cert": None,
                "ocsp_validity_extension": None,
                "ocsp_cert_revocation_extension_device": None,
                "ocsp_cert_revocation_extension_driver_rim": None,
                "ocsp_cert_revocation_extension_vbios_rim": None,
            }
            evidence_list = collect_gpu_evidence(
                nonce,
            )
            result, jwt_token = attest(arguments_as_dictionary, nonce, evidence_list)
            self._gpu_quote = base64.b64encode(
                json.dumps({"result": result, "jwt_token": jwt_token}).encode()
            ).decode()
            return self._gpu_quote
        except Exception as e:
            logging.error("Could not attest GPU: %s", e)
            return self._gpu_quote

    @property
    def uptime(self):
        elapsed_time = time.time() - self._uptime
        days, remainder = divmod(elapsed_time, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{int(days)} days")
        if hours > 0:
            parts.append(f"{int(hours)} hours")
        if minutes > 0:
            parts.append(f"{int(minutes)} minutes")
        if seconds > 0:
            parts.append(f"{int(seconds)} seconds")

        return ", ".join(parts)

    @property
    async def models(self) -> Dict[str, ModelEndpoint]:
        return await self.discovery_service.discover_models()

    async def get_model(self, model_id: str) -> Optional[ModelEndpoint]:
        return await self.discovery_service.get_model(model_id)


state = AppState()
