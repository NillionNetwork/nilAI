import logging
import time
from asyncio import Semaphore
from typing import Dict, Optional

from dotenv import load_dotenv
from nilai_api.crypto import generate_key_pair
from nilai_api.sev.sev import get_quote, init
from nilai_common import ModelServiceDiscovery, SETTINGS
from nilai_common.api_model import ModelEndpoint

logger = logging.getLogger("uvicorn.error")


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        self.sem = Semaphore(2)

        self.discovery_service = ModelServiceDiscovery(
            host=SETTINGS["etcd_host"], port=SETTINGS["etcd_port"]
        )
        self._uptime = time.time()
        self._cpu_quote = None
        self._gpu_quote = "<No GPU>"

    @property
    def cpu_attestation(self) -> str:
        if self._cpu_quote is None:
            try:
                init()
                self._cpu_quote = get_quote()
            except RuntimeError:
                self._cpu_quote = "<Non TEE CPU>"
        return self._cpu_quote

    @property
    def gpu_attestation(self) -> str:
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


load_dotenv()
state = AppState()
