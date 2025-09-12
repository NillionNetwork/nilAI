import logging
import time
from asyncio import Semaphore
from typing import Dict, Optional

from nilai_api.config import CONFIG
from nilai_api.crypto import generate_key_pair
from nilai_common import ModelServiceDiscovery
from nilai_common.api_model import ModelEndpoint

logger = logging.getLogger("uvicorn.error")


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.b64_public_key = generate_key_pair()
        self.sem = Semaphore(2)

        self.discovery_service = ModelServiceDiscovery(
            host=CONFIG.etcd.host, port=CONFIG.etcd.port
        )
        self._uptime = time.time()

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
        if model_id is None or len(model_id) == 0:
            return None
        return await self.discovery_service.get_model(model_id)


state = AppState()
