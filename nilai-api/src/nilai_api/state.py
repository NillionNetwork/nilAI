import time
from asyncio import Semaphore, Lock as Mutex

from dotenv import load_dotenv
import httpx
import logging

from regex import P

from nilai_api.crypto import generate_key_pair
from nilai_common import ModelEndpoint, ModelMetadata


from nilai_api.sev.sev import init, get_quote

logger = logging.getLogger('uvicorn.error')


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        self.sem = Semaphore(2)

        self.model_mutex = Mutex()

        self.models = {}

        self._uptime = time.time()
        self._cpu_quote = None
        self._gpu_quote = None

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
        return "<No GPU>"

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

    async def add_endpoint(self, endpoint: ModelEndpoint):
        async with self.model_mutex:
            if endpoint.url in self.models:
                logger.warning(f"Model {endpoint.url} already exists in the list of models.")
                return
            self.models[endpoint.metadata.name] = endpoint

    async def remove_endpoint(self, model_name: str):
        async with self.model_mutex:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not found in the list of models.")
                return
            del self.models[model_name]


    async def update_endpoints(self):
        async with self.model_mutex:
            async with httpx.AsyncClient() as client:
                for endpoint in self.models.values():
                    print(endpoint.url)

                    try:
                        # Async HTTP GET request with 1 second timeout
                        response = await client.get(endpoint.url + "/health", timeout=1.0)
                        response.raise_for_status()

                        model_info = ModelMetadata(**response.json())
                        if model_info.name in self.models:
                            logger.warning(
                                f"Model {model_info.name} already exists in the list of models."
                            )
                            continue

                        logger.info(f"Adding model {model_info.name} to the list of models.")

                    except httpx.HTTPStatusError as e:
                        logger.warning(
                            f"Model {endpoint.name} with url: {endpoint.url} returned non-200 status code: {e.response.status_code}"
                        )
                    except httpx.ConnectError:
                        print("Connection error")
                    except httpx.TimeoutException:
                        print("Timeout error")
                    except httpx.RequestError:
                        print("Request error")


load_dotenv()
state = AppState()
