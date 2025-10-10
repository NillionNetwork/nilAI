import asyncio
import logging
from asyncio import CancelledError
from datetime import datetime, timezone
from typing import Dict, Optional

import redis.asyncio as redis
from nilai_common.api_model import ModelEndpoint, ModelMetadata
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelServiceDiscovery:
    def __init__(self, host: str = "localhost", port: int = 6379, lease_ttl: int = 60):
        """
        Initialize Redis client for model service discovery.

        :param host: Redis server host
        :param port: Redis server port
        :param lease_ttl: TTL time for endpoint registration (in seconds)
        """
        self.host = host
        self.port = port
        self.lease_ttl = lease_ttl
        self._client: Optional[redis.Redis] = None
        self._model_key: Optional[str] = None

        self.is_healthy = True
        self.last_refresh = None
        self.max_retries = 3
        self.base_delay = 1
        self._shutdown = False

    async def initialize(self):
        """
        Initialize the Redis client.
        """
        if self._client is None:
            self._client = await redis.Redis(
                host=self.host, port=self.port, decode_responses=True
            )

    @property
    async def client(self) -> redis.Redis:
        """
        Get the Redis client.
        """
        if self._client is None:
            await self.initialize()
        if self._client is None:
            # This should never happen
            raise ValueError("Redis client must be initialized")
        return self._client

    async def register_model(
        self, model_endpoint: ModelEndpoint, prefix: str = "/models"
    ) -> str:
        """
        Register a model endpoint in Redis.

        :param model_endpoint: ModelEndpoint to register
        :param prefix: Key prefix for models
        :return: The key used for registration
        """

        # Prepare the key and value
        key = f"{prefix}/{model_endpoint.metadata.id}"
        value = model_endpoint.model_dump_json()

        # Set the key-value pair with TTL
        await (await self.client).setex(key, self.lease_ttl, value)

        # Store the key for keep_alive
        self._model_key = key

        return key

    async def discover_models(
        self,
        name: Optional[str] = None,
        feature: Optional[str] = None,
        prefix: Optional[str] = "/models",
    ) -> Dict[str, ModelEndpoint]:
        """
        Discover models based on optional filters.

        :param name: Optional model name to filter
        :param feature: Optional feature to filter
        :param prefix: Key prefix for models
        :return: Dict of matching ModelEndpoints
        """

        # Get all model keys using SCAN pattern
        discovered_models: Dict[str, ModelEndpoint] = {}
        pattern = f"{prefix}/*"

        cursor = 0
        while True:
            cursor, keys = await (await self.client).scan(
                cursor=cursor, match=pattern, count=100
            )

            for key in keys:
                try:
                    value = await (await self.client).get(key)
                    if value:
                        model_endpoint = ModelEndpoint.model_validate_json(value)

                        # Apply filters if provided
                        if (
                            name
                            and name.lower() not in model_endpoint.metadata.name.lower()
                        ):
                            continue

                        if (
                            feature
                            and feature
                            not in model_endpoint.metadata.supported_features
                        ):
                            continue

                        discovered_models[model_endpoint.metadata.id] = model_endpoint
                except Exception as e:
                    logger.error(f"Error parsing model endpoint from key {key}: {e}")

            if cursor == 0:
                break

        return discovered_models

    async def get_model(
        self, model_id: str, prefix: str = "/models"
    ) -> Optional[ModelEndpoint]:
        """
        Get a model endpoint by ID.

        :param model_id: ID of the model to retrieve
        :param prefix: Key prefix for models
        :return: ModelEndpoint if found, None otherwise
        """
        key = f"{prefix}/{model_id}"
        value = await (await self.client).get(key)

        # Try without prefix if not found
        if not value:
            value = await (await self.client).get(model_id)

        if value:
            return ModelEndpoint.model_validate_json(value)
        return None

    async def unregister_model(self, model_id: str, prefix: str = "/models"):
        """
        Unregister a model from service discovery.

        :param model_id: ID of the model to unregister
        :param prefix: Key prefix for models
        """
        key = f"{prefix}/{model_id}"
        await (await self.client).delete(key)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3)
    )
    async def _refresh_ttl(self, key: str, model_json: str):
        """Refresh the TTL for a Redis key."""
        await (await self.client).setex(key, self.lease_ttl, model_json)
        self.last_refresh = datetime.now(timezone.utc)
        self.is_healthy = True

    async def keep_alive(
        self, key: Optional[str] = None, model_endpoint: Optional[ModelEndpoint] = None
    ):
        """Keep the model registration alive by refreshing TTL with graceful shutdown."""
        if model_endpoint is None and self._model_key is None:
            logger.error("No model endpoint or key provided for keep_alive")
            return

        # Use provided key or stored key
        active_key = key if key else self._model_key

        if not active_key:
            logger.error("No valid key for keep_alive")
            return

        # Get the model JSON once
        if model_endpoint:
            model_json = model_endpoint.model_dump_json()
        else:
            # Fetch current value if not provided
            model_json = await (await self.client).get(active_key)
            if not model_json:
                logger.error(f"No model found at key {active_key}")
                return

        try:
            while not self._shutdown:
                try:
                    await self._refresh_ttl(active_key, model_json)
                    await asyncio.sleep(self.lease_ttl // 2)
                except Exception as e:
                    self.is_healthy = False
                    logger.error(f"TTL refresh failed: {e}")
                    try:
                        await self.initialize()
                    except Exception as init_error:
                        logger.error(f"Reinitialization failed: {init_error}")
                        await asyncio.sleep(self.base_delay)
        except CancelledError:
            logger.info("Keep-alive task cancelled, shutting down...")
            self._shutdown = True
            raise
        finally:
            self.is_healthy = False

    async def close(self):
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()


# Example usage
async def main():
    # Initialize service discovery
    service_discovery = ModelServiceDiscovery(lease_ttl=10)
    await service_discovery.initialize()

    # Create a sample model endpoint
    model_metadata = ModelMetadata(
        name="Image Classification Model",
        version="1.0.0",
        description="ResNet50 based image classifier",
        author="AI Research Team",
        license="MIT",
        source="https://github.com/example/model",
        supported_features=["image_classification", "transfer_learning"],
        tool_support=False,
    )

    model_endpoint = ModelEndpoint(
        url="http://model-service.example.com/predict", metadata=model_metadata
    )

    # Register the model
    key = await service_discovery.register_model(model_endpoint)

    # Start keeping the registration alive in the background
    asyncio.create_task(service_discovery.keep_alive(key, model_endpoint))
    await asyncio.sleep(9)

    # Discover models (with optional filtering)
    discovered_models = await service_discovery.discover_models(
        name="Image Classification", feature="image_classification"
    )
    logger.info(f"FOUND: {len(discovered_models)}")
    for model in discovered_models.values():
        logger.info(f"Discovered Model: {model.metadata.id}")
        logger.info(f"URL: {model.url}")
        logger.info(f"Supported Features: {model.metadata.supported_features}")

    # Keep the service running
    await asyncio.sleep(10)

    # Discover models again
    discovered_models = await service_discovery.discover_models(
        name="Image Classification", feature="image_classification"
    )
    logger.info(f"FOUND: {len(discovered_models)}")
    for model in discovered_models.values():
        logger.info(f"Discovered Model: {model.metadata.id}")
        logger.info(f"URL: {model.url}")
        logger.info(f"Supported Features: {model.metadata.supported_features}")

    # Cleanup
    await service_discovery.unregister_model(model_endpoint.metadata.id)
    await service_discovery.close()


# This allows running the async main function
if __name__ == "__main__":
    asyncio.run(main())
