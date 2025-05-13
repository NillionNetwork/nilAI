import asyncio
import logging
from typing import Dict, Optional

from asyncio import CancelledError
from datetime import datetime, timezone
from tenacity import retry, wait_exponential, stop_after_attempt


from etcd3gw import Lease
from etcd3gw.client import Etcd3Client
from nilai_common.api_model import ModelEndpoint, ModelMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelServiceDiscovery:
    def __init__(self, host: str = "localhost", port: int = 2379, lease_ttl: int = 60):
        """
        Initialize etcd client for model service discovery.

        :param host: etcd server host
        :param port: etcd server port
        :param lease_ttl: Lease time for endpoint registration (in seconds)
        """
        self.host = host
        self.port = port
        self.lease_ttl = lease_ttl
        self.initialize()

        self.is_healthy = True
        self.last_refresh = None
        self.max_retries = 3
        self.base_delay = 1
        self._shutdown = False

    def initialize(self):
        """
        Initialize the etcd client.
        """
        self.client = Etcd3Client(host=self.host, port=self.port)

    async def register_model(
        self, model_endpoint: ModelEndpoint, prefix: str = "/models"
    ) -> Lease:
        """
        Register a model endpoint in etcd.

        :param model_endpoint: ModelEndpoint to register
        :return: Lease ID for the registration
        """
        # Create a lease for the endpoint
        lease = self.client.lease(self.lease_ttl)

        # Prepare the key and value
        key = f"{prefix}/{model_endpoint.metadata.id}"
        value = model_endpoint.model_dump_json()

        # Put the key-value pair with the lease
        self.client.put(key, value, lease=lease)

        return lease

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
        :return: List of matching ModelEndpoints
        """
        # Get all model keys
        model_range = self.client.get_prefix(f"{prefix}/")
        self.client.get_prefix

        discovered_models: Dict[str, ModelEndpoint] = {}
        for resp, other in model_range:
            try:
                model_endpoint = ModelEndpoint.model_validate_json(resp.decode("utf-8"))  # type: ignore

                # Apply filters if provided
                if name and name.lower() not in model_endpoint.metadata.name.lower():
                    continue

                if (
                    feature
                    and feature not in model_endpoint.metadata.supported_features
                ):
                    continue

                discovered_models[model_endpoint.metadata.id] = model_endpoint
            except Exception as e:
                logger.error(f"Error parsing model endpoint: {e}")
        return discovered_models

    async def get_model(
        self, model_id: str, prefix: str = "/models"
    ) -> Optional[ModelEndpoint]:
        """
        Get a model endpoint by ID.

        :param model_id: ID of the model to retrieve
        :return: ModelEndpoint if found, None otherwise
        """
        key = f"{prefix}/{model_id}"
        value = self.client.get(key)
        value = self.client.get(model_id) if not value else value
        if value:
            return ModelEndpoint.model_validate_json(value[0].decode("utf-8"))  # type: ignore
        return None

    async def unregister_model(self, model_id: str):
        """
        Unregister a model from service discovery.

        :param model_id: ID of the model to unregister
        """
        key = f"/models/{model_id}"
        self.client.delete(key)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3)
    )
    async def _refresh_lease(self, lease):
        lease.refresh()
        self.last_refresh = datetime.now(timezone.utc)
        self.is_healthy = True

    async def keep_alive(self, lease):
        """Keep the model registration lease alive with graceful shutdown."""
        try:
            while not self._shutdown:
                try:
                    await self._refresh_lease(lease)
                    await asyncio.sleep(self.lease_ttl // 2)
                except Exception as e:
                    self.is_healthy = False
                    logger.error(f"Lease keepalive failed: {e}")
                    try:
                        self.initialize()
                        lease.client = self.client
                    except Exception as init_error:
                        logger.error(f"Reinitialization failed: {init_error}")
                        await asyncio.sleep(self.base_delay)
        except CancelledError:
            logger.info("Lease keepalive task cancelled, shutting down...")
            self._shutdown = True
            raise
        finally:
            self.is_healthy = False


# Example usage
async def main():
    # Initialize service discovery
    service_discovery = ModelServiceDiscovery(lease_ttl=10)

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
    lease = await service_discovery.register_model(model_endpoint)

    # Start keeping the lease alive in the background
    asyncio.create_task(service_discovery.keep_alive(lease))
    await asyncio.sleep(9)  # Keep running for an hour
    # Discover models (with optional filtering)
    discovered_models = await service_discovery.discover_models(
        name="Image Classification", feature="image_classification"
    )
    logger.info(f"FOUND: {len(discovered_models)}")
    for model in discovered_models.values():
        logger.info(f"Discovered Model: {model.metadata.id}")
        logger.info(f"URL: {model.url}")
        logger.info(f"Supported Features: {model.metadata.supported_features}")

    # Optional: Keep the service running
    await asyncio.sleep(10)  # Keep running for an hour
    # Discover models (with optional filtering)
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


# This allows running the async main function
if __name__ == "__main__":
    asyncio.run(main())
