# nilai/models/model.py
import asyncio
import signal
import logging
import httpx

from nilai_common import (  # Model service discovery and host settings
    SETTINGS,
    ModelServiceDiscovery,
    ModelEndpoint,
    ModelMetadata,
)

logger = logging.getLogger(__name__)


async def get_metadata(num_retries=10):
    """Fetch model metadata from model
    service and return as ModelMetadata object"""
    current_retries = 0
    while True:
        try:
            url = f"http://{SETTINGS['host']}:{SETTINGS['port']}/v1/models"
            # Request model metadata from localhost:8000/v1/models
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                response_data = response.json()
                model_name = response_data["data"][0]["id"]
                return ModelMetadata(
                    id=model_name,  # Unique identifier
                    name=model_name,  # Human-readable name
                    version="1.0",  # Model version
                    description="",
                    author="",  # Model creators
                    license="Apache 2.0",  # Usage license
                    source=f"https://huggingface.co/{model_name}",  # Model source
                    supported_features=["chat_completion"],  # Capabilities
                )

        except Exception as e:
            logger.warning(f"Failed to fetch model metadata from {url}: {e}")
            current_retries += 1
            if current_retries >= num_retries:
                raise e
            await asyncio.sleep(10)


async def run_service(discovery_service, model_endpoint):
    """Runs the model service and keeps it alive"""
    try:
        logger.info(f"Registering model: {model_endpoint.metadata.id}")
        lease = await discovery_service.register_model(model_endpoint, prefix="/models")
        logger.info(f"Model registered successfully: {model_endpoint}")

        await discovery_service.keep_alive(lease)

    except asyncio.CancelledError:
        logger.info("Service shutdown requested")
        raise
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        try:
            await discovery_service.unregister_model(model_endpoint.metadata.id)
            logger.info(f"Model unregistered: {model_endpoint.metadata.id}")
        except Exception as e:
            logger.error(f"Error unregistering model: {e}")


async def main():
    discovery_service = None
    model_endpoint = None

    try:
        # Initialize discovery service
        discovery_service = ModelServiceDiscovery(
            host=SETTINGS["etcd_host"], port=SETTINGS["etcd_port"]
        )

        metadata = await get_metadata()
        model_endpoint = ModelEndpoint(
            url=f"http://{SETTINGS['host']}:{SETTINGS['port']}", metadata=metadata
        )

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

        # Run service
        await run_service(discovery_service, model_endpoint)

    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        raise


async def shutdown():
    """Cleanup and shutdown"""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop = asyncio.get_running_loop()
    loop.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
