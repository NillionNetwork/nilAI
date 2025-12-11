# nilai/models/model.py
import asyncio
import logging
import signal

import httpx

from nilai_common import (
    MODEL_CAPABILITIES,
    MODEL_SETTINGS,
    SETTINGS,
    ModelEndpoint,
    ModelMetadata,
    ModelServiceDiscovery,
)

logger = logging.getLogger(__name__)


async def get_metadata():
    """Fetch model metadata from model service and return as ModelMetadata object."""
    current_retries = 0
    while True:
        url = None
        try:
            url = f"{SETTINGS.url.rstrip('/')}/v1/models"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                response_data = response.json()
                model_name = response_data["data"][0]["id"]
                return ModelMetadata(
                    id=model_name,
                    name=model_name,
                    version="1.0",
                    description="",
                    author="",
                    license="Apache 2.0",
                    source=f"https://huggingface.co/{model_name}",
                    supported_features=["chat_completion"],
                    tool_support=MODEL_CAPABILITIES.tool_support,
                    multimodal_support=MODEL_CAPABILITIES.multimodal_support,
                )

        except Exception as e:
            if not url:
                logger.warning(f"Failed to build url: {e}")
            else:
                logger.warning(f"Failed to fetch model metadata from {url}: {e}")
            current_retries += 1
            if (
                MODEL_SETTINGS.num_retries != -1
                and current_retries >= MODEL_SETTINGS.num_retries
            ):
                raise e
            await asyncio.sleep(MODEL_SETTINGS.timeout)


async def run_service(discovery_service, model_endpoint):
    """Register model with discovery service and keep it alive."""
    key = None
    try:
        logger.info(f"Registering model: {model_endpoint.metadata.id}")
        key = await discovery_service.register_model(model_endpoint, prefix="/models")
        logger.info(f"Model registered successfully: {model_endpoint}")

        await discovery_service.keep_alive(key, model_endpoint)

    except asyncio.CancelledError:
        logger.info("Service shutdown requested")
        raise
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise
    finally:
        if key:
            try:
                await discovery_service.unregister_model(model_endpoint.metadata.id)
                logger.info(f"Model unregistered: {model_endpoint.metadata.id}")
            except Exception as e:
                logger.error(f"Error unregistering model: {e}")

        # Close the discovery service connection
        await discovery_service.close()


async def main():
    """Main entry point for model daemon."""
    logging.basicConfig(level=logging.INFO)

    # Initialize discovery service
    discovery_service = ModelServiceDiscovery(redis_url=SETTINGS.redis_url)
    await discovery_service.initialize()

    # Fetch metadata and create endpoint
    metadata = await get_metadata()
    model_endpoint = ModelEndpoint(url=SETTINGS.url, metadata=metadata)

    # Create service task
    service_task = asyncio.create_task(run_service(discovery_service, model_endpoint))

    # Setup signal handling
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Wait for either shutdown signal or service completion
    wait_task = asyncio.create_task(stop_event.wait())

    done, _ = await asyncio.wait(
        {wait_task, service_task}, return_when=asyncio.FIRST_COMPLETED
    )

    # Handle shutdown
    if wait_task in done:
        logger.info("Stop signal received; shutting down daemon")
        service_task.cancel()
        try:
            await service_task
        except asyncio.CancelledError:
            pass
    else:
        # Service completed (possibly with error)
        wait_task.cancel()
        await service_task  # Re-raise any exception


if __name__ == "__main__":
    asyncio.run(main())
