import asyncio
import logging
import os
import signal
from typing import List, Set

import httpx

from nilai_common import (
    MODEL_SETTINGS,
    SETTINGS,
    ModelEndpoint,
    ModelMetadata,
    ModelServiceDiscovery,
)
from nilai_common.config import to_bool

logger = logging.getLogger(__name__)


def _parse_csv(value: str) -> List[str]:
    """Parse comma-separated string into list of stripped values."""
    return [item.strip() for item in value.split(",") if item.strip()]


async def _fetch_model_ids(
    api_base: str,
    endpoint: str,
    timeout: float,
    retries: int,
    delay: float,
) -> List[str]:
    """Fetch available model IDs from LMStudio API with retry logic."""
    attempt = 0

    while True:
        try:
            async with httpx.AsyncClient(
                base_url=api_base, timeout=timeout, follow_redirects=True
            ) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
                models = [
                    entry["id"]
                    for entry in payload.get("data", [])
                    if isinstance(entry, dict) and entry.get("id")
                ]
                if models:
                    logger.info(
                        "Discovered LMStudio models at %s%s: %s",
                        api_base,
                        endpoint,
                        ", ".join(models),
                    )
                    return models
                logger.warning(
                    "Received empty model list from %s%s, retrying", api_base, endpoint
                )
        except Exception as exc:
            logger.warning(
                "Unable to fetch LMStudio models from %s%s: %s",
                api_base,
                endpoint,
                exc,
            )

        attempt += 1
        if retries != -1 and attempt >= retries:
            raise RuntimeError(
                f"Failed to discover LMStudio models from {api_base}{endpoint}"
            )
        await asyncio.sleep(delay)


async def _announce_model(
    metadata: ModelMetadata,
    base_url: str,
    discovery_host: str,
    discovery_port: int,
    lease_ttl: int,
    prefix: str,
):
    """Register and maintain a model announcement in Redis."""
    discovery = ModelServiceDiscovery(
        host=discovery_host, port=discovery_port, lease_ttl=lease_ttl
    )
    await discovery.initialize()

    endpoint = ModelEndpoint(url=base_url.rstrip("/"), metadata=metadata)
    key = None

    try:
        key = await discovery.register_model(endpoint, prefix=prefix)
        logger.info(
            "Registered model %s at %s (key=%s)",
            metadata.id,
            endpoint.url,
            key,
        )
        await discovery.keep_alive(key, endpoint)
    except asyncio.CancelledError:
        logger.info("Shutdown requested for model %s", metadata.id)
        raise
    finally:
        if key:
            try:
                await discovery.unregister_model(metadata.id)
                logger.info("Unregistered model %s", metadata.id)
            except Exception as exc:
                logger.error("Failed to unregister model %s: %s", metadata.id, exc)

        # Close the discovery service connection
        await discovery.close()


def _create_metadata(
    model_id: str,
    version: str,
    author: str,
    license_name: str,
    description_template: str,
    source_template: str,
    supported_features: List[str],
    tool_models: Set[str],
    tool_default: bool,
    multimodal_models: Set[str],
    multimodal_default: bool,
) -> ModelMetadata:
    """Create ModelMetadata for a given model ID."""
    return ModelMetadata(
        id=model_id,
        name=model_id,
        version=version,
        description=description_template.format(model_id=model_id),
        author=author,
        license=license_name,
        source=source_template.format(model_id=model_id),
        supported_features=supported_features,
        tool_support=model_id in tool_models or tool_default,
        multimodal_support=model_id in multimodal_models or multimodal_default,
    )


async def main():
    """Main entry point for LMStudio model announcer."""
    logging.basicConfig(level=logging.INFO)

    # Load configuration from environment
    api_base = os.getenv(
        "LMSTUDIO_API_BASE", f"http://{SETTINGS.host}:{SETTINGS.port}"
    ).rstrip("/")
    models_endpoint = os.getenv("LMSTUDIO_MODELS_ENDPOINT", "/v1/models")
    registration_url = os.getenv("LMSTUDIO_REGISTRATION_URL", api_base).rstrip("/")
    lease_ttl = int(os.getenv("LMSTUDIO_LEASE_TTL", "60"))
    discovery_prefix = os.getenv("LMSTUDIO_DISCOVERY_PREFIX", "/models")
    fetch_timeout = float(os.getenv("LMSTUDIO_FETCH_TIMEOUT", "15"))

    # Discover or parse model IDs
    model_ids_env = os.getenv("LMSTUDIO_MODEL_IDS", "")
    if model_ids_env.strip():
        model_ids = _parse_csv(model_ids_env)
    else:
        model_ids = await _fetch_model_ids(
            api_base,
            models_endpoint,
            timeout=fetch_timeout,
            retries=MODEL_SETTINGS.num_retries,
            delay=MODEL_SETTINGS.timeout,
        )

    if not model_ids:
        raise RuntimeError("No LMStudio models discovered; nothing to announce")

    # Parse feature configurations
    supported_features = _parse_csv(
        os.getenv("LMSTUDIO_SUPPORTED_FEATURES", "chat_completion")
    ) or ["chat_completion"]

    tool_default = to_bool(os.getenv("LMSTUDIO_TOOL_SUPPORT_DEFAULT", "false"))
    tool_models = set(_parse_csv(os.getenv("LMSTUDIO_TOOL_SUPPORT_MODELS", "")))

    multimodal_default = to_bool(os.getenv("LMSTUDIO_MULTIMODAL_DEFAULT", "false"))
    multimodal_models = set(_parse_csv(os.getenv("LMSTUDIO_MULTIMODAL_MODELS", "")))

    version = os.getenv("LMSTUDIO_MODEL_VERSION", "local")
    author = os.getenv("LMSTUDIO_MODEL_AUTHOR", "LMStudio")
    license_name = os.getenv("LMSTUDIO_MODEL_LICENSE", "local-use-only")

    description_template = os.getenv(
        "LMSTUDIO_MODEL_DESCRIPTION_TEMPLATE", "LMStudio served model {model_id}"
    )
    source_template = os.getenv(
        "LMSTUDIO_MODEL_SOURCE_TEMPLATE", "lmstudio://{model_id}"
    )

    logger.info(
        "Announcing LMStudio models %s via %s with Redis at %s:%s",
        ", ".join(model_ids),
        registration_url,
        SETTINGS.discovery_host,
        SETTINGS.discovery_port,
    )

    # Create announcement tasks for all models
    tasks = [
        asyncio.create_task(
            _announce_model(
                metadata=_create_metadata(
                    model_id,
                    version,
                    author,
                    license_name,
                    description_template,
                    source_template,
                    supported_features,
                    tool_models,
                    tool_default,
                    multimodal_models,
                    multimodal_default,
                ),
                base_url=registration_url,
                discovery_host=SETTINGS.discovery_host,
                discovery_port=SETTINGS.discovery_port,
                lease_ttl=lease_ttl,
                prefix=discovery_prefix,
            )
        )
        for model_id in model_ids
    ]

    # Setup signal handling
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Wait for either shutdown signal or task completion
    wait_task = asyncio.create_task(stop_event.wait())
    announcer_group = asyncio.gather(*tasks, return_exceptions=True)

    done, _ = await asyncio.wait(
        {wait_task, announcer_group}, return_when=asyncio.FIRST_COMPLETED
    )

    # Handle shutdown
    if wait_task in done:
        logger.info("Stop signal received; shutting down announcer")
        announcer_group.cancel()
        try:
            await announcer_group
        except asyncio.CancelledError:
            pass
    else:
        # Tasks completed (possibly with errors)
        wait_task.cancel()
        results = await announcer_group
        for result in results:
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                raise result


if __name__ == "__main__":
    asyncio.run(main())
