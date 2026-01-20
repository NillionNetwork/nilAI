import logging
from typing import Dict, Optional

from redis.asyncio import Redis

from nilai_api.config import CONFIG
from nilai_api.config.pricing import LLMPriceConfig

logger = logging.getLogger(__name__)

# Redis key prefix for pricing data
REDIS_PRICING_PREFIX = "nilai:pricing:"
REDIS_PRICING_ALL_KEY = "nilai:pricing:_all"


class PricingService:
    """Redis-backed pricing service for thread-safe operations."""

    def __init__(self, redis_client: Redis):
        self._redis = redis_client
        self._default_config = CONFIG.llm_pricing.default
        self._model_configs = CONFIG.llm_pricing.models

    async def initialize_from_config(self) -> None:
        """Load defaults from YAML into Redis on startup (only if not already set)."""
        try:
            # Check if pricing data already exists in Redis
            existing_keys = await self._redis.keys(f"{REDIS_PRICING_PREFIX}*")
            if existing_keys:
                logger.info(
                    "Pricing data already exists in Redis, skipping initialization"
                )
                return

            # Initialize default pricing
            await self._set_price_in_redis("default", self._default_config)

            # Initialize model-specific pricing
            for model_name, price_config in self._model_configs.items():
                await self._set_price_in_redis(model_name, price_config)

            logger.info(
                f"Initialized pricing from config: default + {len(self._model_configs)} models"
            )
        except Exception as e:
            logger.error(f"Failed to initialize pricing from config: {e}")
            raise

    async def _set_price_in_redis(
        self, model_name: str, config: LLMPriceConfig
    ) -> None:
        """Set price for a model in Redis using pipeline for atomicity."""
        key = f"{REDIS_PRICING_PREFIX}{model_name}"
        config_json = config.model_dump_json()

        async with self._redis.pipeline(transaction=True) as pipe:
            # Set individual model key
            pipe.set(key, config_json)
            # Update hash for bulk retrieval
            pipe.hset(REDIS_PRICING_ALL_KEY, model_name, config_json)
            await pipe.execute()

    async def get_price(self, model_name: str) -> LLMPriceConfig:
        """
        Get price from Redis for a specific model.

        Falls back to default pricing if model not found or Redis unavailable.
        """
        try:
            key = f"{REDIS_PRICING_PREFIX}{model_name}"
            config_json = await self._redis.get(key)

            if config_json:
                return LLMPriceConfig.model_validate_json(config_json)

            # Fallback to default pricing from Redis
            default_key = f"{REDIS_PRICING_PREFIX}default"
            default_json = await self._redis.get(default_key)

            if default_json:
                return LLMPriceConfig.model_validate_json(default_json)

            # Last resort: return config default
            logger.warning(
                f"No pricing found in Redis for model '{model_name}', using config default"
            )
            return self._default_config

        except Exception as e:
            logger.error(f"Failed to get price from Redis: {e}, using config default")
            return self._default_config

    async def get_all_prices(self) -> Dict[str, LLMPriceConfig]:
        """Get all prices from Redis hash."""
        try:
            all_prices: Dict[str, str] = await self._redis.hgetall(
                REDIS_PRICING_ALL_KEY
            )  # type: ignore[assignment]

            result = {}
            for model_name, config_json in all_prices.items():
                # Handle bytes if Redis returns bytes
                if isinstance(model_name, bytes):
                    model_name = model_name.decode("utf-8")
                if isinstance(config_json, bytes):
                    config_json = config_json.decode("utf-8")

                result[model_name] = LLMPriceConfig.model_validate_json(config_json)

            return result

        except Exception as e:
            logger.error(f"Failed to get all prices from Redis: {e}")
            # Fallback to config defaults
            result = {"default": self._default_config}
            result.update(self._model_configs)
            return result

    async def set_price(self, model_name: str, config: LLMPriceConfig) -> None:
        """Atomic update of price for a model using Redis pipeline."""
        await self._set_price_in_redis(model_name, config)
        logger.info(f"Updated pricing for model '{model_name}'")

    async def delete_price(self, model_name: str) -> bool:
        """
        Remove custom pricing for a model (will use default).

        Returns True if the key existed and was deleted, False otherwise.
        """
        if model_name == "default":
            raise ValueError("Cannot delete default pricing")

        key = f"{REDIS_PRICING_PREFIX}{model_name}"

        async with self._redis.pipeline(transaction=True) as pipe:
            # Check if key exists
            pipe.exists(key)
            # Delete individual model key
            pipe.delete(key)
            # Remove from hash
            pipe.hdel(REDIS_PRICING_ALL_KEY, model_name)
            results = await pipe.execute()

        existed = results[0] > 0
        if existed:
            logger.info(f"Deleted pricing for model '{model_name}'")
        return existed

    async def price_exists(self, model_name: str) -> bool:
        """Check if a custom price exists for a model."""
        key = f"{REDIS_PRICING_PREFIX}{model_name}"
        return await self._redis.exists(key) > 0


# Global pricing service instance
_pricing_service: Optional[PricingService] = None


def set_pricing_service(service: PricingService) -> None:
    """Set the global pricing service instance."""
    global _pricing_service
    _pricing_service = service


def get_pricing_service() -> PricingService:
    """Get the global pricing service instance."""
    if _pricing_service is None:
        raise RuntimeError("Pricing service not initialized")
    return _pricing_service
