import pytest
import pytest_asyncio

from nilai_api.config.pricing import LLMPriceConfig
from nilai_api.pricing_service import (
    PricingService,
    get_pricing_service,
    set_pricing_service,
    REDIS_PRICING_PREFIX,
    REDIS_PRICING_ALL_KEY,
)
from nilai_api.rate_limiting import setup_redis_conn


@pytest_asyncio.fixture
async def redis_client(redis_server):
    """Create a Redis client connected to the test container."""
    host_ip = redis_server.get_container_host_ip()
    host_port = redis_server.get_exposed_port(6379)
    client, _ = await setup_redis_conn(f"redis://{host_ip}:{host_port}")
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def pricing_service(redis_client):
    """Create a PricingService instance and clean up Redis keys before/after."""
    # Clean up any existing pricing keys
    keys = await redis_client.keys(f"{REDIS_PRICING_PREFIX}*")
    if keys:
        await redis_client.delete(*keys)

    service = PricingService(redis_client)
    set_pricing_service(service)
    yield service

    # Clean up after test
    keys = await redis_client.keys(f"{REDIS_PRICING_PREFIX}*")
    if keys:
        await redis_client.delete(*keys)


@pytest.mark.asyncio
async def test_initialize_from_config(pricing_service, redis_client):
    """Test that pricing is initialized from config into Redis."""
    await pricing_service.initialize_from_config()

    # Check that default pricing was set
    default_key = f"{REDIS_PRICING_PREFIX}default"
    default_json = await redis_client.get(default_key)
    assert default_json is not None

    default_config = LLMPriceConfig.model_validate_json(default_json)
    assert default_config.prompt_tokens_price == 0.15
    assert default_config.completion_tokens_price == 0.45
    assert default_config.web_search_cost == 0.05


@pytest.mark.asyncio
async def test_initialize_skips_if_data_exists(pricing_service, redis_client):
    """Test that initialization is skipped if data already exists."""
    # First initialization
    await pricing_service.initialize_from_config()

    # Modify a value directly in Redis
    custom_config = LLMPriceConfig(
        prompt_tokens_price=99.0, completion_tokens_price=99.0, web_search_cost=99.0
    )
    await redis_client.set(
        f"{REDIS_PRICING_PREFIX}default", custom_config.model_dump_json()
    )

    # Second initialization should be skipped
    await pricing_service.initialize_from_config()

    # Verify the custom value is still there
    default_json = await redis_client.get(f"{REDIS_PRICING_PREFIX}default")
    default_config = LLMPriceConfig.model_validate_json(default_json)
    assert default_config.prompt_tokens_price == 99.0


@pytest.mark.asyncio
async def test_get_price_returns_model_specific_price(pricing_service):
    """Test getting price for a specific model."""
    await pricing_service.initialize_from_config()

    # Get price for a model that should have specific pricing
    price = await pricing_service.get_price("meta-llama/Llama-3.2-1B-Instruct")
    assert price.prompt_tokens_price == 0.03
    assert price.completion_tokens_price == 0.09


@pytest.mark.asyncio
async def test_get_price_falls_back_to_default(pricing_service):
    """Test that unknown models fall back to default pricing."""
    await pricing_service.initialize_from_config()

    # Get price for an unknown model
    price = await pricing_service.get_price("unknown/model")
    assert price.prompt_tokens_price == 0.15
    assert price.completion_tokens_price == 0.45
    assert price.web_search_cost == 0.05


@pytest.mark.asyncio
async def test_set_price(pricing_service, redis_client):
    """Test setting price for a model."""
    await pricing_service.initialize_from_config()

    # Set a new price
    new_config = LLMPriceConfig(
        prompt_tokens_price=10.0, completion_tokens_price=15.0, web_search_cost=0.1
    )
    await pricing_service.set_price("test-model", new_config)

    # Verify it was set correctly
    price = await pricing_service.get_price("test-model")
    assert price.prompt_tokens_price == 10.0
    assert price.completion_tokens_price == 15.0
    assert price.web_search_cost == 0.1

    # Verify it's in the hash
    hash_value = await redis_client.hget(REDIS_PRICING_ALL_KEY, "test-model")
    assert hash_value is not None


@pytest.mark.asyncio
async def test_get_all_prices(pricing_service):
    """Test getting all prices."""
    await pricing_service.initialize_from_config()

    all_prices = await pricing_service.get_all_prices()

    assert "default" in all_prices
    assert all_prices["default"].prompt_tokens_price == 0.15
    assert all_prices["default"].completion_tokens_price == 0.45
    assert all_prices["default"].web_search_cost == 0.05
    assert "meta-llama/Llama-3.2-1B-Instruct" in all_prices
    assert all_prices["meta-llama/Llama-3.2-1B-Instruct"].prompt_tokens_price == 0.03
    assert (
        all_prices["meta-llama/Llama-3.2-1B-Instruct"].completion_tokens_price == 0.09
    )
    assert all_prices["meta-llama/Llama-3.2-1B-Instruct"].web_search_cost == 0.05


@pytest.mark.asyncio
async def test_delete_price(pricing_service):
    """Test deleting a custom price."""
    await pricing_service.initialize_from_config()

    # Add a custom price
    custom_config = LLMPriceConfig(
        prompt_tokens_price=0.15, completion_tokens_price=0.45, web_search_cost=0.05
    )
    await pricing_service.set_price("custom-model", custom_config)

    # Verify it exists
    price = await pricing_service.get_price("custom-model")
    assert price.prompt_tokens_price == 0.15
    assert price.completion_tokens_price == 0.45
    assert price.web_search_cost == 0.05

    # Delete it
    existed = await pricing_service.delete_price("custom-model")
    assert existed is True

    # Verify it falls back to default now
    price = await pricing_service.get_price("custom-model")
    assert price.prompt_tokens_price == 0.15
    assert price.completion_tokens_price == 0.45
    assert price.web_search_cost == 0.05


@pytest.mark.asyncio
async def test_delete_price_returns_false_if_not_exists(pricing_service):
    """Test deleting a non-existent price returns False."""
    await pricing_service.initialize_from_config()

    existed = await pricing_service.delete_price("nonexistent-model")
    assert existed is False


@pytest.mark.asyncio
async def test_delete_default_raises_error(pricing_service):
    """Test that deleting default pricing raises an error."""
    await pricing_service.initialize_from_config()

    with pytest.raises(ValueError, match="Cannot delete default pricing"):
        await pricing_service.delete_price("default")


@pytest.mark.asyncio
async def test_price_exists(pricing_service):
    """Test checking if a price exists."""
    await pricing_service.initialize_from_config()

    assert await pricing_service.price_exists("default") is True
    assert (
        await pricing_service.price_exists("meta-llama/Llama-3.2-1B-Instruct") is True
    )
    assert await pricing_service.price_exists("nonexistent-model") is False


@pytest.mark.asyncio
async def test_get_pricing_service_not_initialized():
    """Test that get_pricing_service raises error when not initialized."""
    # Reset the global service
    from nilai_api import pricing_service as ps_module

    old_service = ps_module._pricing_service
    ps_module._pricing_service = None

    try:
        with pytest.raises(RuntimeError, match="Pricing service not initialized"):
            get_pricing_service()
    finally:
        ps_module._pricing_service = old_service


@pytest.mark.asyncio
async def test_update_existing_price(pricing_service):
    """Test updating an existing model's price."""
    await pricing_service.initialize_from_config()

    # Get original price
    original = await pricing_service.get_price("meta-llama/Llama-3.2-1B-Instruct")
    assert original.prompt_tokens_price == 0.03

    # Update it
    new_config = LLMPriceConfig(
        prompt_tokens_price=100.0, completion_tokens_price=100.0, web_search_cost=5.0
    )
    await pricing_service.set_price("meta-llama/Llama-3.2-1B-Instruct", new_config)

    # Verify update
    updated = await pricing_service.get_price("meta-llama/Llama-3.2-1B-Instruct")
    assert updated.prompt_tokens_price == 100.0
    assert updated.completion_tokens_price == 100.0
    assert updated.web_search_cost == 5.0
