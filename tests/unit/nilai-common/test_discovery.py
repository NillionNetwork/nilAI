import asyncio

import pytest
import pytest_asyncio
from nilai_common.api_model import ModelEndpoint, ModelMetadata
from nilai_common.discovery import ModelServiceDiscovery


@pytest_asyncio.fixture
async def model_service_discovery(redis_host_port):
    """Create a ModelServiceDiscovery instance connected to the test Redis container."""
    host, port = redis_host_port
    discovery = ModelServiceDiscovery(host=host, port=port, lease_ttl=60)
    await discovery.initialize()
    yield discovery
    await discovery.close()


@pytest.fixture
def model_endpoint():
    """Create a sample model endpoint for testing."""
    model_metadata = ModelMetadata(
        name="Test Model",
        version="1.0.0",
        description="Test model description",
        author="Test Author",
        license="MIT",
        source="https://github.com/test/model",
        supported_features=["test_feature"],
        tool_support=False,
    )
    return ModelEndpoint(
        url="http://test-model-service.example.com/predict", metadata=model_metadata
    )


@pytest.mark.asyncio
async def test_register_model(model_service_discovery, model_endpoint):
    """Test registering a model in Redis."""
    key = await model_service_discovery.register_model(model_endpoint)

    # Verify the key was created
    assert key == f"/models/{model_endpoint.metadata.id}"

    # Verify we can retrieve it
    retrieved_model = await model_service_discovery.get_model(
        model_endpoint.metadata.id
    )
    assert retrieved_model is not None
    assert retrieved_model.metadata.id == model_endpoint.metadata.id
    assert retrieved_model.url == model_endpoint.url

    # Cleanup
    await model_service_discovery.unregister_model(model_endpoint.metadata.id)


@pytest.mark.asyncio
async def test_discover_models(model_service_discovery, model_endpoint):
    """Test discovering models from Redis."""
    # Register a model
    await model_service_discovery.register_model(model_endpoint)

    # Discover all models
    discovered_models = await model_service_discovery.discover_models()

    assert len(discovered_models) >= 1
    assert model_endpoint.metadata.id in discovered_models
    assert discovered_models[model_endpoint.metadata.id].url == model_endpoint.url

    # Cleanup
    await model_service_discovery.unregister_model(model_endpoint.metadata.id)


@pytest.mark.asyncio
async def test_discover_models_with_filters(model_service_discovery):
    """Test discovering models with name and feature filters."""
    # Create two different models
    model_metadata_1 = ModelMetadata(
        name="Image Model",
        version="1.0.0",
        description="Image classification model",
        author="Test Author",
        license="MIT",
        source="https://github.com/test/model1",
        supported_features=["image_classification"],
        tool_support=False,
    )
    model_endpoint_1 = ModelEndpoint(
        url="http://image-model.example.com/predict", metadata=model_metadata_1
    )

    model_metadata_2 = ModelMetadata(
        name="Text Model",
        version="1.0.0",
        description="Text generation model",
        author="Test Author",
        license="MIT",
        source="https://github.com/test/model2",
        supported_features=["text_generation"],
        tool_support=False,
    )
    model_endpoint_2 = ModelEndpoint(
        url="http://text-model.example.com/predict", metadata=model_metadata_2
    )

    # Register both models
    await model_service_discovery.register_model(model_endpoint_1)
    await model_service_discovery.register_model(model_endpoint_2)

    # Filter by name
    discovered_models = await model_service_discovery.discover_models(name="Image")
    assert len(discovered_models) == 1
    assert model_endpoint_1.metadata.id in discovered_models

    # Filter by feature
    discovered_models = await model_service_discovery.discover_models(
        feature="text_generation"
    )
    assert len(discovered_models) == 1
    assert model_endpoint_2.metadata.id in discovered_models

    # Cleanup
    await model_service_discovery.unregister_model(model_endpoint_1.metadata.id)
    await model_service_discovery.unregister_model(model_endpoint_2.metadata.id)


@pytest.mark.asyncio
async def test_get_model(model_service_discovery, model_endpoint):
    """Test getting a specific model by ID."""
    # Register a model
    await model_service_discovery.register_model(model_endpoint)

    # Get the model by ID
    model = await model_service_discovery.get_model(model_endpoint.metadata.id)

    assert model is not None
    assert model.metadata.id == model_endpoint.metadata.id
    assert model.url == model_endpoint.url
    assert model.metadata.name == model_endpoint.metadata.name

    # Cleanup
    await model_service_discovery.unregister_model(model_endpoint.metadata.id)


@pytest.mark.asyncio
async def test_get_nonexistent_model(model_service_discovery):
    """Test getting a model that doesn't exist."""
    model = await model_service_discovery.get_model("nonexistent-model-id")
    assert model is None


@pytest.mark.asyncio
async def test_unregister_model(model_service_discovery, model_endpoint):
    """Test unregistering a model from Redis."""
    # Register a model
    await model_service_discovery.register_model(model_endpoint)

    # Verify it exists
    model = await model_service_discovery.get_model(model_endpoint.metadata.id)
    assert model is not None

    # Unregister it
    await model_service_discovery.unregister_model(model_endpoint.metadata.id)

    # Verify it's gone
    model = await model_service_discovery.get_model(model_endpoint.metadata.id)
    assert model is None


@pytest.mark.asyncio
async def test_keep_alive(model_service_discovery, model_endpoint):
    """Test the keep_alive functionality that refreshes TTL."""
    # Register a model with a short TTL
    short_ttl_discovery = ModelServiceDiscovery(
        host=model_service_discovery.host,
        port=model_service_discovery.port,
        lease_ttl=2,  # 2 second TTL
    )
    await short_ttl_discovery.initialize()

    key = await short_ttl_discovery.register_model(model_endpoint)

    # Start keep_alive task
    keep_alive_task = asyncio.create_task(
        short_ttl_discovery.keep_alive(key, model_endpoint)
    )

    # Wait for more than one TTL period
    await asyncio.sleep(3)

    # Model should still be there because keep_alive is refreshing it
    model = await short_ttl_discovery.get_model(model_endpoint.metadata.id)
    assert model is not None

    # Cancel the keep_alive task
    keep_alive_task.cancel()
    try:
        await keep_alive_task
    except asyncio.CancelledError:
        pass

    # Wait for TTL to expire
    await asyncio.sleep(3)

    # Model should be gone now
    model = await short_ttl_discovery.get_model(model_endpoint.metadata.id)
    assert model is None

    await short_ttl_discovery.close()


@pytest.mark.asyncio
async def test_keep_alive_with_stored_key(model_service_discovery, model_endpoint):
    """Test keep_alive using the stored key from registration."""
    # Register a model with a short TTL
    short_ttl_discovery = ModelServiceDiscovery(
        host=model_service_discovery.host,
        port=model_service_discovery.port,
        lease_ttl=2,  # 2 second TTL
    )
    await short_ttl_discovery.initialize()

    await short_ttl_discovery.register_model(model_endpoint)

    # Start keep_alive task without passing the key (it should use the stored one)
    keep_alive_task = asyncio.create_task(
        short_ttl_discovery.keep_alive(model_endpoint=model_endpoint)
    )

    # Wait for more than one TTL period
    await asyncio.sleep(3)

    # Model should still be there
    model = await short_ttl_discovery.get_model(model_endpoint.metadata.id)
    assert model is not None

    # Cancel the keep_alive task
    keep_alive_task.cancel()
    try:
        await keep_alive_task
    except asyncio.CancelledError:
        pass

    await short_ttl_discovery.close()
