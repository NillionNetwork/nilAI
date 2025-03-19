import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nilai_common.api_model import ModelEndpoint, ModelMetadata
from nilai_common.discovery import ModelServiceDiscovery


@pytest.fixture
def model_service_discovery():
    with patch("nilai_common.discovery.Etcd3Client") as MockClient:
        mock_client = MockClient.return_value
        discovery = ModelServiceDiscovery()
        discovery.client = mock_client
        yield discovery


@pytest.fixture
def model_endpoint():
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
    lease_mock = MagicMock()
    model_service_discovery.client.lease.return_value = lease_mock

    lease = await model_service_discovery.register_model(model_endpoint)

    model_service_discovery.client.put.assert_called_once_with(
        f"/models/{model_endpoint.metadata.id}",
        model_endpoint.model_dump_json(),
        lease=lease_mock,
    )
    assert lease == lease_mock


@pytest.mark.asyncio
async def test_discover_models(model_service_discovery, model_endpoint):
    model_service_discovery.client.get_prefix.return_value = [
        (model_endpoint.model_dump_json().encode("utf-8"), None)
    ]

    discovered_models = await model_service_discovery.discover_models()

    assert len(discovered_models) == 1
    assert model_endpoint.metadata.id in discovered_models
    assert discovered_models[model_endpoint.metadata.id] == model_endpoint


@pytest.mark.asyncio
async def test_get_model(model_service_discovery, model_endpoint):
    model_service_discovery.client.get.return_value = (
        model_endpoint.model_dump_json().encode("utf-8"),
        None,
    )

    model = await model_service_discovery.get_model(model_endpoint.metadata.id)

    assert model == model_endpoint


@pytest.mark.asyncio
async def test_unregister_model(model_service_discovery, model_endpoint):
    await model_service_discovery.unregister_model(model_endpoint.metadata.id)

    model_service_discovery.client.delete.assert_called_once_with(
        f"/models/{model_endpoint.metadata.id}"
    )


@pytest.mark.asyncio
async def test_keep_alive(model_service_discovery):
    lease_mock = MagicMock()
    lease_mock.refresh = AsyncMock()

    async def keep_alive_task():
        await model_service_discovery.keep_alive(lease_mock)

    task = asyncio.create_task(keep_alive_task())
    await asyncio.sleep(0.1)
    task.cancel()

    lease_mock.refresh.assert_called()
