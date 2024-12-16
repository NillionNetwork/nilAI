from unittest.mock import AsyncMock
from nilai_common.api_model import (
    ChatResponse,
    Choice,
    Message,
    ModelEndpoint,
    ModelMetadata,
    Usage,
)
import pytest
import asyncio
from fastapi.testclient import TestClient
from nilai_api.app import app
from nilai_api.db import UserManager
from nilai_api.state import state

client = TestClient(app)
model_metadata = ModelMetadata(
    id="ABC",  # Unique identifier
    name="ABC",  # Human-readable name
    version="1.0",  # Model version
    description="Description",
    author="Author",  # Model creators
    license="License",  # Usage license
    source="http://test-model-url",  # Model source
    supported_features=["supported_feature"],  # Capabilities
)

model_endpoint = ModelEndpoint(url="http://test-model-url", metadata=model_metadata)


@pytest.mark.asyncio
async def test_runs_in_a_loop():
    assert asyncio.get_running_loop()


@pytest.fixture
def mock_user():
    return {"userid": "test-user-id", "name": "Test User"}


@pytest.fixture
def mock_user_manager(mocker):
    mocker.patch.object(
        UserManager,
        "get_token_usage",
        return_value={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "queries": 10,
        },
    )
    mocker.patch.object(UserManager, "update_token_usage")
    mocker.patch.object(
        UserManager,
        "get_user_token_usage",
        return_value={"prompt_tokens": 100, "completion_tokens": 50},
    )
    mocker.patch.object(
        UserManager,
        "insert_user",
        return_value={"userid": "test-user-id", "apikey": "test-api-key"},
    )
    mocker.patch.object(
        UserManager,
        "check_api_key",
        return_value={"name": "Test User", "userid": "test-user-id"},
    )
    mocker.patch.object(
        UserManager,
        "get_all_users",
        return_value=[
            {"userid": "test-user-id", "apikey": "test-api-key"},
            {"userid": "test-user-id-2", "apikey": "test-api-key"},
        ],
    )


@pytest.fixture
def mock_state(mocker, event_loop):
    # Prepare expected models data

    expected_models = {"ABC": model_endpoint}

    # Create a mock discovery service that returns the expected models
    mock_discovery_service = mocker.Mock()
    mock_discovery_service.discover_models = AsyncMock(return_value=expected_models)

    # Create a mock AppState
    mocker.patch.object(state, "discovery_service", mock_discovery_service)

    # Patch other attributes
    mocker.patch.object(state, "verifying_key", "test-verifying-key")
    mocker.patch.object(state, "_cpu_quote", "test-cpu-attestation")
    mocker.patch.object(state, "_gpu_quote", "test-gpu-attestation")

    # Patch get_model method
    mocker.patch.object(state, "get_model", return_value=model_endpoint)

    return state


# Example test
@pytest.mark.asyncio
async def test_models_property(mock_state):
    # Retrieve the models
    models = await state.models

    # Assert the expected models
    assert models == {"ABC": model_endpoint}


def test_get_usage(mock_user, mock_user_manager):
    response = client.get("/v1/usage", headers={"Authorization": "Bearer test-api-key"})
    assert response.status_code == 200
    assert response.json() == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_get_attestation(mock_user, mock_user_manager, mock_state):
    response = client.get(
        "/v1/attestation/report", headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "verifying_key": "test-verifying-key",
        "cpu_attestation": "test-cpu-attestation",
        "gpu_attestation": "test-gpu-attestation",
    }


def test_get_models(mock_user, mock_user_manager, mock_state):
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert response.json() == [model_metadata.model_dump()]


def test_chat_completion(mock_user, mock_state, mock_user_manager, mocker):
    response = ChatResponse(
        id="test-id",
        object="test-object",
        model="test-model",
        created=123456,
        choices=[
            Choice(
                index=0,
                message=Message(role="test-role", content="test-content"),
                finish_reason="test-finish-reason",
                logprobs={"test-logprobs": "test-value"},
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        signature="test-signature",
    )
    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=mocker.Mock(status_code=200, content=response.model_dump_json()),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Llama-3.2-1B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is your name?"},
            ],
        },
        headers={"Authorization": "Bearer test-api-key"},
    )
    assert response.status_code == 200
    assert "usage" in response.json()
    assert response.json()["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
