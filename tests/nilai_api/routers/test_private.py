import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from nilai_api.app import app
from nilai_api.db import UserManager
from nilai_api.state import state
from tests import model_endpoint, model_metadata, response as RESPONSE

client = TestClient(app)

# @pytest.fixture(scope='session', autouse=True)
# def event_loop():
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     yield loop
#     loop.close()


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
        return_value={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
            "queries": 10,
        },
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
    mocker.patch.object(UserManager, "initialize_db")
    mocker.patch.object(UserManager, "log_query")
    mocker.patch.object(UserManager, "update_last_activity")


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


def test_get_usage(mock_user, mock_user_manager, mock_state):
    response = client.get("/v1/usage", headers={"Authorization": "Bearer test-api-key"})
    assert response.status_code == 200
    assert response.json() == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
        "queries": 10,
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
    mocker.patch("openai.api_key", new="test-api-key")
    # Mock the response from the OpenAI API
    from openai.types.chat import ChatCompletion

    data = RESPONSE.model_dump()
    data.pop("signature")
    response_data = ChatCompletion(**data)
    mocker.patch(
        "openai._base_client.SyncAPIClient._request", return_value=response_data
    )

    # Mock client.post behavior
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

    # Assertions
    assert response.status_code == 200
    assert "usage" in response.json()
    assert response.json()["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }
