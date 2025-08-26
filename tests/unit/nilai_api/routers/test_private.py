import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nilai_api.db.users import UserModel
from nilai_common import AttestationReport

from nilai_api.state import state
from ... import model_endpoint, model_metadata


@pytest.mark.asyncio
async def test_runs_in_a_loop():
    assert asyncio.get_running_loop()


@pytest.fixture
def mock_user():
    mock = MagicMock(spec=UserModel)
    mock.userid = "test-user-id"
    mock.name = "Test User"
    mock.apikey = "test-api-key"
    mock.prompt_tokens = 100
    mock.completion_tokens = 50
    mock.total_tokens = 150
    mock.completion_tokens_details = None
    mock.prompt_tokens_details = None
    mock.queries = 10
    return mock


@pytest.fixture
def mock_user_manager(mock_user, mocker):
    from nilai_api.db.users import UserManager
    from nilai_api.db.logs import QueryLogManager

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
        return_value=mock_user,
    )
    mocker.patch.object(
        UserManager,
        "get_all_users",
        return_value=[
            {"userid": "test-user-id", "apikey": "test-api-key"},
            {"userid": "test-user-id-2", "apikey": "test-api-key"},
        ],
    )
    mocker.patch.object(QueryLogManager, "log_query")
    mocker.patch.object(UserManager, "update_last_activity")
    return UserManager


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
    mocker.patch.object(state, "b64_public_key", "test-verifying-key")

    # Patch get_model method
    mocker.patch.object(state, "get_model", return_value=model_endpoint)

    # Patch get_attestation method
    attestation_response = AttestationReport(
        verifying_key="test-verifying-key",
        nonce="0" * 64,
        cpu_attestation="test-cpu-attestation",
        gpu_attestation="test-gpu-attestation",
    )
    # Patch the get_attestation_report function
    mocker.patch(
        "nilai_api.attestation.get_attestation_report",
        return_value=attestation_response,
    )

    return state


@pytest.fixture
def client(mock_user_manager):
    from nilai_api.app import app

    with TestClient(app) as client:
        yield client


# Example test
@pytest.mark.asyncio
async def test_models_property(mock_state):
    # Retrieve the models
    models = await state.models

    # Assert the expected models
    assert models == {"ABC": model_endpoint}


def test_get_usage(mock_user, mock_user_manager, mock_state, client):
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


def test_get_attestation(mock_user, mock_user_manager, mock_state, client):
    response = client.get(
        "/v1/attestation/report",
        headers={"Authorization": "Bearer test-api-key"},
        params={"nonce": "0" * 64},
    )
    assert response.status_code == 200
    assert response.json()["verifying_key"] == "test-verifying-key"
    assert response.json()["cpu_attestation"] == "test-cpu-attestation"
    assert response.json()["gpu_attestation"] == "test-gpu-attestation"


def test_get_models(mock_user, mock_user_manager, mock_state, client):
    response = client.get(
        "/v1/models", headers={"Authorization": "Bearer test-api-key"}
    )
    assert response.status_code == 200
    assert response.json() == [model_metadata.model_dump()]


def test_web_search_with_multimodal_content_error(
    mock_user, mock_user_manager, mock_state, client
):
    """Test that web search with multimodal content returns 400 error."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-api-key"},
        json={
            "model": "ABC",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=="
                            },
                        },
                    ],
                }
            ],
            "web_search": True,
        },
    )
    assert response.status_code == 400
    assert (
        "Web search is not supported with multimodal (image) content"
        in response.json()["detail"]
    )


def test_web_search_with_text_only_works(
    mock_user, mock_user_manager, mock_state, client, mocker
):
    """Test that web search with text-only content works normally."""

    mock_web_search = mocker.patch("nilai_api.routers.private.handle_web_search")
    mock_web_search.return_value = MagicMock(messages=[], sources=[])

    mock_client = mocker.patch("nilai_api.routers.private.AsyncOpenAI")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.return_value.chat.completions.create.return_value = mock_response

    mocker.patch(
        "nilai_api.routers.private.create_signed_chat_completion",
        return_value={"test": "response"},
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-api-key"},
        json={
            "model": "ABC",
            "messages": [{"role": "user", "content": "What is the latest AI news?"}],
            "web_search": True,
        },
    )

    assert response.status_code == 200
    mock_web_search.assert_called_once()


def test_multimodal_completion(
    mock_user, mock_user_manager, mock_state, client, mocker
):
    """Test basic multimodal completion with image content."""
    mock_client = mocker.patch("nilai_api.routers.private.AsyncOpenAI")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This appears to be an image."
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 8
    mock_response.usage.total_tokens = 23
    mock_client.return_value.chat.completions.create.return_value = mock_response

    mocker.patch(
        "nilai_api.routers.private.create_signed_chat_completion",
        return_value={"test": "response"},
    )

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-api-key"},
        json={
            "model": "ABC",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=="
                            },
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json() == {"test": "response"}
