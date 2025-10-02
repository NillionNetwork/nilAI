import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nilai_api.db.users import RateLimits, UserModel
from nilai_common import AttestationReport, Source

from nilai_api.state import state
from ... import model_endpoint, model_metadata, response as RESPONSE


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
    mock.rate_limits = RateLimits().get_effective_limits().model_dump_json()
    mock.rate_limits_obj = RateLimits().get_effective_limits()
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
        return_value={
            "userid": "test-user-id",
            "apikey": "test-api-key",
            "rate_limits": RateLimits().get_effective_limits().model_dump_json(),
        },
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
            {
                "userid": "test-user-id",
                "apikey": "test-api-key",
                "rate_limits": RateLimits().get_effective_limits().model_dump_json(),
            },
            {
                "userid": "test-user-id-2",
                "apikey": "test-api-key",
                "rate_limits": RateLimits().get_effective_limits().model_dump_json(),
            },
        ],
    )
    mocker.patch.object(QueryLogManager, "log_query")
    mocker.patch.object(UserManager, "update_last_activity")
    return UserManager


@pytest.fixture
def mock_state(mocker):
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
        "nilai_api.routers.private.get_attestation_report",
        new_callable=AsyncMock,
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


def test_chat_completion(mock_user, mock_state, mock_user_manager, mocker, client):
    mocker.patch("openai.api_key", new="test-api-key")
    from openai.types.chat import ChatCompletion

    data = RESPONSE.model_dump()
    data.pop("signature")
    data.pop("sources", None)
    response_data = ChatCompletion(**data)
    # Patch nilai_api.routers.private.AsyncOpenAI to return a mock instance with chat.completions.create as an AsyncMock
    mock_chat_completions = MagicMock()
    mock_chat_completions.create = mocker.AsyncMock(return_value=response_data)
    mock_chat = MagicMock()
    mock_chat.completions = mock_chat_completions
    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.chat = mock_chat
    mocker.patch(
        "nilai_api.routers.private.AsyncOpenAI", return_value=mock_async_openai_instance
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct",
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
        "completion_tokens_details": None,
        "prompt_tokens_details": None,
    }


def test_chat_completion_stream_includes_sources(
    mock_user, mock_state, mock_user_manager, mocker, client
):
    source = Source(source="https://example.com", content="Example result")

    mock_web_search_result = MagicMock()
    mock_web_search_result.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me something new."},
    ]
    mock_web_search_result.sources = [source]

    mocker.patch(
        "nilai_api.routers.private.handle_web_search",
        new=AsyncMock(return_value=mock_web_search_result),
    )

    class MockChunk:
        def __init__(self, data, usage=None):
            self._data = data
            self.usage = usage

        def model_dump(self, exclude_unset=True):
            return self._data

    class MockUsage:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    first_chunk = MockChunk(
        data={
            "id": "stream-1",
            "object": "chat.completion.chunk",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "created": 0,
            "choices": [{"delta": {"content": "Hello"}, "index": 0}],
        }
    )

    final_chunk = MockChunk(
        data={
            "id": "stream-1",
            "object": "chat.completion.chunk",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "created": 0,
            "choices": [
                {"delta": {}, "finish_reason": "stop", "index": 0},
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
            },
        },
        usage=MockUsage(prompt_tokens=5, completion_tokens=7),
    )

    async def chunk_generator():
        yield first_chunk
        yield final_chunk

    mock_chat_completions = MagicMock()
    mock_chat_completions.create = AsyncMock(return_value=chunk_generator())
    mock_chat = MagicMock()
    mock_chat.completions = mock_chat_completions
    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.chat = mock_chat

    mocker.patch(
        "nilai_api.routers.private.AsyncOpenAI",
        return_value=mock_async_openai_instance,
    )

    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me something new."},
        ],
        "stream": True,
        "web_search": True,
    }

    headers = {"Authorization": "Bearer test-api-key"}

    with client.stream(
        "POST", "/v1/chat/completions", json=payload, headers=headers
    ) as response:
        assert response.status_code == 200
        data_lines = [
            line
            for line in response.iter_lines(decode_unicode=True)
            if line and line.startswith("data: ")
        ]

    assert data_lines, "Expected SSE data from stream response"
    first_payload = json.loads(data_lines[0][len("data: "):])
    assert "sources" not in first_payload
    final_payload = json.loads(data_lines[-1][len("data: "):])
    assert "sources" in final_payload
    assert len(final_payload["sources"]) == 1
    assert final_payload["sources"][0]["source"] == "https://example.com"
