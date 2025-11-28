import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nilai_api.db.users import RateLimits, UserModel
from nilai_common import AttestationReport, Source

from nilai_api.state import state
from ... import (
    model_endpoint,
    model_metadata,
    RESPONSES_RESPONSE,
)


@pytest.mark.asyncio
async def test_runs_in_a_loop():
    assert asyncio.get_running_loop()


@pytest.fixture
def mock_user():
    mock = MagicMock(spec=UserModel)
    mock.user_id = "test-user-id"
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

    # Patch QueryLogManager for usage
    mocker.patch.object(
        QueryLogManager,
        "get_user_token_usage",
        return_value={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "queries": 10,
        },
    )
    mocker.patch.object(QueryLogManager, "log_query")

    # Mock validate_credential for authentication
    mocker.patch(
        "nilai_api.auth.strategies.validate_credential",
        new_callable=AsyncMock,
        return_value=mock_user,
    )

    return UserManager


@pytest.fixture
def mock_state(mocker):
    expected_models = {"ABC": model_endpoint}

    mock_discovery_service = mocker.Mock()
    mock_discovery_service.discover_models = AsyncMock(return_value=expected_models)
    mock_discovery_service.initialize = AsyncMock()

    mocker.patch.object(state, "discovery_service", mock_discovery_service)

    mocker.patch.object(state, "b64_public_key", "test-verifying-key")

    mocker.patch.object(state, "get_model", return_value=model_endpoint)

    attestation_response = AttestationReport(
        verifying_key="test-verifying-key",
        nonce="0" * 64,
        cpu_attestation="test-cpu-attestation",
        gpu_attestation="test-gpu-attestation",
    )
    mocker.patch(
        "nilai_api.routers.private.get_attestation_report",
        new_callable=AsyncMock,
        return_value=attestation_response,
    )

    return state


@pytest.fixture
def mock_metering_context(mocker):
    """Mock the metering context to avoid credit service calls during tests."""
    mock_context = MagicMock()
    mock_context.set_response = MagicMock()
    return mock_context


@pytest.fixture
def client(mock_user_manager, mock_state, mock_metering_context):
    from nilai_api.app import app
    from nilai_api.credit import LLMMeter

    # Override the LLMMeter dependency to avoid actual credit service calls
    app.dependency_overrides[LLMMeter] = lambda: mock_metering_context

    with TestClient(app) as client:
        yield client

    # Clean up the override after tests
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_models_property(mock_state):
    models = await state.models

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


def test_create_response(mock_user, mock_state, mock_user_manager, mocker, client):
    mocker.patch("openai.api_key", new="test-api-key")

    response_data = RESPONSES_RESPONSE

    mock_responses = MagicMock()
    mock_responses.create = mocker.AsyncMock(return_value=response_data)
    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.responses = mock_responses

    mocker.patch(
        "nilai_api.routers.endpoints.responses.AsyncOpenAI",
        return_value=mock_async_openai_instance,
    )
    mocker.patch(
        "nilai_api.routers.endpoints.responses.handle_responses_tool_workflow",
        return_value=(response_data, 0, 0),
    )
    mocker.patch(
        "nilai_api.routers.endpoints.responses.state.get_model",
        return_value=model_endpoint,
    )
    mocker.patch("nilai_api.db.logs.QueryLogContext.commit", new_callable=AsyncMock)

    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "instructions": "You are a helpful assistant.",
        "input": "What is your name?",
    }

    response = client.post(
        "/v1/responses",
        json=payload,
        headers={"Authorization": "Bearer test-api-key"},
    )

    assert response.status_code == 200
    assert "usage" in response.json()
    assert response_data.usage is not None
    assert response.json()["usage"] == response_data.usage.model_dump(mode="json")


def test_create_response_stream_includes_sources(
    mock_user, mock_state, mock_user_manager, mocker, client
):
    from openai.types.responses import Response as OpenAIResponse, ResponseUsage
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
    )
    from nilai_common import ResponseCompletedEvent

    mock_user.rate_limits_obj.web_search_rate_limit_minute = 100

    source = Source(source="https://example.com", content="Example result")

    mock_web_search_result = MagicMock()
    mock_web_search_result.input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Tell me something new."},
            ],
            "type": "message",
        }
    ]
    mock_web_search_result.instructions = "You are a helpful assistant."
    mock_web_search_result.sources = [source]

    mocker.patch(
        "nilai_api.routers.endpoints.responses.handle_web_search_for_responses",
        new=AsyncMock(return_value=mock_web_search_result),
    )

    class MockEvent:
        def __init__(self, data):
            self._data = data

        def model_dump(self, exclude_unset=True):
            return self._data

    streaming_usage = ResponseUsage(
        input_tokens=5,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=7,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=12,
    )

    streaming_response = OpenAIResponse(
        **{
            **RESPONSES_RESPONSE.model_dump(),
            "usage": streaming_usage,
        }
    )

    first_event = MockEvent(
        {
            "type": "response.output_text.delta",
            "response": {"id": "resp-stream-1"},
            "delta": {"text": "Hello"},
        }
    )

    final_event = ResponseCompletedEvent(
        response=streaming_response, sequence_number=1, type="response.completed"
    )

    async def chunk_generator():
        yield first_event
        yield final_event

    mock_responses = MagicMock()
    mock_responses.create = AsyncMock(return_value=chunk_generator())
    mock_async_openai_instance = MagicMock()
    mock_async_openai_instance.responses = mock_responses

    mocker.patch(
        "nilai_api.routers.endpoints.responses.AsyncOpenAI",
        return_value=mock_async_openai_instance,
    )
    mocker.patch(
        "nilai_api.routers.endpoints.responses.state.get_model",
        return_value=model_endpoint,
    )
    mocker.patch("nilai_api.db.logs.QueryLogContext.commit", new_callable=AsyncMock)

    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "instructions": "You are a helpful assistant.",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Tell me something new."},
                ],
                "type": "message",
            }
        ],
        "stream": True,
        "web_search": True,
    }

    headers = {"Authorization": "Bearer test-api-key"}

    with client.stream("POST", "/v1/responses", json=payload, headers=headers) as resp:
        assert resp.status_code == 200
        data_lines = [
            line for line in resp.iter_lines() if line and line.startswith("data: ")
        ]

    assert data_lines, "Expected SSE data from stream response"

    first_payload = json.loads(data_lines[0][len("data: ") :])
    assert "data" not in first_payload or "sources" not in first_payload.get("data", {})

    final_payload = json.loads(data_lines[-1][len("data: ") :])
    assert "data" in final_payload
    assert "sources" in final_payload["data"]
    assert len(final_payload["data"]["sources"]) == 1
    assert final_payload["data"]["sources"][0]["source"] == "https://example.com"
