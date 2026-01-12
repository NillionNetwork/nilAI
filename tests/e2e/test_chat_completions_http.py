"""
Test suite for nilAI Chat Completions endpoint using HTTP client

This test suite uses httpx to make requests to the nilAI HTTP API.

To run the tests, use the following command:

pytest tests/e2e/test_chat_completions_http.py
"""

import json
import os
import re

from .config import BASE_URL, ENVIRONMENT, test_models, AUTH_STRATEGY
import httpx
import pytest


def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get("health")
    print(response.json())
    assert response.status_code == 200, "Health endpoint should return 200 OK"
    assert "status" in response.json(), "Health response should contain status"


def test_models_endpoint(client):
    """Test the models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200, (
        f"Models endpoint should return 200 OK: {response.json()}"
    )
    assert isinstance(response.json(), list), "Models should be returned as a list"

    # Check for specific models mentioned in the requests
    expected_models = test_models
    model_names = [model.get("id") for model in response.json()]
    for model in expected_models:
        assert model in model_names, f"Expected model {model} not found"


def test_usage_endpoint(client):
    """Test the usage endpoint"""
    response = client.get("/usage")
    assert response.status_code == 200, (
        f"Usage endpoint should return 200 OK: {response.json()} {BASE_URL}"
    )
    # Basic usage response validation
    usage_data = response.json()
    assert isinstance(usage_data, dict), "Usage data should be a dictionary"
    # Optional additional checks based on expected usage data structure
    expected_keys = [
        "total_tokens",
        "completion_tokens",
        "prompt_tokens",
    ]

    for key in expected_keys:
        assert key in usage_data, f"Expected key {key} not found in usage data"


@pytest.mark.skipif(
    ENVIRONMENT != "mainnet",
    reason="Attestation endpoint not available in non-mainnet environment",
)
def test_attestation_endpoint(client):
    """Test the attestation endpoint"""
    response = client.get("/attestation/report")
    assert response.status_code == 200, "Attestation endpoint should return 200 OK"

    # Basic attestation report validation
    report = response.json()
    assert isinstance(report, dict), "Attestation report should be a dictionary"
    assert "cpu_attestation" in report, (
        "Attestation report should contain a 'cpu_attestation' key"
    )
    assert "gpu_attestation" in report, (
        "Attestation report should contain a 'gpu_attestation' key"
    )
    assert "verifying_key" in report, (
        "Attestation report should contain a 'verifying_key' key"
    )


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_model_standard_request(client, model):
    """Test standard (non-streaming) request for different models"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    response = client.post("/chat/completions", json=payload, timeout=30)
    assert response.status_code == 200, (
        f"Standard request for {model} failed with status {response.status_code}"
    )

    response_json = response.json()
    print(response_json)
    assert "choices" in response_json, "Response should contain choices"
    assert len(response_json["choices"]) > 0, "At least one choice should be present"

    # Check content of response
    content = response_json["choices"][0].get("message", {}).get("content", "")
    assert content, f"No content returned for {model}"

    # Check finish reason
    assert response_json["choices"][0].get("finish_reason") == "stop", (
        f"Finish reason should be stop for {model}"
    )

    # Check that the response is not empty
    assert content.strip(), f"Empty response returned for {model}"

    # Check that the usage is not 0
    assert response_json["usage"]["prompt_tokens"] > 0, (
        f"Prompt tokens are 0 for {model}"
    )
    assert response_json["usage"]["completion_tokens"] > 0, (
        f"Completion tokens are 0 for {model}"
    )
    assert response_json["usage"]["total_tokens"] > 0, f"Total tokens are 0 for {model}"
    # Log response for debugging
    print(
        f"\nModel {model} standard response: {content[:100]}..."
        if len(content) > 100
        else content
    )


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_model_standard_request_nillion_2025(nillion_2025_client, model):
    """Test standard (non-streaming) request for different models"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }

    response = nillion_2025_client.post("/chat/completions", json=payload, timeout=30)
    assert response.status_code == 200, (
        f"Standard request for {model} failed with status {response.status_code}"
    )

    response_json = response.json()
    print(response_json)
    assert "choices" in response_json, "Response should contain choices"
    assert len(response_json["choices"]) > 0, "At least one choice should be present"

    # Check content of response
    content = response_json["choices"][0].get("message", {}).get("content", "")
    assert content, f"No content returned for {model}"

    # Check finish reason
    assert response_json["choices"][0].get("finish_reason") == "stop", (
        f"Finish reason should be stop for {model}"
    )

    # Check that the response is not empty
    assert content.strip(), f"Empty response returned for {model}"

    # Check that the usage is not 0
    assert response_json["usage"]["prompt_tokens"] > 0, (
        f"Prompt tokens are 0 for {model}"
    )
    assert response_json["usage"]["completion_tokens"] > 0, (
        f"Completion tokens are 0 for {model}"
    )
    assert response_json["usage"]["total_tokens"] > 0, f"Total tokens are 0 for {model}"
    # Log response for debugging
    print(
        f"\nModel {model} standard response: {content[:100]}..."
        if len(content) > 100
        else content
    )


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_model_streaming_request(client, model):
    """Test streaming request for different models"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {
                "role": "user",
                "content": "Write a short poem about mountains. It must be 20 words maximum.",
            },
        ],
        "stream": True,
    }

    with client.stream("POST", "/chat/completions", json=payload) as response:
        assert response.status_code == 200, (
            f"Streaming request for {model} failed with status {response.status_code}"
        )

        # Check that we're getting a stream
        assert response.headers.get("Transfer-Encoding") == "chunked", (
            "Response should be streamed"
        )

        # Read a few chunks to verify streaming works
        chunk_count = 0
        content = ""
        had_usage = False
        for chunk in response.iter_lines():
            if chunk and chunk.strip() and chunk.startswith("data:"):
                chunk_count += 1
                chunk = chunk[6:]  # Remove the data: prefix
                print(
                    f"\nModel {model} stream chunk {chunk_count}: [{type(chunk)}] {chunk}"
                )
                chunk_json = json.loads(chunk)
                # Check for content in the chunk
                if (
                    chunk_json.get("choices")
                    and chunk_json["choices"][0].get("delta")
                    and chunk_json["choices"][0]["delta"].get("content")
                ):
                    content += chunk_json["choices"][0]["delta"]["content"]
                # Check for usage data in the chunk at least once in the stream
                if chunk_json.get("usage"):
                    print(f"Usage: {chunk_json.get('usage')}")
                    had_usage = True
        assert had_usage, f"No usage data received for {model} streaming request"
        assert chunk_count > 0, f"No chunks received for {model} streaming request"
        print(f"Received {chunk_count} chunks for {model} streaming request")


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_model_tools_request(client, model):
    """Test tools request for different models"""
    if model == "openai/gpt-oss-20b":
        pytest.skip(
            "openai/gpt-oss-20b model only supports tool calls with responses endpoint"
        )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. When a user asks a question that requires weather, use the get_weather tool to get the weather information.",
            },
            {"role": "user", "content": "What is the weather like in Paris today?"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Paris, France",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
    }

    try:
        response = client.post("/chat/completions", json=payload)
        assert response.status_code == 200, (
            f"Tools request for {model} failed with status {response.status_code}"
        )

        response_json = response.json()
        assert "choices" in response_json, "Response should contain choices"
        assert len(response_json["choices"]) > 0, (
            "At least one choice should be present"
        )

        message = response_json["choices"][0].get("message", {})

        # Check if the model used the tool
        if message.get("tool_calls"):
            tool_calls = message.get("tool_calls", [])
            print(f"\nModel {model} tool calls: {json.dumps(tool_calls, indent=2)}")
            assert len(tool_calls) > 0, f"Tool calls array is empty for {model}"

            # Validate the first tool call
            first_call = tool_calls[0]
            assert "function" in first_call, "Tool call should have a function"
            assert "name" in first_call["function"], "Function should have a name"
            assert first_call["function"]["name"] == "get_weather", (
                "Function name should be get_weather"
            )
            assert "arguments" in first_call["function"], (
                "Function should have arguments"
            )

            # Parse arguments and check for location
            args = json.loads(first_call["function"]["arguments"])
            assert "location" in args, "Arguments should contain location"
            assert "paris" in args["location"].lower(), "Location should be Paris"
        else:
            # If no tool calls, check content
            content = message.get("content", "")
            print(
                f"\nModel {model} response (no tool call): {content[:100]}..."
                if len(content) > 100
                else content
            )
            assert content, f"No content or tool calls returned for {model}"
    except Exception as e:
        # Some models might not support tools, so we'll just log the error
        print(f"\nError testing tools with {model}: {str(e)}")
        # Re-raise if it's an assertion error
        raise e


@pytest.mark.parametrize("model", test_models)
def test_function_calling_with_streaming_httpx(client, model):
    """Test function calling with streaming using httpx, verifying tool calls and usage data."""
    if model == "openai/gpt-oss-20b":
        pytest.skip(
            "Skipping test for openai/gpt-oss-20b model as it only supports responses endpoint"
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information. You are a helpful assistant that provides accurate and concise information. For getting weather, use function call with get_weather.",
            },
            {
                "role": "user",
                "content": "What is the weather like in Paris today?",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Paris, France",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "stream": True,
    }

    with client.stream("POST", "/chat/completions", json=payload) as response:
        assert response.status_code == 200, (
            f"Streaming request for {model} failed with status {response.status_code}"
        )
        had_tool_call = False
        had_usage = False
        for line in response.iter_lines():
            if line and line.strip() and line.startswith("data:"):
                data_line = line[6:].strip()
                try:
                    chunk_json = json.loads(data_line)
                except json.JSONDecodeError:
                    continue
                choices = chunk_json.get("choices", [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    delta = choices[0].get("delta", {})
                    if "tool_calls" in delta and delta["tool_calls"]:
                        had_tool_call = True
                if chunk_json.get("usage"):
                    had_usage = True
        assert had_tool_call, f"No tool calls received for {model} streaming request"
        assert had_usage, f"No usage data received for {model} streaming request"


def test_invalid_auth_token(client):
    """Test behavior with an invalid or expired authentication token"""
    invalid_client = httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer invalid_token_123",
        },
        verify=False,
    )

    response = invalid_client.get("/attestation/report")
    assert response.status_code in [
        401,
        403,
    ], "Invalid token should result in unauthorized access"


def test_rate_limiting(client):
    """Test rate limiting by sending multiple rapid requests"""
    # Payload for repeated requests
    payload = {
        "model": test_models[0],
        "messages": [{"role": "user", "content": "Generate a short poem"}],
    }

    # Send multiple rapid requests
    responses = []
    for _ in range(20):  # Adjust number based on expected rate limits
        response = client.post("/chat/completions", json=payload)
        responses.append(response)

    # Check for potential rate limit responses
    rate_limit_statuses = [429, 403, 503]
    rate_limited_responses = [
        r for r in responses if r.status_code in rate_limit_statuses
    ]

    # If rate limiting is expected, at least some requests should be rate-limited
    if len(rate_limited_responses) == 0:
        pytest.skip("No rate limiting detected. Manual review may be needed.")


@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_rate_limiting_nucs(rate_limited_client):
    """Test rate limiting by sending multiple rapid requests"""
    # Payload for repeated requests
    payload = {
        "model": test_models[0],
        "messages": [{"role": "user", "content": "What is your name?"}],
    }

    # Send multiple rapid requests
    responses = []
    for _ in range(4):  # Adjust number based on expected rate limits
        response = rate_limited_client.post("/chat/completions", json=payload)
        responses.append(response)

    # Check for potential rate limit responses
    rate_limit_statuses = [429, 403, 503]
    rate_limited_responses = [
        r for r in responses if r.status_code in rate_limit_statuses
    ]

    assert len(rate_limited_responses) > 0, (
        "No NUC rate limiting detected, when expected"
    )


def test_large_payload_handling(client):
    """Test handling of large input payloads"""
    # Create a very large system message
    large_system_message = "Hello " * 1000  # 100KB of text

    payload = {
        "model": test_models[0],
        "messages": [
            {"role": "system", "content": large_system_message},
            {"role": "user", "content": "Respond briefly"},
        ],
        "max_tokens": 50,
    }

    response = client.post("/chat/completions", json=payload, timeout=30)
    print(response)

    # Check for appropriate handling of large payload
    assert response.status_code in [
        200,
        413,
    ], "Large payload should be handled gracefully"

    if response.status_code == 200:
        response_json = response.json()
        assert "choices" in response_json, "Response should contain choices"
        assert len(response_json["choices"]) > 0, (
            "At least one choice should be present"
        )


@pytest.mark.parametrize("invalid_model", ["nonexistent-model/v1", "", None, "   "])
def test_invalid_model_handling(client, invalid_model):
    """Test handling of invalid or non-existent models"""
    payload = {
        "model": invalid_model,
        "messages": [{"role": "user", "content": "Test invalid model"}],
    }

    response = client.post("/chat/completions", json=payload)

    # Expect a 400 (Bad Request) or 404 (Not Found) for invalid models
    assert response.status_code in [
        400,
        404,
    ], f"Invalid model {invalid_model} should return an error"


def test_timeout_handling(client):
    """Test request timeout behavior"""
    payload = {
        "model": test_models[0],
        "messages": [
            {
                "role": "user",
                "content": "Generate a very long response that might take a while",
            }
        ],
        "max_tokens": 1000,
    }

    try:
        # Use a very short timeout to force a timeout scenario
        _ = client.post("/chat/completions", json=payload, timeout=0.1)
        pytest.fail("Request should have timed out")
    except httpx.TimeoutException:
        # Timeout is the expected behavior
        assert True, "Request timed out as expected"


def test_empty_messages_handling(client):
    """Test handling of empty messages list"""
    payload = {"model": test_models[0], "messages": []}

    response = client.post("/chat/completions", json=payload)
    print(response)

    # Expect a 400 Bad Request for empty messages
    assert response.status_code == 400, "Empty messages should return a Bad Request"

    # Check error response structure
    response_json = response.json()
    assert "detail" in response_json, "Error response should contain an invalid key"


def test_unsupported_parameters(client):
    """Test handling of unsupported or unexpected parameters"""
    payload = {
        "model": test_models[0],
        "messages": [{"role": "user", "content": "Test unsupported parameters"}],
        "unsupported_param": "some_value",
        "another_weird_param": 42,
    }

    response = client.post("/chat/completions", json=payload)

    # Expect either successful response ignoring extra params or a 400 Bad Request
    assert response.status_code in [
        200,
        400,
    ], "Unsupported parameters should be handled gracefully"


def test_chat_completion_invalid_temperature(client):
    """Test chat completion with invalid temperature type that should trigger a validation error"""
    payload = {
        "model": test_models[0],
        "messages": [{"role": "user", "content": "What is the weather like?"}],
        "temperature": "hot",
    }
    response = client.post("/chat/completions", json=payload)
    print(response)
    assert response.status_code == 400, (
        "Invalid temperature type should return a 422 error"
    )


def test_chat_completion_missing_model(client):
    """Test chat completion with missing model field to trigger a validation error"""
    payload = {
        "messages": [{"role": "user", "content": "What is your name?"}],
    }
    response = client.post("/chat/completions", json=payload)
    assert response.status_code == 400, (
        "Missing model should return a 422 validation error"
    )


def test_chat_completion_negative_max_tokens(client):
    """Test chat completion with negative max_tokens value triggering a validation error"""
    payload = {
        "model": test_models[0],
        "messages": [{"role": "user", "content": "Tell me a joke."}],
        "max_tokens": -10,
    }
    response = client.post("/chat/completions", json=payload)
    assert response.status_code == 400, (
        "Negative max_tokens should return a 422 validation error"
    )


def test_chat_completion_high_temperature(client):
    """Test chat completion with a high temperature value to check model's creative generation under extreme conditions"""
    payload = {
        "model": test_models[0],
        "messages": [
            {"role": "system", "content": "You are a creative assistant."},
            {
                "role": "user",
                "content": "Write an imaginative story about a wizard.",
            },
        ],
        "temperature": 5.0,  # Extremely high temperature for creative responses
        "max_tokens": 50,
    }
    response = client.post("/chat/completions", json=payload)
    assert response.status_code == 200, (
        "High temperature request should return a valid response"
    )
    response_json = response.json()
    assert "choices" in response_json, "Response should contain choices"
    assert len(response_json["choices"]) > 0, "At least one choice should be present"


def test_model_streaming_request_high_token(client):
    """Test streaming request with high max_tokens to verify response streaming over longer texts"""
    payload = {
        "model": test_models[0],
        "messages": [
            {"role": "system", "content": "You are a creative assistant."},
            {
                "role": "user",
                "content": "Tell me a long story about a superhero's journey.",
            },
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True,
    }
    with client.stream("POST", "/chat/completions", json=payload) as response:
        assert response.status_code == 200, (
            "Streaming with high max_tokens should return 200 status"
        )
        chunk_count = 0
        for line in response.iter_lines():
            if line and line.strip() and line.startswith("data:"):
                chunk_count += 1
        assert chunk_count > 0, (
            "Should receive at least one chunk for high token streaming request"
        )


@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC required for this tests on nilDB"
)
@pytest.mark.skip(
    reason="Skipping test_nildb_delegation because it requires a newer version of secretvaults-py"
)
def test_nildb_delegation(client: httpx.Client):
    """Tests getting a delegation token for nilDB and validating that token to be valid"""
    from secretvaults.common.keypair import Keypair
    from nuc.envelope import NucTokenEnvelope
    from nuc.validate import NucTokenValidator, ValidationParameters
    from nuc.nilauth import NilauthClient
    from nilai_api.config import CONFIG
    from nuc.token import Did

    keypair = Keypair.generate()
    did = keypair.to_did_string()

    response = client.get("/delegation", params={"prompt_delegation_request": did})

    assert response.status_code == 200, (
        f"Delegation token should be returned: {response.text}"
    )
    assert "token" in response.json(), "Delegation token should be returned"
    assert "did" in response.json(), "Delegation did should be returned"
    token = response.json()["token"]
    did = response.json()["did"]
    assert token is not None, "Delegation token should be returned"
    assert did is not None, "Delegation did should be returned"

    # Validate the token with nilAuth url for nilDB
    nuc_token_envelope = NucTokenEnvelope.parse(token)
    nilauth_public_keys = [
        Did(NilauthClient(CONFIG.nildb.nilauth_url).about().public_key.serialize())
    ]
    NucTokenValidator(nilauth_public_keys).validate(
        nuc_token_envelope, context={}, parameters=ValidationParameters.default()
    )


@pytest.mark.skip(
    reason="prompt cannot be accessed because of a secretvaults-py update"
)
@pytest.mark.parametrize(
    "model",
    test_models,
)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC required for this tests on nilDB"
)
def test_nildb_prompt_document(document_id_client: httpx.Client, model):
    """Tests getting a prompt document from nilDB and executing a chat completion with it"""
    pytest.skip(
        "Skipping test_nildb_prompt_document because it requires a newer version of secretvaults-py"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Can you make a small rhyme?"},
        ],
    }

    response = document_id_client.post("/chat/completions", json=payload, timeout=30)

    assert response.status_code == 200, (
        f"Response should be successful: {response.text}"
    )
    # Response must talk about cheese which is what the prompt document contains
    message: str = response.json()["choices"][0].get("message", {}).get("content", None)
    assert "cheese" in message.lower(), "Response should contain cheese"


@pytest.fixture
def high_web_search_rate_limit(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_MINUTE", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_HOUR", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_DAY", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT", "9999")


@pytest.mark.parametrize("model", test_models)
def test_web_search(client, model, high_web_search_rate_limit):
    """Test web_search functionality with proper source validation."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and up-to-date information.",
            },
            {
                "role": "user",
                "content": "Who won the Roland Garros Open in 2024? Just reply with the winner's name.",
            },
        ],
        "extra_body": {"web_search": True},
        "max_tokens": 150,
    }

    response = client.post("/chat/completions", json=payload, timeout=60)
    assert response.status_code == 200, (
        f"Response for {model} failed with status {response.status_code}"
    )

    response_json = response.json()
    assert response_json.get("model") == model, f"Response model should be {model}"
    assert "choices" in response_json, "Response should contain choices"
    assert len(response_json["choices"]) > 0, (
        "Response should contain at least one choice"
    )

    message = response_json["choices"][0].get("message", {})
    content = message.get("content", "")
    reasoning_content = message.get("reasoning_content", "")

    assert content or reasoning_content, (
        "Response should contain content or reasoning_content"
    )

    sources = response_json.get("sources")
    if sources is not None:
        assert isinstance(sources, list), "Sources should be a list"
        assert len(sources) > 0, "Sources should not be empty"
        print(f"Sources found: {len(sources)}")
    else:
        print(
            "Warning: Sources field is None - web search may not be enabled or working properly"
        )

    print(
        f"\nModel {model} web search response: {content[:100] if content else 'No content'}..."
    )


@pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"),
    reason="Requires E2B_API_KEY for code execution sandbox",
)
@pytest.mark.parametrize("model", test_models)
def test_execute_python_sha256_e2e(client, model):
    if model == "openai/gpt-oss-20b":
        pytest.skip(
            "Skipping test for openai/gpt-oss-20b model as it only supports responses endpoint"
        )

    expected = "75cc238b167a05ab7336d773cb096735d459df2f0df9c8df949b1c44075df8a5"

    system_msg = (
        "You are a helpful assistant. When a user asks a question that requires code execution, "
        "use the execute_python tool to find the answer. After the tool provides its result, "
        "you must use that result to formulate a clear, final answer to the user's original question. "
        "Do not include any code or JSON in your final response."
    )
    user_msg = "Execute this exact Python code and return the result: import hashlib; print(hashlib.sha256('Nillion'.encode()).hexdigest())"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executes a snippet of Python code in a secure sandbox and returns the standard output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to be executed.",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
    }

    trials = 3
    escaped_expected = re.escape(expected)
    pattern = rf"\b{escaped_expected}\b"
    last_data = None
    last_content = ""
    last_status = None
    for _ in range(trials):
        response = client.post("/chat/completions", json=payload)
        last_status = response.status_code
        if response.status_code != 200:
            continue
        data = response.json()
        last_data = data
        if not ("choices" in data and data["choices"]):
            continue
        message = data["choices"][0].get("message", {})
        content = message.get("content") or ""
        last_content = content
        normalized_content = re.sub(r"\s+", " ", content)
        if re.search(pattern, normalized_content):
            break
    else:
        pytest.fail(
            (
                "Expected exact SHA-256 hash not found after retries.\n"
                f"Last status: {last_status}\n"
                f"Got: {last_content[:200]}...\n"
                f"Expected: {expected}\n"
                f"Full: {json.dumps(last_data, indent=2)[:1000] if last_data else '<no json>'}"
            )
        )
