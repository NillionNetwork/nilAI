"""
Test suite for nilAI HTTP API

This test suite uses httpx to make requests to the nilAI HTTP API.

To run the tests, use the following command:

pytest tests/e2e/test_http.py
"""

import json

from .config import BASE_URL, test_models
from .nuc import get_nuc_token
import httpx
import pytest


@pytest.fixture
def client():
    """Create an HTTPX client with default headers"""
    invocation_token = get_nuc_token()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token.token}",
        },
        timeout=None,
    )


def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get("health")
    print(response.json())
    assert response.status_code == 200, "Health endpoint should return 200 OK"
    assert "status" in response.json(), "Health response should contain status"


def test_models_endpoint(client):
    """Test the models endpoint"""
    response = client.get("/models")
    assert response.status_code == 200, "Models endpoint should return 200 OK"
    assert isinstance(response.json(), list), "Models should be returned as a list"

    # Check for specific models mentioned in the requests
    expected_models = test_models
    model_names = [model.get("id") for model in response.json()]
    for model in expected_models:
        assert model in model_names, f"Expected model {model} not found"


def test_usage_endpoint(client):
    """Test the usage endpoint"""
    response = client.get("/usage")
    assert response.status_code == 200, "Usage endpoint should return 200 OK"

    # Basic usage response validation
    usage_data = response.json()
    assert isinstance(usage_data, dict), "Usage data should be a dictionary"
    # Optional additional checks based on expected usage data structure
    expected_keys = [
        "total_tokens",
        "completion_tokens",
        "prompt_tokens",
        "queries",
    ]

    for key in expected_keys:
        assert key in usage_data, f"Expected key {key} not found in usage data"


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
        "temperature": 0.2,
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
def test_model_streaming_request(client, model):
    """Test streaming request for different models"""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {"role": "user", "content": "Write a short poem about mountains."},
        ],
        "temperature": 0.2,
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
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
            },
            {"role": "user", "content": "What is the weather like in Paris today?"},
        ],
        "temperature": 0.2,
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
        if "tool_calls" in message:
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
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides accurate and concise information.",
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
        "temperature": 0.2,
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


def test_large_payload_handling(client):
    """Test handling of large input payloads"""
    # Create a very large system message
    large_system_message = "Hello " * 10000  # 100KB of text

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
        "temperature": 0.2,
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
        "temperature": 0.2,
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
