import json
import os
import re

from .config import BASE_URL, test_models, AUTH_STRATEGY, api_key_getter
from .nuc import (
    get_rate_limited_nuc_token,
    get_invalid_rate_limited_nuc_token,
    get_nildb_nuc_token,
    get_document_id_nuc_token,
)
import httpx
import pytest


@pytest.fixture
def client():
    invocation_token: str = api_key_getter()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token}",
        },
        verify=False,
        timeout=None,
    )


@pytest.fixture
def rate_limited_client():
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token.token}",
        },
        timeout=None,
        verify=False,
    )


@pytest.fixture
def invalid_rate_limited_client():
    invocation_token = get_invalid_rate_limited_nuc_token()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token.token}",
        },
        timeout=None,
        verify=False,
    )


@pytest.fixture
def nildb_client():
    invocation_token = get_nildb_nuc_token()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token.token}",
        },
        timeout=None,
        verify=False,
    )


@pytest.fixture
def nillion_2025_client():
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer Nillion2025",
        },
        verify=False,
        timeout=None,
    )


@pytest.fixture
def document_id_client():
    invocation_token = get_document_id_nuc_token()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token.token}",
        },
        verify=False,
        timeout=None,
    )


@pytest.mark.parametrize("model", test_models)
def test_model_standard_request(client, model):
    payload = {
        "model": model,
        "input": "What is the capital of France?",
        "instructions": "You are a helpful assistant that provides accurate and concise information.",
        "temperature": 0.2,
    }

    response = client.post("/v1/responses", json=payload, timeout=30)
    assert response.status_code == 200, (
        f"Standard request for {model} failed with status {response.status_code}"
    )

    response_json = response.json()
    print(response_json)
    assert "output" in response_json, "Response should contain output"
    assert len(response_json["output"]) > 0, (
        "At least one output item should be present"
    )

    text_items = [
        item for item in response_json["output"] if item.get("type") == "text"
    ]
    assert len(text_items) > 0, "Response should contain text items"
    content = text_items[0].get("text", "")
    assert content, f"No content returned for {model}"

    assert response_json.get("status") == "completed", (
        f"Status should be completed for {model}"
    )

    assert content.strip(), f"Empty response returned for {model}"

    assert response_json["usage"]["input_tokens"] > 0, f"Input tokens are 0 for {model}"
    assert response_json["usage"]["output_tokens"] > 0, (
        f"Output tokens are 0 for {model}"
    )
    assert response_json["usage"]["total_tokens"] > 0, f"Total tokens are 0 for {model}"

    assert "signature" in response_json, "Response should contain signature"

    print(
        f"\nModel {model} standard response: {content[:100]}..."
        if len(content) > 100
        else content
    )


@pytest.mark.parametrize("model", test_models)
def test_model_standard_request_nillion_2025(nillion_2025_client, model):
    payload = {
        "model": model,
        "input": "What is the capital of France?",
        "instructions": "You are a helpful assistant that provides accurate and concise information.",
        "temperature": 0.2,
    }

    response = nillion_2025_client.post("/v1/responses", json=payload, timeout=30)
    assert response.status_code == 200, (
        f"Standard request for {model} failed with status {response.status_code}"
    )

    response_json = response.json()
    print(response_json)
    assert "output" in response_json, "Response should contain output"
    assert len(response_json["output"]) > 0, (
        "At least one output item should be present"
    )

    text_items = [
        item for item in response_json["output"] if item.get("type") == "text"
    ]
    assert len(text_items) > 0, "Response should contain text items"
    content = text_items[0].get("text", "")
    assert content, f"No content returned for {model}"

    assert content.strip(), f"Empty response returned for {model}"

    assert response_json["usage"]["input_tokens"] > 0, f"Input tokens are 0 for {model}"
    assert response_json["usage"]["output_tokens"] > 0, (
        f"Output tokens are 0 for {model}"
    )
    assert response_json["usage"]["total_tokens"] > 0, f"Total tokens are 0 for {model}"

    print(
        f"\nModel {model} standard response: {content[:100]}..."
        if len(content) > 100
        else content
    )


@pytest.mark.parametrize("model", test_models)
def test_model_streaming_request(client, model):
    payload = {
        "model": model,
        "input": "Write a short poem about mountains.",
        "instructions": "You are a helpful assistant that provides accurate and concise information.",
        "temperature": 0.2,
        "stream": True,
    }

    with client.stream("POST", "/v1/responses", json=payload) as response:
        assert response.status_code == 200, (
            f"Streaming request for {model} failed with status {response.status_code}"
        )

        assert response.headers.get("Transfer-Encoding") == "chunked", (
            "Response should be streamed"
        )

        chunk_count = 0
        content = ""
        had_usage = False
        had_completed_event = False

        for chunk in response.iter_lines():
            if chunk and chunk.strip() and chunk.startswith("data:"):
                chunk_count += 1
                chunk_data = chunk[6:].strip()

                if chunk_data == "[DONE]":
                    continue

                print(f"\nModel {model} stream chunk {chunk_count}: {chunk_data}")
                chunk_json = json.loads(chunk_data)

                if chunk_json.get("type") == "response.text.delta":
                    delta = chunk_json.get("delta", "")
                    content += delta

                if chunk_json.get("type") == "response.output_item.added":
                    item = chunk_json.get("item", {})
                    if item.get("type") == "message" and isinstance(
                        item.get("content"), list
                    ):
                        for content_item in item["content"]:
                            if content_item.get("type") == "text":
                                content += content_item.get("text", "")

                if chunk_json.get("type") == "response.completed":
                    had_completed_event = True
                    if chunk_json.get("usage"):
                        print(f"Usage: {chunk_json.get('usage')}")
                        had_usage = True

        assert had_usage, f"No usage data received for {model} streaming request"
        assert had_completed_event, f"No completed event received for {model}"
        assert chunk_count > 0, f"No chunks received for {model} streaming request"
        print(f"Received {chunk_count} chunks for {model} streaming request")


@pytest.mark.parametrize("model", test_models)
def test_model_tools_request(client, model):
    payload = {
        "model": model,
        "input": "What is the weather like in Paris today?",
        "instructions": "You are a helpful assistant. When a user asks a question that requires calculation, use the execute_python tool to find the answer. After the tool provides its result, you must use that result to formulate a clear, final answer to the user's original question. Do not include any code or JSON in your final response.",
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
        response = client.post("/v1/responses", json=payload)
        assert response.status_code == 200, (
            f"Tools request for {model} failed with status {response.status_code}"
        )

        response_json = response.json()
        assert "output" in response_json, "Response should contain output"
        assert len(response_json["output"]) > 0, (
            "At least one output item should be present"
        )

        output = response_json["output"]

        tool_calls = [item for item in output if item.get("type") == "function_call"]

        if tool_calls:
            print(f"\nModel {model} tool calls: {json.dumps(tool_calls, indent=2)}")
            assert len(tool_calls) > 0, f"Tool calls array is empty for {model}"

            first_call = tool_calls[0]
            assert first_call.get("name") == "get_weather", (
                "Function name should be get_weather"
            )
            assert "arguments" in first_call, "Function should have arguments"

            args = json.loads(first_call["arguments"])
            assert "location" in args, "Arguments should contain location"
            assert "paris" in args["location"].lower(), "Location should be Paris"
        else:
            text_items = [item for item in output if item.get("type") == "text"]
            if text_items:
                content = text_items[0].get("text", "")
                print(
                    f"\nModel {model} response (no tool call): {content[:100]}..."
                    if len(content) > 100
                    else content
                )
                assert content, f"No content or tool calls returned for {model}"
    except Exception as e:
        print(f"\nError testing tools with {model}: {str(e)}")
        raise e


@pytest.mark.parametrize("model", test_models)
def test_function_calling_with_streaming_httpx(client, model):
    payload = {
        "model": model,
        "input": "What is the weather like in Paris today?",
        "instructions": "You are a helpful assistant that provides accurate and concise information.",
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

    with client.stream("POST", "/v1/responses", json=payload) as response:
        assert response.status_code == 200, (
            f"Streaming request for {model} failed with status {response.status_code}"
        )
        had_tool_call = False
        had_usage = False
        for line in response.iter_lines():
            if line and line.strip() and line.startswith("data:"):
                data_line = line[6:].strip()
                if data_line == "[DONE]":
                    continue
                try:
                    chunk_json = json.loads(data_line)
                except json.JSONDecodeError:
                    continue

                if chunk_json.get("type") == "response.function_call_arguments.delta":
                    had_tool_call = True

                if chunk_json.get("type") == "response.output_item.added":
                    item = chunk_json.get("item", {})
                    if item.get("type") == "function_call":
                        had_tool_call = True

                if chunk_json.get("type") == "response.completed":
                    usage = chunk_json.get("usage")
                    if usage:
                        had_usage = True

        assert had_tool_call, f"No tool calls received for {model} streaming request"
        assert had_usage, f"No usage data received for {model} streaming request"


def test_invalid_auth_token():
    invalid_client = httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": "Bearer invalid_token_123",
        },
        verify=False,
    )

    payload = {
        "model": test_models[0],
        "input": "Test",
    }

    response = invalid_client.post("/v1/responses", json=payload)
    assert response.status_code in [401, 403], (
        "Invalid token should result in unauthorized access"
    )


def test_rate_limiting(client):
    payload = {
        "model": test_models[0],
        "input": "Generate a short poem",
    }

    responses = []
    for _ in range(20):
        response = client.post("/v1/responses", json=payload)
        responses.append(response)

    rate_limit_statuses = [429, 403, 503]
    rate_limited_responses = [
        r for r in responses if r.status_code in rate_limit_statuses
    ]

    if len(rate_limited_responses) == 0:
        pytest.skip("No rate limiting detected. Manual review may be needed.")


@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_rate_limiting_nucs(rate_limited_client):
    payload = {
        "model": test_models[0],
        "input": "What is your name?",
    }

    responses = []
    for _ in range(4):
        response = rate_limited_client.post("/v1/responses", json=payload)
        responses.append(response)

    rate_limit_statuses = [429, 403, 503]
    rate_limited_responses = [
        r for r in responses if r.status_code in rate_limit_statuses
    ]

    assert len(rate_limited_responses) > 0, (
        "No NUC rate limiting detected, when expected"
    )


@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_rate_limiting_nucs(invalid_rate_limited_client):
    payload = {
        "model": test_models[0],
        "input": "What is your name?",
    }

    responses = []
    for _ in range(4):
        response = invalid_rate_limited_client.post("/v1/responses", json=payload)
        responses.append(response)

    rate_limit_statuses = [401]
    rate_limited_responses = [
        r for r in responses if r.status_code in rate_limit_statuses
    ]

    assert len(rate_limited_responses) > 0, (
        "No NUC rate limiting detected, when expected"
    )


@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_nildb_command_nucs(nildb_client):
    payload = {
        "model": test_models[0],
        "input": "What is your name?",
    }
    response = nildb_client.post("/v1/responses", json=payload)
    assert response.status_code == 401, "Invalid NILDB command should return 401"


def test_large_payload_handling(client):
    large_instructions = "Hello " * 10000

    payload = {
        "model": test_models[0],
        "input": "Respond briefly",
        "instructions": large_instructions,
        "max_output_tokens": 50,
    }

    response = client.post("/v1/responses", json=payload, timeout=30)
    print(response)

    assert response.status_code in [200, 413], (
        "Large payload should be handled gracefully"
    )

    if response.status_code == 200:
        response_json = response.json()
        assert "output" in response_json, "Response should contain output"
        assert len(response_json["output"]) > 0, (
            "At least one output item should be present"
        )


@pytest.mark.parametrize("invalid_model", ["nonexistent-model/v1", "", None, "   "])
def test_invalid_model_handling(client, invalid_model):
    payload = {
        "model": invalid_model,
        "input": "Test invalid model",
    }

    response = client.post("/v1/responses", json=payload)

    assert response.status_code in [400, 404], (
        f"Invalid model {invalid_model} should return an error"
    )


def test_timeout_handling(client):
    payload = {
        "model": test_models[0],
        "input": "Generate a very long response that might take a while",
        "max_output_tokens": 1000,
    }

    try:
        _ = client.post("/v1/responses", json=payload, timeout=0.1)
        pytest.fail("Request should have timed out")
    except httpx.TimeoutException:
        assert True, "Request timed out as expected"


def test_empty_input_handling(client):
    payload = {
        "model": test_models[0],
        "input": "",
    }

    response = client.post("/v1/responses", json=payload)
    print(response)

    assert response.status_code == 400, "Empty input should return a Bad Request"

    response_json = response.json()
    assert "detail" in response_json, "Error response should contain detail"


def test_unsupported_parameters(client):
    payload = {
        "model": test_models[0],
        "input": "Test unsupported parameters",
        "unsupported_param": "some_value",
        "another_weird_param": 42,
    }

    response = client.post("/v1/responses", json=payload)

    assert response.status_code in [200, 400], (
        "Unsupported parameters should be handled gracefully"
    )


def test_response_invalid_temperature(client):
    payload = {
        "model": test_models[0],
        "input": "What is the weather like?",
        "temperature": "hot",
    }
    response = client.post("/v1/responses", json=payload)
    print(response)
    assert response.status_code == 400, (
        "Invalid temperature type should return a 400 error"
    )


def test_response_missing_model(client):
    payload = {
        "input": "What is your name?",
        "temperature": 0.2,
    }
    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 400, (
        "Missing model should return a 400 validation error"
    )


def test_response_negative_max_tokens(client):
    payload = {
        "model": test_models[0],
        "input": "Tell me a joke.",
        "temperature": 0.2,
        "max_output_tokens": -10,
    }
    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 400, (
        "Negative max_output_tokens should return a 400 validation error"
    )


def test_response_high_temperature(client):
    payload = {
        "model": test_models[0],
        "input": "Write an imaginative story about a wizard.",
        "instructions": "You are a creative assistant.",
        "temperature": 2.0,
        "max_output_tokens": 50,
    }
    response = client.post("/v1/responses", json=payload)
    assert response.status_code == 200, (
        "High temperature request should return a valid response"
    )
    response_json = response.json()
    assert "output" in response_json, "Response should contain output"
    assert len(response_json["output"]) > 0, (
        "At least one output item should be present"
    )


def test_model_streaming_request_high_token(client):
    payload = {
        "model": test_models[0],
        "input": "Tell me a long story about a superhero's journey.",
        "instructions": "You are a creative assistant.",
        "temperature": 0.7,
        "max_output_tokens": 100,
        "stream": True,
    }
    with client.stream("POST", "/v1/responses", json=payload) as response:
        assert response.status_code == 200, (
            "Streaming with high max_output_tokens should return 200 status"
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
def test_nildb_delegation(client: httpx.Client):
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

    nuc_token_envelope = NucTokenEnvelope.parse(token)
    nilauth_public_keys = [
        Did(NilauthClient(CONFIG.nildb.nilauth_url).about().public_key.serialize())
    ]
    NucTokenValidator(nilauth_public_keys).validate(
        nuc_token_envelope, context={}, parameters=ValidationParameters.default()
    )


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC required for this tests on nilDB"
)
def test_nildb_prompt_document(document_id_client: httpx.Client, model):
    payload = {
        "model": model,
        "input": "Can you make a small rhyme?",
        "instructions": "You are a helpful assistant.",
        "temperature": 0.2,
    }

    response = document_id_client.post("/v1/responses", json=payload, timeout=30)

    assert response.status_code == 200, (
        f"Response should be successful: {response.text}"
    )

    response_json = response.json()
    text_items = [
        item for item in response_json["output"] if item.get("type") == "text"
    ]
    assert len(text_items) > 0, "Response should contain text items"

    message: str = text_items[0].get("text", "")
    assert "cheese" in message.lower(), "Response should contain cheese"


@pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"),
    reason="Requires E2B_API_KEY for code execution sandbox",
)
@pytest.mark.parametrize("model", test_models)
def test_execute_python_sha256_e2e(client, model):
    expected = "75cc238b167a05ab7336d773cb096735d459df2f0df9c8df949b1c44075df8a5"

    instructions = (
        "You are a helpful assistant. When a user asks a question that requires code execution, "
        "use the execute_python tool to find the answer. After the tool provides its result, "
        "you must use that result to formulate a clear, final answer to the user's original question. "
        "Do not include any code or JSON in your final response."
    )
    user_input = "Execute this exact Python code and return the result: import hashlib; print(hashlib.sha256('Nillion'.encode()).hexdigest())"

    payload = {
        "model": model,
        "input": user_input,
        "instructions": instructions,
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
        response = client.post("/v1/responses", json=payload)
        last_status = response.status_code
        if response.status_code != 200:
            continue
        data = response.json()
        last_data = data
        if not ("output" in data and data["output"]):
            continue

        text_items = [item for item in data["output"] if item.get("type") == "text"]
        if not text_items:
            continue

        content = text_items[0].get("text", "")
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
