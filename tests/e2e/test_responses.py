import json
import os
import httpx
import pytest
import pytest_asyncio
from openai import OpenAI
from openai import AsyncOpenAI

from .config import BASE_URL, test_models, AUTH_STRATEGY, api_key_getter
from .nuc import (
    get_rate_limited_nuc_token,
    get_invalid_rate_limited_nuc_token,
    get_nildb_nuc_token,
)


def _create_openai_client(api_key: str) -> OpenAI:
    """Helper function to create an OpenAI client with SSL verification disabled"""
    transport = httpx.HTTPTransport(verify=False)
    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(transport=transport),
    )


def _create_async_openai_client(api_key: str) -> AsyncOpenAI:
    transport = httpx.AsyncHTTPTransport(verify=False)
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.fixture
def client():
    invocation_token: str = api_key_getter()
    return _create_openai_client(invocation_token)


@pytest_asyncio.fixture
async def async_client():
    invocation_token: str = api_key_getter()
    transport = httpx.AsyncHTTPTransport(verify=False)
    httpx_client = httpx.AsyncClient(transport=transport)
    client = AsyncOpenAI(
        base_url=BASE_URL, api_key=invocation_token, http_client=httpx_client
    )
    yield client
    await httpx_client.aclose()


@pytest.fixture
def rate_limited_client():
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    return _create_openai_client(invocation_token.token)


@pytest.fixture
def invalid_rate_limited_client():
    invocation_token = get_invalid_rate_limited_nuc_token()
    return _create_openai_client(invocation_token.token)


@pytest.fixture
def nildb_client():
    invocation_token = get_nildb_nuc_token()
    return _create_openai_client(invocation_token.token)


@pytest.fixture
def high_web_search_rate_limit(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_MINUTE", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_HOUR", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_DAY", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT", "9999")


@pytest.mark.parametrize("model", test_models)
def test_response_generation(client, model):
    """Test basic response generation with different models"""
    try:
        response = client.responses.create(
            model=model,
            input="What is the capital of France?",
            instructions="You are a helpful assistant that provides accurate and concise information.",
            temperature=0.2,
            max_output_tokens=100,
        )

        assert hasattr(response, "output"), "Response should contain output"
        assert hasattr(response, "signature"), "Response should contain signature"
        assert hasattr(response, "usage"), "Response should contain usage"
        assert response.model == model, f"Response model should be {model}"

        output = response.output
        assert isinstance(output, list), "Output should be a list"
        assert len(output) > 0, "Output should contain at least one item"

        message_item = next(
            (item for item in output if getattr(item, "type", None) == "message"), None
        )
        assert message_item is not None, "Output should contain a message item"

        message_content_list = getattr(message_item, "content", [])
        assert len(message_content_list) > 0, "Message item should have content"

        text_item = next(
            (
                c
                for c in message_content_list
                if getattr(c, "type", None) == "output_text"
            ),
            None,
        )
        assert text_item is not None, (
            "Message content should contain an output_text item"
        )

        content = getattr(text_item, "text", "")

        assert content, f"No content returned for {model}"
        print(
            f"\nModel {model} response: {content[:100]}..."
            if len(content) > 100
            else content
        )
        if model == "openai/gpt-oss-20b":
            return
        assert response.usage.input_tokens > 0, f"No input tokens returned for {model}"
        assert response.usage.output_tokens > 0, (
            f"No output tokens returned for {model}"
        )
        assert response.usage.total_tokens > 0, f"No total tokens returned for {model}"

        assert "paris" in content.lower(), (
            "Response should mention Paris as the capital of France"
        )

    except Exception as e:
        pytest.fail(f"Error testing response generation with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_rate_limiting_nucs(rate_limited_client, model):
    """Test rate limiting by sending multiple rapid requests"""
    import openai

    rate_limited = False
    for _ in range(4):
        try:
            _ = rate_limited_client.responses.create(
                model=model,
                input="What is the capital of France?",
                instructions="You are a helpful assistant that provides accurate and concise information.",
                temperature=0.2,
                max_output_tokens=100,
            )
        except openai.RateLimitError:
            rate_limited = True

    assert rate_limited, "No NUC rate limiting detected, when expected"


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_rate_limiting_nucs(invalid_rate_limited_client, model):
    """Test invalid rate limiting by sending multiple rapid requests"""
    import openai

    forbidden = False
    for _ in range(4):
        try:
            invalid_rate_limited_client.responses.create(
                model=model,
                input="What is the capital of France?",
                instructions="You are a helpful assistant that provides accurate and concise information.",
                temperature=0.2,
                max_output_tokens=100,
            )
        except openai.AuthenticationError:
            forbidden = True
            break

    assert forbidden, "No NUC rate limiting detected, when expected"


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_nildb_command_nucs(nildb_client, model):
    """Test invalid NILDB command handling"""
    import openai

    forbidden = False
    for _ in range(4):
        try:
            nildb_client.responses.create(
                model=model,
                input="What is the capital of France?",
                instructions="You are a helpful assistant that provides accurate and concise information.",
                temperature=0.2,
                max_output_tokens=100,
            )
        except openai.AuthenticationError:
            forbidden = True
            break

    assert forbidden, "No NILDB command detected, when expected"


@pytest.mark.parametrize("model", test_models)
def test_streaming_response(client, model):
    """Test streaming response generation with different models"""
    try:
        stream = client.responses.create(
            model=model,
            input="Write a short poem about mountains. It must be 20 words maximum.",
            instructions="You are a helpful assistant that provides accurate and concise information.",
            temperature=0.2,
            max_output_tokens=1000,
            stream=True,
        )

        chunk_count = 0
        full_content = ""
        had_usage = False

        for chunk in stream:
            chunk_count += 1
            print(f"Model {model} stream chunk {chunk_count}: {chunk}")

            if hasattr(chunk, "type"):
                if chunk.type == "response.output_item.added":
                    item = getattr(chunk, "item", None)
                    if item and hasattr(item, "type") and item.type == "message":
                        content_list = getattr(item, "content", [])
                        if isinstance(content_list, list):
                            for content_item in content_list:
                                if (
                                    hasattr(content_item, "type")
                                    and content_item.type == "text"
                                ):
                                    full_content += getattr(content_item, "text", "")

                if chunk.type == "response.output_text.delta":
                    delta = getattr(chunk, "delta", "")
                    full_content += delta

                if chunk.type == "response.completed":
                    response_obj = getattr(chunk, "response", None)
                    if response_obj:
                        usage = getattr(response_obj, "usage", None)
                        if usage:
                            had_usage = True
                            print(f"Model {model} usage: {usage}")

        assert had_usage, f"No usage data received for {model} streaming request"
        assert chunk_count > 0, f"No chunks received for {model} streaming request"
        assert full_content, f"No content assembled from stream for {model}"
        print(f"Received {chunk_count} chunks for {model} streaming request")
        print(
            f"Assembled content: {full_content[:100]}..."
            if len(full_content) > 100
            else full_content
        )

    except Exception as e:
        pytest.fail(f"Error testing streaming response with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_function_calling(client, model):
    """Test function calling with different models"""
    try:
        tools_def = [
            {
                "type": "function",
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
            }
        ]

        first = client.responses.create(
            model=model,
            input="What is the weather like in Paris today?",
            instructions="You are a helpful assistant that provides accurate and concise information.",
            tools=tools_def,
            tool_choice="auto",
            temperature=0.2,
        )

        assert hasattr(first, "output")
        calls = [o for o in first.output if getattr(o, "type", None) == "function_call"]

        if not calls:
            msg_items = [
                o for o in first.output if getattr(o, "type", None) == "message"
            ]
            if msg_items:
                parts = getattr(msg_items[0], "content", []) or []
                text = ""
                for p in parts:
                    t = getattr(p, "text", None) or (
                        p.get("text") if isinstance(p, dict) else None
                    )
                    if t:
                        text += t
                assert text, f"No content or tool calls returned for {model}"
                return
            texts = [o for o in first.output if getattr(o, "type", None) == "text"]
            assert texts, f"No content or tool calls returned for {model}"
            assert getattr(texts[0], "text", "")
            return

        fc = calls[0]
        assert getattr(fc, "name", None) == "get_weather"
        args_str = getattr(fc, "arguments", None)
        assert args_str
        args = json.loads(args_str)
        assert "location" in args
        assert "paris" in args["location"].lower()

        tool_result = "The weather in Paris is currently 22°C and sunny."
        prompt = (
            "You are Llama 1B, a detail-oriented AI tasked with verifying and analyzing the output of a recent tool call. "
            "Review the provided tool output and answer the user's question succinctly."
        )

        second = client.responses.create(
            model=model,
            input=[
                {"type": "message", "role": "system", "content": prompt},
                {
                    "type": "message",
                    "role": "user",
                    "content": "What is the weather like in Paris today?",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": f"Tool output for get_weather with arguments {json.dumps({'location': args['location']})}: {tool_result}",
                },
            ],
            temperature=0.2,
            tool_choice="auto",
        )

        out = second.output
        msg_items = [o for o in out if getattr(o, "type", None) == "message"]
        txt = ""
        if msg_items:
            parts = getattr(msg_items[0], "content", []) or []
            for p in parts:
                t = getattr(p, "text", None) or (
                    p.get("text") if isinstance(p, dict) else None
                )
                if t:
                    txt += t
        else:
            texts = [o for o in out if getattr(o, "type", None) == "text"]
            if texts:
                txt = getattr(texts[0], "text", "") or ""

        assert txt, "No content in follow-up response"
        assert ("22°C" in txt) or ("sunny" in txt.lower()) or ("weather" in txt.lower())

    except Exception as e:
        pytest.fail(f"Error testing function calling with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_function_calling_with_streaming(client, model):
    """Test function calling with streaming"""
    if model == "openai/gpt-oss-20b":
        pytest.skip(
            "Skipping test for openai/gpt-oss-20b model as it only supports non streaming with responsesendpoint"
        )

    try:
        stream = client.responses.create(
            model=model,
            input="What is the weather like in Paris today?",
            instructions="You are a helpful assistant that provides accurate and concise information.",
            tools=[
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
            temperature=0.2,
            stream=True,
        )

        had_tool_call = False
        had_usage = False

        for chunk in stream:
            print(f"Model {model} stream chunk: {chunk}")

            if hasattr(chunk, "type"):
                if chunk.type == "response.function_call_arguments.delta":
                    had_tool_call = True

                if chunk.type == "response.output_item.added":
                    item = getattr(chunk, "item", None)
                    if item and hasattr(item, "type") and item.type == "function_call":
                        had_tool_call = True

                if chunk.type == "response.completed":
                    response_obj = getattr(chunk, "response", None)
                    if response_obj:
                        usage = getattr(response_obj, "usage", None)
                        if usage:
                            had_usage = True
                            print(f"Model {model} usage: {usage}")

        assert had_tool_call, f"No tool calls received for {model} streaming request"
        assert had_usage, f"No usage data received for {model} streaming request"

    except Exception as e:
        pytest.fail(f"Error testing streaming function calling with {model}: {str(e)}")


def test_usage_endpoint(client):
    """Test retrieving usage statistics"""
    try:
        import requests

        invocation_token = api_key_getter()

        url = BASE_URL + "/usage"
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            verify=False,
        )
        assert response.status_code == 200, "Usage endpoint should return 200 OK"

        usage_data = response.json()
        assert isinstance(usage_data, dict), "Usage data should be a dictionary"

        expected_keys = [
            "total_tokens",
            "completion_tokens",
            "prompt_tokens",
            "queries",
        ]
        for key in expected_keys:
            assert key in usage_data, f"Expected key {key} not found in usage data"

        print(f"\nUsage data: {json.dumps(usage_data, indent=2)}")

    except Exception as e:
        pytest.fail(f"Error testing usage endpoint: {str(e)}")


def test_attestation_endpoint(client):
    """Test retrieving attestation report"""
    try:
        import requests

        invocation_token = api_key_getter()

        url = BASE_URL + "/attestation/report"
        response = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            params={"nonce": "0" * 64},
            verify=False,
        )

        assert response.status_code == 200, "Attestation endpoint should return 200 OK"

        report = response.json()
        assert isinstance(report, dict), "Attestation report should be a dictionary"

        expected_keys = ["cpu_attestation", "gpu_attestation", "verifying_key"]
        for key in expected_keys:
            assert key in report, f"Expected key {key} not found in attestation report"

        print(f"\nAttestation report received with keys: {list(report.keys())}")

    except Exception as e:
        pytest.fail(f"Error testing attestation endpoint: {str(e)}")


def test_health_endpoint(client):
    """Test health check endpoint"""
    try:
        import requests

        url = BASE_URL + "/health"
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            verify=False,
        )

        print(f"Health response: {response.status_code} {response.text}")
        assert response.status_code == 200, "Health endpoint should return 200 OK"

        health_data = response.json()
        assert isinstance(health_data, dict), "Health data should be a dictionary"
        assert "status" in health_data, "Health response should contain status"

        print(f"\nHealth status: {health_data.get('status')}")

    except Exception as e:
        pytest.fail(f"Error testing health endpoint: {str(e)}")


@pytest.mark.parametrize("invalid_model", ["nonexistent-model/v1", "", None, "   "])
def test_invalid_model_handling(client, invalid_model):
    """Test handling of invalid or non-existent models"""
    try:
        client.responses.create(
            model=invalid_model,
            input="Test invalid model",
        )
        pytest.fail(f"Invalid model {invalid_model} should raise an error")
    except Exception as e:
        assert True, (
            f"Invalid model {invalid_model} raised an error as expected: {str(e)}"
        )


def test_timeout_handling(client):
    """Test request timeout behavior"""
    try:
        client.responses.create(
            model=test_models[0],
            input="Generate a very long response that might take a while",
            max_output_tokens=1000,
            timeout=0.01,
        )
        pytest.fail("Request should have timed out")
    except Exception as e:
        assert "time" in str(e).lower(), "Request timed out as expected"


def test_empty_input_handling(client):
    """Test handling of empty input"""
    try:
        client.responses.create(
            model=test_models[0],
            input="",
        )
        pytest.fail("Empty input should raise an error")
    except Exception as e:
        assert True, f"Empty input raised an error as expected: {str(e)}"


def test_unsupported_parameters(client):
    """Test handling of unsupported or unexpected parameters"""
    try:
        response = client.responses.create(
            model=test_models[0],
            input="Test unsupported parameters",
            unsupported_param="some_value",
            another_weird_param=42,
        )
        assert response, "Request with unsupported parameters should still work"
    except Exception as e:
        assert True, f"Unsupported parameters handled as expected: {str(e)}"


def test_response_invalid_temperature(client):
    """Test response with invalid temperature type"""
    try:
        client.responses.create(
            model=test_models[0],
            input="What is the weather like?",
            temperature="hot",
        )
        pytest.fail("Invalid temperature type should raise an error")
    except Exception as e:
        assert True, f"Invalid temperature raised an error as expected: {str(e)}"


def test_response_missing_model(client):
    """Test response with missing model field"""
    try:
        client.responses.create(
            input="What is your name?",
            temperature=0.2,
        )
        pytest.fail("Missing model should raise an error")
    except Exception as e:
        assert True, f"Missing model raised an error as expected: {str(e)}"


def test_response_negative_max_tokens(client):
    """Test response with negative max_output_tokens value"""
    try:
        client.responses.create(
            model=test_models[0],
            input="Tell me a joke.",
            temperature=0.2,
            max_output_tokens=-10,
        )
        pytest.fail("Negative max_output_tokens should raise an error")
    except Exception as e:
        assert True, f"Negative max_output_tokens raised an error as expected: {str(e)}"


def test_response_high_temperature(client):
    """Test response with a high temperature value"""
    response = client.responses.create(
        model=test_models[0],
        input="Write an imaginative story about a wizard. Only write 10 words",
        instructions="You are a creative assistant.",
        temperature=1.30,
        max_output_tokens=1500,
    )

    assert response, "High temperature request should return a valid response"
    assert hasattr(response, "output"), "Response should contain output"
    assert len(response.output) > 0, "At least one output item should be present"

    message_items = [
        item for item in response.output if getattr(item, "type", None) == "message"
    ]

    assert len(message_items) > 0, "Response should contain a 'message' object"

    message = message_items[0]
    assert hasattr(message, "content") and len(message.content) > 0, (
        "Message object should have content"
    )

    final_text = getattr(message.content[0], "text", "")

    assert len(final_text) > 0, "The message content should not be empty"


def test_streaming_request_high_token(client):
    """Test streaming request with high max_output_tokens"""
    stream = client.responses.create(
        model=test_models[0],
        input="Tell me a long story about a superhero's journey.",
        instructions="You are a creative assistant.",
        temperature=0.7,
        max_output_tokens=100,
        stream=True,
    )

    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if hasattr(chunk, "type") and chunk.type == "response.text.delta":
            delta = getattr(chunk, "delta", None)
            assert delta is not None, "Chunk should contain delta"
        if chunk_count >= 20:
            break

    assert chunk_count > 0, (
        "Should receive at least one chunk for high token streaming request"
    )


@pytest.mark.parametrize("model", test_models)
def test_web_search(client, model, high_web_search_rate_limit):
    """Test web_search functionality with proper source validation"""

    response = client.responses.create(
        model=model,
        input="Who won the Roland Garros Open in 2024? Just reply with the winner's name.",
        instructions="You are a helpful assistant that provides accurate and up-to-date information. Answer in 10 words maximum and do not reason.",
        extra_body={"web_search": True},
        temperature=0.01,
    )

    assert response is not None, "Response should not be None"
    assert response.model == model, f"Response model should be {model}"
    assert hasattr(response, "output"), "Response should contain output"
    assert len(response.output) > 0, "Response should contain at least one output item"

    output_types = [getattr(item, "type", None) for item in response.output]

    message_items = [
        item for item in response.output if getattr(item, "type", None) == "message"
    ]
    assert len(message_items) > 0, (
        f"Response should contain message items. Found types: {output_types}"
    )

    message = message_items[0]
    assert hasattr(message, "content") and len(message.content) > 0, (
        "Message should have content"
    )

    text_item = next(
        (c for c in message.content if getattr(c, "type", None) == "output_text"), None
    )
    assert text_item is not None, "Message content should contain an output_text item"

    content = getattr(text_item, "text", "")
    assert content, "Response should contain content"

    sources = getattr(response, "sources", None)
    assert sources is not None, "Sources field should not be None"
    assert isinstance(sources, list), "Sources should be a list"
    assert len(sources) > 0, "Sources should not be empty"


@pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"),
    reason="Requires E2B_API_KEY for code execution sandbox",
)
@pytest.mark.parametrize("model", test_models)
def test_execute_python_sha256_e2e(client, model):
    """Test Python code execution via execute_python tool"""
    expected = "75cc238b167a05ab7336d773cb096735d459df2f0df9c8df949b1c44075df8a5"

    instructions = (
        "You are a helpful assistant. When a user asks a question that requires code execution, "
        "use the execute_python tool to find the answer. After the tool provides its result, "
        "reply with the value ONLY. No prose, no explanations, no code blocks, no JSON, no quotes."
    )
    user_input = (
        "Execute this exact Python code and return ONLY the result: "
        "import hashlib; print(hashlib.sha256('Nillion'.encode()).hexdigest())"
    )

    response = client.responses.create(
        model=model,
        input=user_input,
        instructions=instructions,
        temperature=0,
        tools=[
            {
                "type": "function",
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
            }
        ],
        tool_choice="auto",  # force the tool
    )

    # Must have exactly one text output item
    assert response.output[1].content[0].text, (
        f"No output. Full: {response.model_dump_json()}"
    )

    # Enforce "only the result": exact 64-hex chars and equals expected
    assert response.output[1].content[0].text == expected, (
        f"Got: {response.output[1].content[0].text!r}  Expected: {expected}"
    )
