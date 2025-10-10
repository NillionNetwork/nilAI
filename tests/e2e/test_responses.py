import json
import os
import re
import httpx
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .config import BASE_URL, test_models, AUTH_STRATEGY, api_key_getter
from .nuc import (
    get_rate_limited_nuc_token,
    get_invalid_rate_limited_nuc_token,
    get_nildb_nuc_token,
)


def _create_openai_client(api_key: str) -> OpenAI:
    transport = httpx.HTTPTransport(verify=False)
    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(transport=transport),
    )


@pytest.fixture
def client():
    invocation_token: str = api_key_getter()
    return _create_openai_client(invocation_token)


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


@pytest.mark.parametrize("model", test_models)
def test_response_generation(client, model):
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

        text_items = [item for item in output if getattr(item, "type", None) == "text"]
        assert len(text_items) > 0, "Output should contain at least one text item"

        content = getattr(text_items[0], "text", "")
        assert content, f"No content returned for {model}"
        print(
            f"\nModel {model} response: {content[:100]}..."
            if len(content) > 100
            else content
        )

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
    import openai

    rate_limited = False
    for _ in range(4):
        try:
            rate_limited_client.responses.create(
                model=model,
                input="What is the capital of France?",
                instructions="You are a helpful assistant that provides accurate and concise information.",
                temperature=0.2,
                max_output_tokens=100,
            )
        except (openai.RateLimitError, openai.APIStatusError) as e:
            if hasattr(e, "status_code") and e.status_code in [429, 403, 503]:
                rate_limited = True
                break

    assert rate_limited, "No NUC rate limiting detected, when expected"


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_rate_limiting_nucs(invalid_rate_limited_client, model):
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
    try:
        stream = client.responses.create(
            model=model,
            input="Write a short poem about mountains.",
            instructions="You are a helpful assistant that provides accurate and concise information.",
            temperature=0.2,
            max_output_tokens=100,
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

                if chunk.type == "response.text.delta":
                    delta = getattr(chunk, "delta", "")
                    full_content += delta

                if chunk.type == "response.completed":
                    usage = getattr(chunk, "usage", None)
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
    try:
        response = client.responses.create(
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
        )

        assert hasattr(response, "output"), "Response should contain output"

        output = response.output
        tool_calls = [
            item for item in output if getattr(item, "type", None) == "function_call"
        ]

        if tool_calls:
            tool_calls_json = [
                {
                    "type": getattr(item, "type", None),
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None),
                    "call_id": getattr(item, "call_id", None),
                }
                for item in tool_calls
            ]
            print(
                f"\nModel {model} tool calls: {json.dumps(tool_calls_json, indent=2)}"
            )

            assert len(tool_calls) > 0, f"Tool calls array is empty for {model}"

            first_call = tool_calls[0]
            assert getattr(first_call, "name", None) == "get_weather", (
                "Function name should be get_weather"
            )

            arguments = getattr(first_call, "arguments", None)
            assert arguments, "Function should have arguments"

            args = json.loads(arguments)
            assert "location" in args, "Arguments should contain location"
            assert "paris" in args["location"].lower(), "Location should be Paris"

            function_response = "The weather in Paris is currently 22°C and sunny."

            follow_up_response = client.responses.create(
                model=model,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": "What is the weather like in Paris today?",
                    },
                    {
                        "type": "function_call",
                        "name": getattr(first_call, "name", None),
                        "arguments": arguments,
                        "call_id": getattr(first_call, "call_id", None),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": getattr(first_call, "call_id", None),
                        "output": json.dumps({"result": function_response}),
                    },
                ],
                instructions="You are Llama 1B, a detail-oriented AI tasked with verifying and analyzing the output of a recent tool call. Your first responsibility is to review, line by line, the produced output. Check that every section conforms to the expected format and contains all required information. Look for any discrepancies, missing data, or anomalies—be it in structure, content, or data types. Once you have completed your review, list any errors or inconsistencies found and suggest specific corrections if needed. Do not proceed with any further processing until you have fully validated and reported on the integrity of the tool calls output.",
                temperature=0.2,
            )

            output = follow_up_response.output
            text_items = [
                item for item in output if getattr(item, "type", None) == "text"
            ]

            if text_items:
                follow_up_content = getattr(text_items[0], "text", "")
                assert follow_up_content, "No content in follow-up response"
                print(f"\nFollow-up response: {follow_up_content}")
                assert (
                    "22°C" in follow_up_content
                    or "sunny" in follow_up_content.lower()
                    or "weather" in follow_up_content.lower()
                ), "Follow-up should mention the weather details"
        else:
            text_items = [
                item for item in output if getattr(item, "type", None) == "text"
            ]
            if text_items:
                content = getattr(text_items[0], "text", "")
                print(
                    f"\nModel {model} response (no tool call): {content[:100]}..."
                    if len(content) > 100
                    else content
                )
                assert content, f"No content or tool calls returned for {model}"

    except Exception as e:
        pytest.fail(f"Error testing function calling with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_function_calling_with_streaming(client, model):
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
                    usage = getattr(chunk, "usage", None)
                    if usage:
                        had_usage = True
                        print(f"Model {model} usage: {usage}")

        assert had_tool_call, f"No tool calls received for {model} streaming request"
        assert had_usage, f"No usage data received for {model} streaming request"

    except Exception as e:
        pytest.fail(f"Error testing streaming function calling with {model}: {str(e)}")


def test_usage_endpoint(client):
    try:
        url = BASE_URL + "/usage"
        response = client.get(url)
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
    try:
        url = BASE_URL + "/attestation/report"
        response = client.get(url, params={"nonce": "0" * 64})

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
    try:
        url = BASE_URL + "/health"
        response = client.get(url)

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
    try:
        client.responses.create(
            model=test_models[0],
            input="",
        )
        pytest.fail("Empty input should raise an error")
    except Exception as e:
        assert True, f"Empty input raised an error as expected: {str(e)}"


def test_unsupported_parameters(client):
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
    try:
        client.responses.create(
            input="What is your name?",
            temperature=0.2,
        )
        pytest.fail("Missing model should raise an error")
    except Exception as e:
        assert True, f"Missing model raised an error as expected: {str(e)}"


def test_response_negative_max_tokens(client):
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
    response = client.responses.create(
        model=test_models[0],
        input="Write an imaginative story about a wizard.",
        instructions="You are a creative assistant.",
        temperature=2.0,
        max_output_tokens=50,
    )

    assert response, "High temperature request should return a valid response"
    assert hasattr(response, "output"), "Response should contain output"
    assert len(response.output) > 0, "At least one output item should be present"

    text_items = [
        item for item in response.output if getattr(item, "type", None) == "text"
    ]
    assert len(text_items) > 0, "Response should contain text content"
    assert getattr(text_items[0], "text", None), "Response should contain text"


def test_streaming_request_high_token(client):
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
def test_web_search(client, model):
    import time

    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}...")

            response = client.responses.create(
                model=model,
                input="Who won the Roland Garros Open in 2024? Just reply with the winner's name.",
                instructions="You are a helpful assistant that provides accurate and up-to-date information.",
                extra_body={"web_search": True},
                temperature=0.2,
                max_output_tokens=150,
            )

            assert response.model == model, f"Response model should be {model}"
            assert hasattr(response, "output"), "Response should contain output"
            assert len(response.output) > 0, (
                "Response should contain at least one output item"
            )

            text_items = [
                item
                for item in response.output
                if getattr(item, "type", None) == "text"
            ]
            assert len(text_items) > 0, "Response should contain text content"
            content = getattr(text_items[0], "text", "")
            assert content, "Response should contain content"

            sources = getattr(response, "sources", None)
            assert sources is not None, "Sources field should not be None"
            assert isinstance(sources, list), "Sources should be a list"
            assert len(sources) > 0, "Sources should not be empty"

            print(f"Success on attempt {attempt + 1}")
            return
        except AssertionError as e:
            print(f"Assertion failed on attempt {attempt + 1}: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(1)
            else:
                print("All retries failed.")
                raise last_exception


def test_web_search_brave_rps_e2e(client):
    import openai

    request_barrier = threading.Barrier(40)
    responses = []
    start_time = None

    def make_request():
        request_barrier.wait()

        nonlocal start_time
        if start_time is None:
            start_time = time.time()

        try:
            response = client.responses.create(
                model=test_models[0],
                input="What is the latest news?",
                extra_body={"web_search": True},
                max_output_tokens=10,
                temperature=0.0,
            )
            completion_time = time.time() - start_time
            responses.append((completion_time, response, "success"))
        except (openai.RateLimitError, openai.APIStatusError) as e:
            completion_time = time.time() - start_time
            if hasattr(e, "status_code") and e.status_code in [429, 503]:
                responses.append((completion_time, e, "rate_limited"))
            else:
                responses.append((completion_time, e, "error"))
        except Exception as e:
            completion_time = time.time() - start_time
            responses.append((completion_time, e, "error"))

    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(make_request) for _ in range(40)]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread execution error: {e}")

    assert len(responses) == 40, "All requests should complete"

    successful_responses = [(t, r) for t, r, status in responses if status == "success"]
    rate_limited_responses = [
        (t, r) for t, r, status in responses if status == "rate_limited"
    ]
    error_responses = [(t, r) for t, r, status in responses if status == "error"]

    print(
        f"Successful: {len(successful_responses)}, Rate limited: {len(rate_limited_responses)}, Errors: {len(error_responses)}"
    )

    assert len(rate_limited_responses) > 0 or len(successful_responses) < 40, (
        "Rate limiting should be enforced - either some requests should be rate limited or delayed"
    )

    for t, response in successful_responses:
        sources = getattr(response, "sources", None)
        assert sources is not None, (
            "Successful web search responses should have sources"
        )
        assert isinstance(sources, list), "Sources should be a list"
        assert len(sources) > 0, "Sources should not be empty"

    for t, error in rate_limited_responses:
        assert hasattr(error, "status_code") and error.status_code in [429, 503], (
            "Rate limited responses should have 429 or 503 status"
        )


def test_web_search_queueing_next_second_e2e(client):
    import openai

    request_barrier = threading.Barrier(25)
    responses = []
    start_time = None

    def make_request():
        request_barrier.wait()

        nonlocal start_time
        if start_time is None:
            start_time = time.time()

        try:
            response = client.responses.create(
                model=test_models[0],
                input="What is the weather like?",
                extra_body={"web_search": True},
                max_output_tokens=10,
                temperature=0.0,
            )
            completion_time = time.time() - start_time
            responses.append((completion_time, response, "success"))
        except (openai.RateLimitError, openai.APIStatusError) as e:
            completion_time = time.time() - start_time
            if hasattr(e, "status_code") and e.status_code in [429, 503]:
                responses.append((completion_time, e, "rate_limited"))
            else:
                responses.append((completion_time, e, "error"))
        except Exception as e:
            completion_time = time.time() - start_time
            responses.append((completion_time, e, "error"))

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(make_request) for _ in range(25)]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread execution error: {e}")

    assert len(responses) == 25, "All requests should complete"

    successful_responses = [(t, r) for t, r, status in responses if status == "success"]
    rate_limited_responses = [
        (t, r) for t, r, status in responses if status == "rate_limited"
    ]
    error_responses = [(t, r) for t, r, status in responses if status == "error"]

    print(
        f"Successful: {len(successful_responses)}, Rate limited: {len(rate_limited_responses)}, Errors: {len(error_responses)}"
    )

    assert len(rate_limited_responses) > 0 or len(successful_responses) < 25, (
        "Queuing should be enforced - either some requests should be rate limited or delayed"
    )

    for t, response in successful_responses:
        assert hasattr(response, "output"), "Response should contain output"
        assert len(response.output) > 0, (
            "Response should contain at least one output item"
        )

        text_items = [
            item for item in response.output if getattr(item, "type", None) == "text"
        ]
        assert len(text_items) > 0, "Response should contain text content"
        assert getattr(text_items[0], "text", None), "Response should contain text"

        sources = getattr(response, "sources", None)
        assert sources is not None, "Web search responses should have sources"
        assert isinstance(sources, list), "Sources should be a list"
        assert len(sources) > 0, "Sources should not be empty"

        first_source = sources[0]
        assert isinstance(first_source, dict), "First source should be a dictionary"
        assert "title" in first_source, "First source should have title"
        assert "url" in first_source, "First source should have url"
        assert "snippet" in first_source, "First source should have snippet"

    for t, error in rate_limited_responses:
        assert hasattr(error, "status_code") and error.status_code in [429, 503], (
            "Rate limited responses should have 429 or 503 status"
        )


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

    trials = 3
    escaped_expected = re.escape(expected)
    pattern = rf"\b{escaped_expected}\b"
    last_response = None
    last_content = ""

    for _ in range(trials):
        response = client.responses.create(
            model=model,
            input=user_input,
            instructions=instructions,
            temperature=0,
            tools=[
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
        )
        last_response = response

        if not response.output:
            continue

        text_items = [
            item for item in response.output if getattr(item, "type", None) == "text"
        ]
        if not text_items:
            continue

        content = getattr(text_items[0], "text", "")
        last_content = content
        normalized_content = re.sub(r"\s+", " ", content)

        if re.search(pattern, normalized_content):
            break
    else:
        pytest.fail(
            (
                "Expected exact SHA-256 hash not found after retries.\n"
                f"Got: {last_content[:200]}...\n"
                f"Expected: {expected}\n"
                f"Full: {last_response.model_dump_json() if last_response else '<no response>'}"
            )
        )
