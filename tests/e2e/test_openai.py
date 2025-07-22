"""
Test suite for nilAI OpenAI client

This test suite uses the OpenAI client to make requests to the nilAI API.

To run the tests, use the following command:

pytest tests/e2e/test_openai.py
"""

import json
import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
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


@pytest.fixture
def client():
    """Create an OpenAI client configured to use the Nilai API"""
    invocation_token: str = api_key_getter()
    return _create_openai_client(invocation_token)


@pytest.fixture
def rate_limited_client():
    """Create an OpenAI client configured to use the Nilai API with rate limiting"""
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    return _create_openai_client(invocation_token.token)


@pytest.fixture
def invalid_rate_limited_client():
    """Create an OpenAI client configured to use the Nilai API with rate limiting"""
    invocation_token = get_invalid_rate_limited_nuc_token()
    return _create_openai_client(invocation_token.token)


@pytest.fixture
def nildb_client():
    """Create an OpenAI client configured to use the Nilai API with rate limiting"""
    invocation_token = get_nildb_nuc_token()
    return _create_openai_client(invocation_token.token)


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_chat_completion(client, model):
    """Test basic chat completion with different models"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate and concise information.",
                },
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0.2,
            max_tokens=100,
        )

        # Verify response structure
        assert isinstance(response, ChatCompletion), (
            "Response should be a ChatCompletion object"
        )
        assert response.model == model, f"Response model should be {model}"
        assert len(response.choices) > 0, "Response should contain at least one choice"

        # Check content
        content = response.choices[0].message.content
        assert content, f"No content returned for {model}"
        print(
            f"\nModel {model} response: {content[:100]}..."
            if len(content) > 100
            else content
        )

        assert response.usage, f"No usage data returned for {model}"
        print(f"Model {model} usage: {response.usage}")

        assert response.usage.prompt_tokens > 0, (
            f"No prompt tokens returned for {model}"
        )
        assert response.usage.completion_tokens > 0, (
            f"No completion tokens returned for {model}"
        )
        assert response.usage.total_tokens > 0, f"No total tokens returned for {model}"

        # Check for Paris in the response
        assert "paris" in content.lower() or "Paris" in content, (
            "Response should mention Paris as the capital of France"
        )

    except Exception as e:
        pytest.fail(f"Error testing chat completion with {model}: {str(e)}")


@pytest.mark.parametrize(
    "model",
    test_models,
)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_rate_limiting_nucs(rate_limited_client, model):
    """Test rate limiting by sending multiple rapid requests"""
    import openai

    # Send multiple rapid requests
    rate_limited = False
    for _ in range(4):  # Adjust number based on expected rate limits
        try:
            _ = rate_limited_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate and concise information.",
                    },
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                temperature=0.2,
                max_tokens=100,
            )
        except openai.RateLimitError:
            rate_limited = True

    assert rate_limited, "No NUC rate limiting detected, when expected"


@pytest.mark.parametrize(
    "model",
    test_models,
)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_rate_limiting_nucs(invalid_rate_limited_client, model):
    """Test rate limiting by sending multiple rapid requests"""
    import openai

    # Send multiple rapid requests
    forbidden = False
    for _ in range(4):  # Adjust number based on expected rate limits
        try:
            _ = invalid_rate_limited_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate and concise information.",
                    },
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                temperature=0.2,
                max_tokens=100,
            )
        except openai.AuthenticationError:
            forbidden = True

    assert forbidden, "No NUC rate limiting detected, when expected"


@pytest.mark.parametrize(
    "model",
    test_models,
)
@pytest.mark.skipif(
    AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key"
)
def test_invalid_nildb_command_nucs(nildb_client, model):
    """Test rate limiting by sending multiple rapid requests"""
    import openai

    # Send multiple rapid requests
    forbidden = False
    for _ in range(4):  # Adjust number based on expected rate limits
        try:
            _ = nildb_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate and concise information.",
                    },
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                temperature=0.2,
                max_tokens=100,
            )
        except openai.AuthenticationError:
            forbidden = True

    assert forbidden, "No NILDB command detected, when expected"


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_streaming_chat_completion(client, model):
    """Test streaming chat completion with different models"""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate and concise information.",
                },
                {"role": "user", "content": "Write a short poem about mountains."},
            ],
            temperature=0.2,
            max_tokens=100,
            stream=True,
        )

        # Process the stream
        chunk_count = 0
        full_content = ""
        had_usage = False

        for chunk in stream:
            chunk_count += 1
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                full_content += content_piece

                print(f"Model {model} stream chunk {chunk_count}: {chunk}")
                if chunk.usage:
                    had_usage = True
                    print(f"Model {model} usage: {chunk.usage}")

            # Limit processing to avoid long tests
            if chunk_count >= 20:
                break
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
        pytest.fail(f"Error testing streaming chat completion with {model}: {str(e)}")


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_function_calling(client, model):
    """Test function calling with different models"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate and concise information.",
                },
                {
                    "role": "user",
                    "content": "What is the weather like in Paris today?",
                },
            ],
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

        # Verify response structure
        assert isinstance(response, ChatCompletion), (
            "Response should be a ChatCompletion object"
        )
        assert len(response.choices) > 0, "Response should contain at least one choice"

        message = response.choices[0].message

        # Check if the model used the tool
        if message.tool_calls:
            tool_calls = message.tool_calls
            print(
                f"\nModel {model} tool calls: {json.dumps([tc.model_dump() for tc in tool_calls], indent=2)}"
            )

            assert len(tool_calls) > 0, f"Tool calls array is empty for {model}"

            # Validate the first tool call
            first_call = tool_calls[0]
            assert first_call.function.name == "get_weather", (
                "Function name should be get_weather"
            )

            # Parse arguments and check for location
            args = json.loads(first_call.function.arguments)
            assert "location" in args, "Arguments should contain location"
            assert "paris" in args["location"].lower(), "Location should be Paris"

            # Test function response
            function_response = "The weather in Paris is currently 22°C and sunny."

            prompt = "You are Llama 1B, a detail-oriented AI tasked with verifying and analyzing the output of a recent tool call. Your first responsibility is to review, line by line, the produced output. Check that every section conforms to the expected format and contains all required information. Look for any discrepancies, missing data, or anomalies—be it in structure, content, or data types. Once you have completed your review, list any errors or inconsistencies found and suggest specific corrections if needed. Do not proceed with any further processing until you have fully validated and reported on the integrity of the tool calls output."
            follow_up_response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": "What is the weather like in Paris today?",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": first_call.id,
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": first_call.function.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": first_call.id,
                    },
                ],
                temperature=0.2,
            )

            follow_up_content = follow_up_response.choices[0].message.content
            assert follow_up_content, "No content in follow-up response"
            print(f"\nFollow-up response: {follow_up_content}")
            assert (
                "22°C" in follow_up_content
                or "sunny" in follow_up_content.lower()
                or "weather" in follow_up_content.lower()
            ), "Follow-up should mention the weather details"

        else:
            # If no tool calls, check content
            content = message.content
            if content:
                print(
                    f"\nModel {model} response (no tool call): {content[:100]}..."
                    if len(content) > 100
                    else content
                )
            assert content, f"No content or tool calls returned for {model}"

    except Exception as e:
        pytest.fail(f"Error testing function calling with {model}: {str(e)}")


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_function_calling_with_streaming(client, model):
    """Test function calling with different models"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate and concise information.",
                },
                {
                    "role": "user",
                    "content": "What is the weather like in Paris today?",
                },
            ],
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
        for chunk in response:
            print(f"Model {model} stream chunk: {chunk}")
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                assert chunk.choices[0].delta.tool_calls, "No tool calls in chunk"
                had_tool_call = True

            if chunk.usage:
                had_usage = True
                print(f"Model {model} usage: {chunk.usage}")

        assert had_tool_call, f"No tool calls received for {model} streaming request"
        assert had_usage, f"No usage data received for {model} streaming request"

    except Exception as e:
        pytest.fail(f"Error testing function calling with {model}: {str(e)}")


def test_usage_endpoint(client):
    """Test retrieving usage statistics"""
    try:
        # This is a custom endpoint, so we need to use a raw request
        # The OpenAI client doesn't have a built-in method for this
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

        # Check for expected keys
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
        # This is a custom endpoint, so we need to use a raw request
        import requests

        url = BASE_URL + "/attestation/report"
        invocation_token = api_key_getter()
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

        # Check for expected keys
        expected_keys = ["cpu_attestation", "gpu_attestation", "verifying_key"]
        for key in expected_keys:
            assert key in report, f"Expected key {key} not found in attestation report"

        print(f"\nAttestation report received with keys: {list(report.keys())}")

    except Exception as e:
        pytest.fail(f"Error testing attestation endpoint: {str(e)}")


def test_health_endpoint(client):
    """Test health check endpoint"""
    try:
        # This is a custom endpoint, so we need to use a raw request
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
        client.chat.completions.create(
            model=invalid_model,
            messages=[{"role": "user", "content": "Test invalid model"}],
        )
        pytest.fail(f"Invalid model {invalid_model} should raise an error")
    except Exception as e:
        # The OpenAI client will raise an exception for invalid models
        assert True, (
            f"Invalid model {invalid_model} raised an error as expected: {str(e)}"
        )


def test_timeout_handling(client):
    """Test request timeout behavior"""
    try:
        client.chat.completions.create(
            model=test_models[0],
            messages=[
                {
                    "role": "user",
                    "content": "Generate a very long response that might take a while",
                }
            ],
            max_tokens=1000,
            timeout=0.01,  # Very short timeout to force timeout scenario
        )
        pytest.fail("Request should have timed out")
    except Exception as e:
        # Timeout is the expected behavior
        print(f"Error: {e}")
        assert "time" in str(e).lower(), "Request timed out as expected"


def test_empty_messages_handling(client):
    """Test handling of empty messages list"""
    try:
        client.chat.completions.create(
            model=test_models[0],
            messages=[],
        )
        pytest.fail("Empty messages should raise an error")
    except Exception as e:
        # The OpenAI client will raise an exception for empty messages
        assert True, f"Empty messages raised an error as expected: {str(e)}"


def test_unsupported_parameters(client):
    """Test handling of unsupported or unexpected parameters"""
    try:
        # The OpenAI client will ignore unsupported parameters
        response = client.chat.completions.create(
            model=test_models[0],
            messages=[{"role": "user", "content": "Test unsupported parameters"}],
            unsupported_param="some_value",
            another_weird_param=42,
        )
        assert response, "Request with unsupported parameters should still work"
    except Exception as e:
        # Some unsupported parameters might cause errors, which is also acceptable
        assert True, f"Unsupported parameters handled as expected: {str(e)}"


def test_chat_completion_invalid_temperature(client):
    """Test chat completion with invalid temperature type that should trigger a validation error"""
    try:
        client.chat.completions.create(
            model=test_models[0],
            messages=[{"role": "user", "content": "What is the weather like?"}],
            temperature="hot",  # Invalid temperature type
        )
        pytest.fail("Invalid temperature type should raise an error")
    except Exception as e:
        # The OpenAI client will raise an exception for invalid temperature
        assert True, f"Invalid temperature raised an error as expected: {str(e)}"


def test_chat_completion_missing_model(client):
    """Test chat completion with missing model field to trigger a validation error"""
    try:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "What is your name?"}],
            temperature=0.2,
        )
        pytest.fail("Missing model should raise an error")
    except Exception as e:
        # The OpenAI client will raise an exception for missing model
        assert True, f"Missing model raised an error as expected: {str(e)}"


def test_chat_completion_negative_max_tokens(client):
    """Test chat completion with negative max_tokens value triggering a validation error"""
    try:
        client.chat.completions.create(
            model=test_models[0],
            messages=[{"role": "user", "content": "Tell me a joke."}],
            temperature=0.2,
            max_tokens=-10,  # Invalid negative value
        )
        pytest.fail("Negative max_tokens should raise an error")
    except Exception as e:
        # The OpenAI client will raise an exception for negative max_tokens
        assert True, f"Negative max_tokens raised an error as expected: {str(e)}"


def test_chat_completion_high_temperature(client):
    """Test chat completion with a high temperature value to check model's creative generation under extreme conditions"""
    response = client.chat.completions.create(
        model=test_models[0],
        messages=[
            {"role": "system", "content": "You are a creative assistant."},
            {
                "role": "user",
                "content": "Write an imaginative story about a wizard.",
            },
        ],
        temperature=5.0,  # Extremely high temperature for creative responses
        max_tokens=50,
    )
    assert response, "High temperature request should return a valid response"
    assert response.choices, "Response should contain choices"
    assert len(response.choices) > 0, "At least one choice should be present"
    assert response.choices[0].message.content, "Response should contain content"


def test_model_streaming_request_high_token(client):
    """Test streaming request with high max_tokens to verify response streaming over longer texts"""
    stream = client.chat.completions.create(
        model=test_models[0],
        messages=[
            {"role": "system", "content": "You are a creative assistant."},
            {
                "role": "user",
                "content": "Tell me a long story about a superhero's journey.",
            },
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if chunk.choices and chunk.choices[0].delta.content:
            assert chunk.choices[0].delta.content, "Chunk should contain content"
        if chunk_count >= 20:  # Limit processing to avoid long tests
            break
    assert chunk_count > 0, (
        "Should receive at least one chunk for high token streaming request"
    )


@pytest.mark.parametrize(
    "model",
    test_models,
)
def test_web_search_roland_garros_2024(client, model):
    """Test web_search using a query that requires up-to-date information (Roland Garros 2024 winner)."""
    max_retries = 3
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}...")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate and up-to-date information.",
                    },
                    {
                        "role": "user",
                        "content": "Who won the Roland Garros Open in 2024? Just reply with the winner's name.",
                    },
                ],
                extra_body={"web_search": True},
                temperature=0.2,
                max_tokens=150,
            )

            assert isinstance(response, ChatCompletion)
            assert response.model == model
            assert len(response.choices) > 0

            content = response.choices[0].message.content
            assert content, "Response content is empty."

            keywords = ["carlos", "alcaraz", "iga", "świątek", "swiatek"]
            assert any(k in content.lower() for k in keywords)

            print(f"Success on attempt {attempt + 1}")
            return
        except AssertionError as e:
            print(f"Assertion failed on attempt {attempt + 1}: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("All retries failed.")
                raise last_exception
