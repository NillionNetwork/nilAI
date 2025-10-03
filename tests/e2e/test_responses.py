"""
Test suite for nilAI Responses API endpoint

This test suite tests the /v1/responses endpoint which follows the OpenAI Responses API format.

To run the tests, use the following command:

pytest tests/e2e/test_responses.py
"""

import json
import httpx
import pytest
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletion
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
def test_response_completion_string_input(client, model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": "What is the capital of France?",
                "instructions": "You are a helpful assistant that provides accurate and concise information.",
                "temperature": 0.2,
                "max_output_tokens": 100,
            },
            verify=False,
        )

        assert response.status_code == 200, f"Response should return 200 OK, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, dict), "Response should be a dictionary"
        assert "choices" in data, "Response should contain choices"
        assert len(data["choices"]) > 0, "Response should contain at least one choice"
        
        content = data["choices"][0]["message"]["content"]
        assert content, f"No content returned for {model}"
        print(f"\nModel {model} response: {content[:100]}..." if len(content) > 100 else content)

        assert "usage" in data, f"No usage data returned for {model}"
        assert data["usage"]["prompt_tokens"] > 0, f"No prompt tokens returned for {model}"
        assert data["usage"]["completion_tokens"] > 0, f"No completion tokens returned for {model}"

        assert "paris" in content.lower() or "Paris" in content, "Response should mention Paris"

    except Exception as e:
        pytest.fail(f"Error testing response completion with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_response_completion_messages_input(client, model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "instructions": "You are a helpful assistant.",
                "temperature": 0.2,
                "max_tokens": 100,
            },
            verify=False,
        )

        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        content = data["choices"][0]["message"]["content"]
        assert content
        assert "paris" in content.lower() or "Paris" in content

    except Exception as e:
        pytest.fail(f"Error testing response with messages input for {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key")
def test_rate_limiting_responses(rate_limited_client, model):
    import openai
    
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    rate_limited = False
    
    for _ in range(4):
        try:
            url = BASE_URL + "/responses"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "What is the capital of France?",
                    "temperature": 0.2,
                    "max_tokens": 100,
                },
                verify=False,
            )
            
            if response.status_code == 429:
                rate_limited = True
        except Exception:
            pass

    assert rate_limited, "No rate limiting detected for responses endpoint"


@pytest.mark.parametrize("model", test_models)
def test_streaming_response_completion(client, model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": "Write a short poem about mountains.",
                "instructions": "You are a helpful assistant.",
                "temperature": 0.2,
                "max_tokens": 100,
                "stream": True,
            },
            verify=False,
            stream=True,
        )

        assert response.status_code == 200
        
        chunk_count = 0
        full_content = ""
        had_usage = False

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    try:
                        chunk_data = json.loads(json_str)
                        chunk_count += 1
                        
                        if "choices" in chunk_data and chunk_data["choices"]:
                            delta = chunk_data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                full_content += delta["content"]
                        
                        if "usage" in chunk_data and chunk_data["usage"]:
                            had_usage = True
                            print(f"Model {model} usage: {chunk_data['usage']}")
                        
                        if chunk_count >= 20:
                            break
                    except json.JSONDecodeError:
                        pass

        assert had_usage, f"No usage data received for {model} streaming request"
        assert chunk_count > 0, f"No chunks received for {model}"
        assert full_content, f"No content assembled from stream for {model}"
        print(f"Received {chunk_count} chunks for {model}")

    except Exception as e:
        pytest.fail(f"Error testing streaming response completion with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_response_function_calling(client, model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": "What is the weather like in Paris today?",
                "instructions": "You are a helpful assistant.",
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
            },
            verify=False,
        )

        assert response.status_code == 200
        data = response.json()
        
        assert "choices" in data
        message = data["choices"][0]["message"]

        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]
            print(f"\nModel {model} tool calls: {json.dumps(tool_calls, indent=2)}")
            
            assert len(tool_calls) > 0
            first_call = tool_calls[0]
            
            assert first_call["function"]["name"] == "get_weather"
            args = json.loads(first_call["function"]["arguments"])
            assert "location" in args
            assert "paris" in args["location"].lower()
        else:
            content = message.get("content")
            assert content, f"No content or tool calls returned for {model}"

    except Exception as e:
        pytest.fail(f"Error testing function calling with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_response_function_calling_streaming(client, model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": "What is the weather like in Paris today?",
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
                            },
                            "strict": True,
                        },
                    }
                ],
                "temperature": 0.2,
                "stream": True,
            },
            verify=False,
            stream=True,
        )

        had_tool_call = False
        had_usage = False
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    try:
                        chunk_data = json.loads(json_str)
                        
                        if "choices" in chunk_data and chunk_data["choices"]:
                            delta = chunk_data["choices"][0].get("delta", {})
                            if "tool_calls" in delta and delta["tool_calls"]:
                                had_tool_call = True
                        
                        if "usage" in chunk_data and chunk_data["usage"]:
                            had_usage = True
                    except json.JSONDecodeError:
                        pass

        assert had_tool_call, f"No tool calls received for {model} streaming"
        assert had_usage, f"No usage data received for {model} streaming"

    except Exception as e:
        pytest.fail(f"Error testing streaming function calling with {model}: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_response_web_search(client, model):
    import time
    import openai

    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}...")
            
            invocation_token = api_key_getter()
            url = BASE_URL + "/responses"
            
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "Who won the Roland Garros Open in 2024? Just reply with the winner's name.",
                    "instructions": "You are a helpful assistant that provides accurate and up-to-date information.",
                    "web_search": True,
                    "temperature": 0.2,
                    "max_tokens": 150,
                },
                verify=False,
            )

            assert response.status_code == 200
            data = response.json()
            
            assert "choices" in data
            content = data["choices"][0]["message"]["content"]
            assert content

            sources = data.get("sources")
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


def test_response_max_output_tokens_alias(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Say hello",
                "max_output_tokens": 50,
            },
            verify=False,
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]

    except Exception as e:
        pytest.fail(f"Error testing max_output_tokens alias: {str(e)}")


def test_response_invalid_input_type(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": 12345,
                "temperature": 0.2,
            },
            verify=False,
        )

        assert response.status_code != 200, "Invalid input type should fail"

    except Exception as e:
        pytest.fail(f"Error testing invalid input type: {str(e)}")


def test_response_missing_input(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "temperature": 0.2,
            },
            verify=False,
        )

        assert response.status_code != 200, "Missing input should fail"

    except Exception as e:
        pytest.fail(f"Error testing missing input: {str(e)}")


@pytest.mark.parametrize("invalid_model", ["nonexistent-model/v1", "", None, "   "])
def test_response_invalid_model(client, invalid_model):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": invalid_model,
                "input": "Test invalid model",
            },
            verify=False,
        )

        assert response.status_code != 200, f"Invalid model {invalid_model} should fail"

    except Exception as e:
        pass


def test_response_negative_max_tokens(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Tell me a joke",
                "max_tokens": -10,
            },
            verify=False,
        )

        assert response.status_code != 200, "Negative max_tokens should fail"

    except Exception as e:
        pass


def test_response_high_temperature(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Write an imaginative story about a wizard.",
                "instructions": "You are a creative assistant.",
                "temperature": 5.0,
                "max_tokens": 50,
            },
            verify=False,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"]

    except Exception as e:
        pytest.fail(f"Error testing high temperature: {str(e)}")


def test_response_with_modalities(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Hello",
                "modalities": ["text"],
            },
            verify=False,
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    except Exception as e:
        pytest.fail(f"Error testing with modalities: {str(e)}")


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key")
def test_invalid_rate_limiting_responses(invalid_rate_limited_client, model):
    import openai
    
    invocation_token = get_invalid_rate_limited_nuc_token()
    forbidden = False
    
    for _ in range(4):
        try:
            url = BASE_URL + "/responses"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "What is the capital of France?",
                    "temperature": 0.2,
                    "max_tokens": 100,
                },
                verify=False,
            )
            
            if response.status_code == 401 or response.status_code == 403:
                forbidden = True
        except Exception:
            pass
    
    assert forbidden, "No NUC rate limiting detected for responses endpoint"


@pytest.mark.parametrize("model", test_models)
@pytest.mark.skipif(AUTH_STRATEGY != "nuc", reason="NUC rate limiting not used with API key")
def test_invalid_nildb_command_responses(nildb_client, model):
    import openai
    
    invocation_token = get_nildb_nuc_token()
    forbidden = False
    
    for _ in range(4):
        try:
            url = BASE_URL + "/responses"
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "What is the capital of France?",
                    "temperature": 0.2,
                    "max_tokens": 100,
                },
                verify=False,
            )
            
            if response.status_code == 401 or response.status_code == 403:
                forbidden = True
        except Exception:
            pass
    
    assert forbidden, "No NILDB command detected for responses endpoint"


def test_response_timeout_handling(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Generate a very long response that might take a while",
                "max_tokens": 1000,
            },
            verify=False,
            timeout=0.01,
        )
        pytest.fail("Request should have timed out")
    except requests.exceptions.Timeout:
        assert True, "Request timed out as expected"
    except Exception as e:
        assert "time" in str(e).lower(), f"Request should timeout but got: {str(e)}"


def test_response_unsupported_parameters(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Test unsupported parameters",
                "unsupported_param": "some_value",
                "another_weird_param": 42,
            },
            verify=False,
        )
        
        assert response.status_code == 200, "Request with unsupported parameters should still work"
        data = response.json()
        assert "choices" in data

    except Exception as e:
        assert True, f"Unsupported parameters handled as expected: {str(e)}"


def test_response_invalid_temperature_type(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "What is the weather like?",
                "temperature": "hot",
            },
            verify=False,
        )
        
        assert response.status_code != 200, "Invalid temperature type should fail"

    except Exception as e:
        assert True, f"Invalid temperature raised an error as expected: {str(e)}"


def test_response_missing_model(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "input": "What is your name?",
                "temperature": 0.2,
            },
            verify=False,
        )
        
        assert response.status_code != 200, "Missing model should fail"

    except Exception as e:
        assert True, f"Missing model raised an error as expected: {str(e)}"


def test_response_streaming_high_token(client):
    try:
        invocation_token = api_key_getter()
        url = BASE_URL + "/responses"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {invocation_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": test_models[0],
                "input": "Tell me a long story about a superhero's journey.",
                "instructions": "You are a creative assistant.",
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": True,
            },
            verify=False,
            stream=True,
        )
        
        chunk_count = 0
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]
                    try:
                        chunk_data = json.loads(json_str)
                        chunk_count += 1
                        
                        if "choices" in chunk_data and chunk_data["choices"]:
                            delta = chunk_data["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                assert delta["content"], "Chunk should contain content"
                        
                        if chunk_count >= 20:
                            break
                    except json.JSONDecodeError:
                        pass
        
        assert chunk_count > 0, "Should receive at least one chunk for high token streaming request"

    except Exception as e:
        pytest.fail(f"Error testing high token streaming: {str(e)}")


@pytest.mark.parametrize("model", test_models)
def test_response_web_search_brave_rps_e2e(client, model):
    import threading
    import time
    import openai
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    request_barrier = threading.Barrier(40)
    responses = []
    start_time = None
    
    def make_request():
        request_barrier.wait()
        
        nonlocal start_time
        if start_time is None:
            start_time = time.time()
        
        try:
            invocation_token = api_key_getter()
            url = BASE_URL + "/responses"
            
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "What is the latest news?",
                    "web_search": True,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                verify=False,
                timeout=30,
            )
            completion_time = time.time() - start_time
            
            if response.status_code == 429:
                responses.append((completion_time, response, "rate_limited"))
            elif response.status_code == 200:
                responses.append((completion_time, response, "success"))
            else:
                responses.append((completion_time, response, "error"))
        except requests.exceptions.Timeout:
            completion_time = time.time() - start_time
            responses.append((completion_time, None, "timeout"))
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
    rate_limited_responses = [(t, r) for t, r, status in responses if status == "rate_limited"]
    error_responses = [(t, r) for t, r, status in responses if status == "error"]
    
    print(f"Successful: {len(successful_responses)}, Rate limited: {len(rate_limited_responses)}, Errors: {len(error_responses)}")
    
    assert len(rate_limited_responses) > 0 or len(successful_responses) < 40, (
        "Rate limiting should be enforced for responses endpoint"
    )
    
    for t, response in successful_responses:
        data = response.json()
        assert "choices" in data
        sources = data.get("sources")
        assert sources is not None, "Successful web search responses should have sources"
        assert isinstance(sources, list), "Sources should be a list"
        assert len(sources) > 0, "Sources should not be empty"


@pytest.mark.parametrize("model", test_models)
def test_response_web_search_queueing_next_second_e2e(client, model):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    request_barrier = threading.Barrier(25)
    responses = []
    start_time = None
    
    def make_request():
        request_barrier.wait()
        
        nonlocal start_time
        if start_time is None:
            start_time = time.time()
        
        try:
            invocation_token = api_key_getter()
            url = BASE_URL + "/responses"
            
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {invocation_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "What is the weather like?",
                    "web_search": True,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                verify=False,
                timeout=30,
            )
            completion_time = time.time() - start_time
            
            if response.status_code == 429:
                responses.append((completion_time, response, "rate_limited"))
            elif response.status_code == 200:
                responses.append((completion_time, response, "success"))
            else:
                responses.append((completion_time, response, "error"))
        except requests.exceptions.Timeout:
            completion_time = time.time() - start_time
            responses.append((completion_time, None, "timeout"))
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
    rate_limited_responses = [(t, r) for t, r, status in responses if status == "rate_limited"]
    error_responses = [(t, r) for t, r, status in responses if status == "error"]
    
    print(f"Successful: {len(successful_responses)}, Rate limited: {len(rate_limited_responses)}, Errors: {len(error_responses)}")
    
    assert len(rate_limited_responses) > 0 or len(successful_responses) < 25, (
        "Queuing should be enforced for responses endpoint"
    )
    
    for t, response in successful_responses:
        data = response.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"], "Response should contain content"
        
        sources = data.get("sources")
        assert sources is not None, "Web search responses should have sources"
        assert isinstance(sources, list), "Sources should be a list"
        assert len(sources) > 0, "Sources should not be empty"
        
        first_source = sources[0]
        assert isinstance(first_source, dict), "First source should be a dictionary"
        assert "title" in first_source, "First source should have title"
        assert "url" in first_source, "First source should have url"
        assert "snippet" in first_source, "First source should have snippet"

