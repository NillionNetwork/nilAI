import json

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
from .config import BASE_URL, AUTH_TOKEN, test_models


class TestOpenAIClient:
    """Test suite for Nilai API using the OpenAI client"""

    @pytest.fixture
    def client(self):
        """Create an OpenAI client configured to use the Nilai API"""
        return OpenAI(base_url=BASE_URL, api_key=AUTH_TOKEN)

    @pytest.mark.parametrize(
        "model",
        test_models,
    )
    def test_chat_completion(self, client, model):
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
            assert len(response.choices) > 0, (
                "Response should contain at least one choice"
            )

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
            assert response.usage.total_tokens > 0, (
                f"No total tokens returned for {model}"
            )

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
    def test_streaming_chat_completion(self, client, model):
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
            pytest.fail(
                f"Error testing streaming chat completion with {model}: {str(e)}"
            )

    @pytest.mark.parametrize(
        "model",
        test_models,
    )
    def test_function_calling(self, client, model):
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
            assert len(response.choices) > 0, (
                "Response should contain at least one choice"
            )

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
                    "22°C" in follow_up_content or "sunny" in follow_up_content.lower()
                ), "Follow-up should mention the weather details"

            else:
                # If no tool calls, check content
                content = message.content
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
    def test_function_calling_with_streaming(self, client, model):
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

            assert had_tool_call, (
                f"No tool calls received for {model} streaming request"
            )
            assert had_usage, f"No usage data received for {model} streaming request"

        except Exception as e:
            pytest.fail(f"Error testing function calling with {model}: {str(e)}")

    def test_usage_endpoint(self, client):
        """Test retrieving usage statistics"""
        try:
            # This is a custom endpoint, so we need to use a raw request
            # The OpenAI client doesn't have a built-in method for this
            import requests

            url = BASE_URL + "/usage"
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
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

    def test_attestation_endpoint(self, client):
        """Test retrieving attestation report"""
        try:
            # This is a custom endpoint, so we need to use a raw request
            import requests

            url = BASE_URL + "/attestation/report"
            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {AUTH_TOKEN}",
                    "Content-Type": "application/json",
                },
            )

            assert response.status_code == 200, (
                "Attestation endpoint should return 200 OK"
            )

            report = response.json()
            assert isinstance(report, dict), "Attestation report should be a dictionary"

            # Check for expected keys
            expected_keys = ["cpu_attestation", "gpu_attestation", "verifying_key"]
            for key in expected_keys:
                assert key in report, (
                    f"Expected key {key} not found in attestation report"
                )

            print(f"\nAttestation report received with keys: {list(report.keys())}")

        except Exception as e:
            pytest.fail(f"Error testing attestation endpoint: {str(e)}")

    def test_health_endpoint(self, client):
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
            )

            print(f"Health response: {response.status_code} {response.text}")
            assert response.status_code == 200, "Health endpoint should return 200 OK"

            health_data = response.json()
            assert isinstance(health_data, dict), "Health data should be a dictionary"
            assert "status" in health_data, "Health response should contain status"

            print(f"\nHealth status: {health_data.get('status')}")

        except Exception as e:
            pytest.fail(f"Error testing health endpoint: {str(e)}")


def run_tests():
    """Run the OpenAI client test suite"""
    pytest.main(["-xvs", __file__])


if __name__ == "__main__":
    run_tests()
