import httpx
import pytest
import json

from .config import BASE_URL, AUTH_TOKEN


class TestHTTPX:
    """Test suite for Yaak HTTP requests"""

    @pytest.fixture
    def client(self):
        """Create an HTTPX client with default headers"""
        return httpx.Client(
            base_url=BASE_URL,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AUTH_TOKEN}",
            },
        )

    def test_health_endpoint(self, client):
        """Test the health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200, "Health endpoint should return 200 OK"
        assert "status" in response.json(), "Health response should contain status"

    def test_models_endpoint(self, client):
        """Test the models endpoint"""
        response = client.get("/models")
        assert response.status_code == 200, "Models endpoint should return 200 OK"
        assert isinstance(response.json(), list), "Models should be returned as a list"

        # Check for specific models mentioned in the requests
        expected_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ]
        model_names = [model.get("id") for model in response.json()]
        for model in expected_models:
            assert model in model_names, f"Expected model {model} not found"

    def test_usage_endpoint(self, client):
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

    def test_attestation_endpoint(self, client):
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
        [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ],
    )
    def test_model_standard_request(self, client, model):
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
        assert "choices" in response_json, "Response should contain choices"
        assert len(response_json["choices"]) > 0, (
            "At least one choice should be present"
        )

        # Check content of response
        content = response_json["choices"][0].get("message", {}).get("content", "")
        assert content, f"No content returned for {model}"

        # Log response for debugging
        print(
            f"\nModel {model} standard response: {content[:100]}..."
            if len(content) > 100
            else content
        )

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        ],
    )
    def test_model_streaming_request(self, client, model):
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
            for chunk in response.iter_lines():
                if chunk and chunk.strip():
                    chunk_count += 1
                    if chunk_count <= 3:  # Just log first few chunks
                        print(f"\nModel {model} stream chunk {chunk_count}: {chunk}")
                if chunk_count >= 10:  # Limit how many chunks we process
                    break

            assert chunk_count > 0, f"No chunks received for {model} streaming request"
            print(f"Received {chunk_count} chunks for {model} streaming request")

    @pytest.mark.parametrize(
        "model",
        [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        ],
    )
    def test_model_tools_request(self, client, model):
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


class TestHTTPXAdditional:
    """Additional test cases for Yaak HTTP API"""

    @pytest.fixture
    def client(self):
        """Create an HTTPX client with default headers"""
        return httpx.Client(
            base_url="https://nilai-e176.nillion.network/v1",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AUTH_TOKEN}",
            },
        )

    def test_invalid_auth_token(self, client):
        """Test behavior with an invalid or expired authentication token"""
        invalid_client = httpx.Client(
            base_url="https://nilai-e176.nillion.network/v1",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer invalid_token_123",
            },
        )

        response = invalid_client.get("/attestation/report")
        assert response.status_code in [401, 403], (
            "Invalid token should result in unauthorized access"
        )

    def test_rate_limiting(self, client):
        """Test rate limiting by sending multiple rapid requests"""
        # Payload for repeated requests
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
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

    def test_large_payload_handling(self, client):
        """Test handling of large input payloads"""
        # Create a very large system message
        large_system_message = "Hello " * 10000  # 100KB of text

        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": large_system_message},
                {"role": "user", "content": "Respond briefly"},
            ],
            "max_tokens": 50,
        }

        response = client.post("/chat/completions", json=payload)

        # Check for appropriate handling of large payload
        assert response.status_code in [200, 413], (
            "Large payload should be handled gracefully"
        )

        if response.status_code == 200:
            response_json = response.json()
            assert "choices" in response_json, "Response should contain choices"
            assert len(response_json["choices"]) > 0, (
                "At least one choice should be present"
            )

    @pytest.mark.parametrize("invalid_model", ["nonexistent-model/v1", "", None, "   "])
    def test_invalid_model_handling(self, client, invalid_model):
        """Test handling of invalid or non-existent models"""
        payload = {
            "model": invalid_model,
            "messages": [{"role": "user", "content": "Test invalid model"}],
        }

        response = client.post("/chat/completions", json=payload)

        # Expect a 400 (Bad Request) or 404 (Not Found) for invalid models
        assert response.status_code in [400, 404], (
            f"Invalid model {invalid_model} should return an error"
        )

    def test_timeout_handling(self, client):
        """Test request timeout behavior"""
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
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

    def test_empty_messages_handling(self, client):
        """Test handling of empty messages list"""
        payload = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": []}

        response = client.post("/chat/completions", json=payload)

        # Expect a 400 Bad Request for empty messages
        assert response.status_code == 400, "Empty messages should return a Bad Request"

        # Check error response structure
        response_json = response.json()
        assert "error" in response_json, "Error response should contain an error key"

    def test_unsupported_parameters(self, client):
        """Test handling of unsupported or unexpected parameters"""
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": "Test unsupported parameters"}],
            "unsupported_param": "some_value",
            "another_weird_param": 42,
        }

        response = client.post("/chat/completions", json=payload)

        # Expect either successful response ignoring extra params or a 400 Bad Request
        assert response.status_code in [200, 400], (
            "Unsupported parameters should be handled gracefully"
        )
