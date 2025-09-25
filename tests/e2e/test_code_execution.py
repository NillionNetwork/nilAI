import os
import json
import re

import httpx
import pytest

from .config import BASE_URL, test_models, api_key_getter


# Skip entire module if sandbox key not present
pytestmark = pytest.mark.skipif(
    not os.environ.get("E2B_API_KEY"),
    reason="Requires E2B_API_KEY for code execution sandbox",
)


@pytest.fixture
def client():
    try:
        token = api_key_getter()
    except Exception as e:
        pytest.skip(f"Skipping: missing auth token for e2e ({e})")
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        verify=False,
        timeout=None,
    )


@pytest.mark.parametrize("model", test_models)
def test_execute_python_sha256_e2e(client, model):
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

    # Retry up to 3 times since small models can be non-deterministic
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
