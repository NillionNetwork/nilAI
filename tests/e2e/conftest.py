from .config import BASE_URL, api_key_getter
from .nuc import (
    get_rate_limited_nuc_token,
    get_invalid_rate_limited_nuc_token,
    get_document_id_nuc_token,
)
import httpx
import pytest
import pytest_asyncio
from openai import OpenAI, AsyncOpenAI


# ============================================================================
# HTTP Client Fixtures (for test_chat_completions_http.py, test_responses_http.py)
# ============================================================================


@pytest.fixture
def http_client():
    """Create an HTTPX client with default headers for HTTP-based tests"""
    invocation_token: str = api_key_getter()
    print("invocation_token", invocation_token)
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


# Alias for backward compatibility
client = http_client


@pytest.fixture
def rate_limited_http_client():
    """Create an HTTPX client with rate limiting for HTTP-based tests"""
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token}",
        },
        timeout=None,
        verify=False,
    )


# Alias for backward compatibility
rate_limited_client = rate_limited_http_client


@pytest.fixture
def invalid_rate_limited_http_client():
    """Create an HTTPX client with invalid rate limiting for HTTP-based tests"""
    invocation_token = get_invalid_rate_limited_nuc_token()
    return httpx.Client(
        base_url=BASE_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {invocation_token}",
        },
        timeout=None,
        verify=False,
    )


# Alias for backward compatibility
invalid_rate_limited_client = invalid_rate_limited_http_client


@pytest.fixture
def nillion_2025_client():
    """Create an HTTPX client with default headers"""
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
    """Create an HTTPX client with default headers"""
    invocation_token = get_document_id_nuc_token()
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


# ============================================================================
# OpenAI SDK Client Fixtures (for test_chat_completions.py, test_responses.py)
# ============================================================================


def _create_openai_client(api_key: str) -> OpenAI:
    """Helper function to create an OpenAI client with SSL verification disabled"""
    transport = httpx.HTTPTransport(verify=False)
    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.Client(transport=transport),
    )


def _create_async_openai_client(api_key: str) -> AsyncOpenAI:
    """Helper function to create an async OpenAI client with SSL verification disabled"""
    transport = httpx.AsyncHTTPTransport(verify=False)
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.fixture
def openai_client():
    """Create an OpenAI SDK client configured to use the Nilai API"""
    invocation_token: str = api_key_getter()
    return _create_openai_client(invocation_token)


@pytest_asyncio.fixture
async def async_openai_client():
    """Create an async OpenAI SDK client configured to use the Nilai API"""
    invocation_token: str = api_key_getter()
    transport = httpx.AsyncHTTPTransport(verify=False)
    httpx_client = httpx.AsyncClient(transport=transport)
    client = AsyncOpenAI(
        base_url=BASE_URL, api_key=invocation_token, http_client=httpx_client
    )
    yield client
    await httpx_client.aclose()


@pytest.fixture
def rate_limited_openai_client():
    """Create an OpenAI SDK client with rate limiting"""
    invocation_token = get_rate_limited_nuc_token(rate_limit=1)
    return _create_openai_client(invocation_token)


@pytest.fixture
def invalid_rate_limited_openai_client():
    """Create an OpenAI SDK client with invalid rate limiting"""
    invocation_token = get_invalid_rate_limited_nuc_token()
    return _create_openai_client(invocation_token)


@pytest.fixture
def document_id_openai_client():
    """Create an OpenAI SDK client with document ID token"""
    invocation_token = get_document_id_nuc_token()
    return _create_openai_client(invocation_token)


@pytest.fixture
def high_web_search_rate_limit(monkeypatch):
    """Set high rate limits for web search for RPS tests"""
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_MINUTE", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_HOUR", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT_DAY", "9999")
    monkeypatch.setenv("WEB_SEARCH_RATE_LIMIT", "9999")
    monkeypatch.setenv("USER_RATE_LIMIT_MINUTE", "9999")
    monkeypatch.setenv("USER_RATE_LIMIT_HOUR", "9999")
    monkeypatch.setenv("USER_RATE_LIMIT_DAY", "9999")
    monkeypatch.setenv("USER_RATE_LIMIT", "9999")
    monkeypatch.setenv(
        "MODEL_CONCURRENT_RATE_LIMIT",
        (
            '{"meta-llama/Llama-3.2-1B-Instruct": 500, '
            '"meta-llama/Llama-3.2-3B-Instruct": 500, '
            '"meta-llama/Llama-3.1-8B-Instruct": 300, '
            '"cognitivecomputations/Dolphin3.0-Llama3.1-8B": 300, '
            '"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 50, '
            '"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": 50, '
            '"openai/gpt-oss-20b": 500, '
            '"google/gemma-3-27b-it": 500, '
            '"default": 500}'
        ),
    )


# ============================================================================
# Convenience Aliases for OpenAI SDK Tests
# These allow test files to use 'client' instead of 'openai_client'
# Note: These will be shadowed by local fixtures in test_chat_completions.py
# and test_responses.py if those files redefine them
# ============================================================================

# Uncomment these if you want to use the conftest fixtures without shadowing:
# client = openai_client
# async_client = async_openai_client
# rate_limited_client = rate_limited_openai_client
# invalid_rate_limited_client = invalid_rate_limited_openai_client
# nildb_client = document_id_openai_client
