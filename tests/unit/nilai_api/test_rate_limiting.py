import asyncio
import string
import random
import time
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone

from nilai_api.auth import TokenRateLimit, TokenRateLimits
from nilai_api.config import CONFIG
from nilai_api.db.users import RateLimits
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from nilai_api.handlers.web_search import (
    _get_brave_rate_limiter,
    _make_brave_api_request,
)
from nilai_api.rate_limiting import RateLimit, UserRateLimits, setup_redis_conn


@pytest_asyncio.fixture
async def redis_client(redis_server):
    host_ip = redis_server.get_container_host_ip()
    host_port = redis_server.get_exposed_port(6379)
    return await setup_redis_conn(f"redis://{host_ip}:{host_port}")


@pytest.fixture
def req(redis_client) -> Request:
    mock_request = MagicMock(spec=Request)
    mock_request.url.path = "/test"
    mock_request.state.redis = redis_client[0]
    mock_request.state.redis_rate_limit_command = redis_client[1]
    return mock_request


def random_id() -> str:
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for _ in range(40))


async def consume_generator(gen):
    async for _ in gen:
        await asyncio.sleep(0.1)  # mimic time to process the request


@pytest.mark.asyncio
async def test_concurrent_rate_limit(req):
    rate_limit = RateLimit(concurrent_extractor=lambda _: (5, "test"))

    user_limits = UserRateLimits(
        subscription_holder=random_id(),
        token_rate_limit=None,
        rate_limits=RateLimits(
            user_rate_limit_day=None,
            user_rate_limit_hour=None,
            user_rate_limit_minute=None,
            web_search_rate_limit_day=None,
            web_search_rate_limit_hour=None,
            web_search_rate_limit_minute=None,
            user_rate_limit=None,
            web_search_rate_limit=None,
        ),
    )

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(5)]
    await asyncio.gather(*futures)

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(6)]
    with pytest.raises(HTTPException):
        await asyncio.gather(*futures)

    # we have to wait until all the futures in the previous step finish as the gather raises an exception as soon as one of the futures raises an exception
    await asyncio.sleep(0.2)

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(5)]
    await asyncio.gather(*futures)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_limits",
    [
        UserRateLimits(
            subscription_holder=random_id(),
            token_rate_limit=None,
            rate_limits=RateLimits(
                user_rate_limit_day=10,
                user_rate_limit_hour=None,
                user_rate_limit_minute=None,
                web_search_rate_limit_day=None,
                web_search_rate_limit_hour=None,
                web_search_rate_limit_minute=None,
                user_rate_limit=None,
                web_search_rate_limit=None,
            ),
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            token_rate_limit=None,
            rate_limits=RateLimits(
                user_rate_limit_day=None,
                user_rate_limit_hour=11,
                user_rate_limit_minute=None,
                web_search_rate_limit_day=None,
                web_search_rate_limit_hour=None,
                web_search_rate_limit_minute=None,
                user_rate_limit=None,
                web_search_rate_limit=None,
            ),
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            token_rate_limit=None,
            rate_limits=RateLimits(
                user_rate_limit_day=None,
                user_rate_limit_hour=None,
                user_rate_limit_minute=12,
                web_search_rate_limit_day=None,
                web_search_rate_limit_hour=None,
                web_search_rate_limit_minute=None,
                user_rate_limit=None,
                web_search_rate_limit=None,
            ),
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            token_rate_limit=TokenRateLimits(
                limits=[
                    TokenRateLimit(
                        signature=random_id(),
                        usage_limit=11,
                        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                    )
                ]
            ),
            rate_limits=RateLimits(
                user_rate_limit_day=None,
                user_rate_limit_hour=None,
                user_rate_limit_minute=None,
                web_search_rate_limit_day=None,
                web_search_rate_limit_hour=None,
                web_search_rate_limit_minute=None,
                user_rate_limit=None,
                web_search_rate_limit=None,
            ),
        ),
    ],
)
async def test_user_limit(req, user_limits):
    rate_limit = RateLimit()

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(10)]
    await asyncio.gather(*futures)

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(3)]
    with pytest.raises(HTTPException):
        await asyncio.gather(*futures)


@pytest.mark.asyncio
async def test_web_search_rate_limits(redis_client):
    """Verify that a user is rate limited for web-search requests across all time windows."""

    # Build a dummy authenticated user
    apikey = random_id()

    # Mock the incoming request with web_search enabled
    mock_request = MagicMock(spec=Request)
    mock_request.state.redis = redis_client[0]
    mock_request.state.redis_rate_limit_command = redis_client[1]

    async def json_body():
        return {
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "messages": [{"role": "user", "content": "hi"}],
            "web_search": True,
        }

    mock_request.json = json_body

    # Create rate limit with web search enabled
    async def web_search_extractor(request):
        return True

    rate_limit = RateLimit(web_search_extractor=web_search_extractor)
    user_limits = UserRateLimits(
        subscription_holder=apikey,
        token_rate_limit=None,
        rate_limits=RateLimits(
            user_rate_limit_day=None,
            user_rate_limit_hour=None,
            user_rate_limit_minute=None,
            web_search_rate_limit_day=72,
            web_search_rate_limit_hour=3,
            web_search_rate_limit_minute=1,
            user_rate_limit=None,
            web_search_rate_limit=None,
        ),
    )
    # First request should succeed (minute limit: 1, hour limit: 3, day limit: 72)
    await consume_generator(rate_limit(mock_request, user_limits))

    # Second request should be rejected due to minute limit (1 per minute)
    with pytest.raises(HTTPException):
        await consume_generator(rate_limit(mock_request, user_limits))


@pytest.mark.asyncio
async def test_web_search_rps_queues_when_exceeded(monkeypatch):
    """Ensure AsyncLimiter queues requests that exceed the configured RPS."""
    _get_brave_rate_limiter.cache_clear()
    call_times: list[float] = []

    async def fake_send_brave(query: str):
        timestamp = time.perf_counter()
        call_times.append(timestamp)
        await asyncio.sleep(0)  # yield control to mimic network scheduling
        return {"query": query}

    monkeypatch.setattr(CONFIG.web_search, "api_key", "test-key", raising=False)
    monkeypatch.setattr(CONFIG.web_search, "rps", 2, raising=False)
    monkeypatch.setattr(
        "nilai_api.handlers.web_search._send_brave_api_request",
        fake_send_brave,
    )

    try:
        await asyncio.gather(
            *[_make_brave_api_request(f"query-{idx}") for idx in range(4)]
        )
    finally:
        _get_brave_rate_limiter.cache_clear()

    assert len(call_times) == 4
    sorted_times = sorted(call_times)
    # First two calls run immediately; later calls must wait ~0.5s before firing.
    assert sorted_times[2] - sorted_times[0] >= 0.4
    assert sorted_times[3] - sorted_times[1] >= 0.4


@pytest.mark.asyncio
async def test_web_search_rps_shared_globally_across_users(monkeypatch):
    """Validate that the Brave limiter is shared globally across concurrent users."""
    _get_brave_rate_limiter.cache_clear()
    call_log: dict[str, float] = {}

    async def fake_send_brave(query: str):
        timestamp = time.perf_counter()
        call_log[query] = timestamp
        await asyncio.sleep(0)
        return {"query": query}

    monkeypatch.setattr(CONFIG.web_search, "api_key", "test-key", raising=False)
    monkeypatch.setattr(CONFIG.web_search, "rps", 2, raising=False)
    monkeypatch.setattr(
        "nilai_api.handlers.web_search._send_brave_api_request",
        fake_send_brave,
    )

    user_a_queries = [f"user-a-{i}" for i in range(2)]
    user_b_queries = [f"user-b-{i}" for i in range(2)]

    try:
        await asyncio.gather(*[_make_brave_api_request(q) for q in user_a_queries])
        await asyncio.gather(*[_make_brave_api_request(q) for q in user_b_queries])
    finally:
        _get_brave_rate_limiter.cache_clear()

    assert set(call_log) == set(user_a_queries + user_b_queries)
    latest_a = max(call_log[q] for q in user_a_queries)
    earliest_b = min(call_log[q] for q in user_b_queries)
    # Even though user B starts after A finished, they still wait for the global limiter slot.
    assert earliest_b - latest_a >= 0.4
