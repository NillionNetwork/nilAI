import asyncio
import string
import random
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone

from nilai_api.auth import TokenRateLimit, TokenRateLimits
from nilai_api.db.users import RateLimits
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from nilai_api.rate_limiting import RateLimit, UserRateLimits, setup_redis_conn
from nilai_api.config import CONFIG


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
        user_id=random_id(),
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
async def test_web_search_rps_limit(redis_client):
    mock_request = MagicMock(spec=Request)
    mock_request.state.redis = redis_client[0]
    mock_request.state.redis_rate_limit_command = redis_client[1]
    # Ensure a clean slate for the global RPS key used by the limiter
    await redis_client[0].delete("web_search_rps")

    async def web_search_extractor(_):
        return True

    rate_limit = RateLimit(web_search_extractor=web_search_extractor)
    user_limits = UserRateLimits(
        user_id=random_id(),
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

    old_rps = CONFIG.web_search.rps
    CONFIG.web_search.rps = 2
    try:
        await consume_generator(rate_limit(mock_request, user_limits))
        await consume_generator(rate_limit(mock_request, user_limits))
        with pytest.raises(HTTPException):
            await consume_generator(rate_limit(mock_request, user_limits))
    finally:
        CONFIG.web_search.rps = old_rps
        await redis_client[0].delete("web_search_rps")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_limits",
    [
        UserRateLimits(
            user_id=random_id(),
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
            user_id=random_id(),
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
            user_id=random_id(),
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
            user_id=random_id(),
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
        user_id=apikey,
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
