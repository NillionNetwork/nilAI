import asyncio
import string
import random
from unittest.mock import MagicMock
from datetime import datetime, timedelta, timezone

from nilai_api.auth import TokenRateLimit, TokenRateLimits
import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

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
        day_limit=None,
        hour_limit=None,
        minute_limit=None,
        token_rate_limit=None,
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
            day_limit=10,
            hour_limit=None,
            minute_limit=None,
            token_rate_limit=None,
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            day_limit=None,
            hour_limit=11,
            minute_limit=None,
            token_rate_limit=None,
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            day_limit=None,
            hour_limit=None,
            minute_limit=12,
            token_rate_limit=None,
        ),
        UserRateLimits(
            subscription_holder=random_id(),
            day_limit=None,
            hour_limit=None,
            minute_limit=None,
            token_rate_limit=TokenRateLimits(
                limits=[
                    TokenRateLimit(
                        signature=random_id(),
                        usage_limit=11,
                        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                    )
                ]
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
async def test_web_search_rate_limit_hour(redis_client):
    """Verify that a user can only perform three web-search requests per hour."""
    from nilai_api.rate_limiting import web_search_rate_limit
    from nilai_api.auth.common import AuthenticationInfo
    from nilai_api.db.users import UserData

    # Build a dummy authenticated user
    user_id = random_id()
    user = UserData(
        userid=user_id,
        name="test",
        apikey=random_id(),
        prompt_tokens=0,
        completion_tokens=0,
        queries=0,
        signup_date=datetime.now(timezone.utc),
        ratelimit_day=None,
        ratelimit_hour=None,
        ratelimit_minute=None,
    )
    auth_info = AuthenticationInfo(user=user, token_rate_limit=None)

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

    # First three requests should succeed
    for _ in range(3):
        await web_search_rate_limit(mock_request, auth_info)

    # Fourth request should be rejected
    with pytest.raises(HTTPException):
        await web_search_rate_limit(mock_request, auth_info)
