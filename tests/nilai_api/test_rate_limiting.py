import asyncio
import os
import string
import random
import time
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException, Request

from nilai_api.rate_limiting import RateLimit, UserRateLimits, setup_redis_conn
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="module", autouse=True)
def redis_server():
    container = RedisContainer()
    container.start()
    return container

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
    return ''.join(random.choice(letters) for _ in range(40))

async def consume_generator(gen):
    async for _ in gen:
        await asyncio.sleep(0.1) # mimic time to process the request


@pytest.mark.asyncio
async def test_concurrent_rate_limit(req):
    rate_limit = RateLimit(concurrent_extractor=lambda _: (5, "test"))

    user_limits = UserRateLimits(id=random_id(), day_limit=None, hour_limit=None, minute_limit=None)


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
@pytest.mark.parametrize("user_limits", [
    UserRateLimits(id=random_id(), day_limit=10, hour_limit=None, minute_limit=None),
    UserRateLimits(id=random_id(), day_limit=None, hour_limit=11, minute_limit=None),
    UserRateLimits(id=random_id(), day_limit=None, hour_limit=None, minute_limit=12)
])
async def test_user_limit(req, user_limits):
    rate_limit = RateLimit()

    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(10)]
    await asyncio.gather(*futures)


    futures = [consume_generator(rate_limit(req, user_limits)) for _ in range(3)]
    with pytest.raises(HTTPException):
        await asyncio.gather(*futures)