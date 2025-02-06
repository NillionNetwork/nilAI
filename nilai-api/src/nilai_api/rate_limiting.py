from asyncio import iscoroutine
from typing import Callable, Tuple, Awaitable, Annotated

from pydantic import BaseModel

from fastapi.params import Depends
from fastapi import status, HTTPException, Request
from redis.asyncio import from_url, Redis

from nilai_api.auth import get_user
from nilai_api.db import UserModel

LUA_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire_time = ARGV[2]

local current = tonumber(redis.call('get', key) or "0")
if current > 0 then
    if current + 1 > limit then
        return redis.call("PTTL", key)
    else
        redis.call("INCR", key)
        return 0
    end
else
    redis.call("SET", key, 1, "px", expire_time)
    return 0
end
"""

DAY_MS = 24 * 60 * 60 * 1000
HOUR_MS = 60 * 60 * 1000
MINUTE_MS = 60 * 1000


async def setup_redis_conn(redis_url):
    client = from_url(redis_url, encoding="utf8")
    lua_sha = await client.script_load(LUA_RATE_LIMIT_SCRIPT)
    return client, lua_sha


class UserRateLimits(BaseModel):
    id: str
    day_limit: int | None
    hour_limit: int | None
    minute_limit: int | None


def get_user_limits(user: Annotated[UserModel, Depends(get_user)]) -> UserRateLimits:
    return UserRateLimits(
        id=user.userid,
        day_limit=user.ratelimit_day,
        hour_limit=user.ratelimit_hour,
        minute_limit=user.ratelimit_minute,
    )


class RateLimit:
    def __init__(
        self,
        concurrent: int | None = None,
        concurrent_extractor: Callable[
            [Request], Tuple[int, str] | Awaitable[Tuple[int, str]]
        ]
        | None = None,
    ):
        """
        concurrent: Maximum number of concurrent requests allowed for a single path
        concurrent_extractor: A callable that extracts the concurrent limit and key from the request

        concurrent and concurrent_extractor are mutually exclusive
        """
        self.max_concurrent = concurrent
        self.concurrent_extractor = concurrent_extractor

    async def __call__(
        self,
        request: Request,
        user_limits: Annotated[UserRateLimits, Depends(get_user_limits)],
    ):
        redis = request.state.redis
        redis_rate_limit_command = request.state.redis_rate_limit_command
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"minute:{user_limits.id}",
            user_limits.minute_limit,
            MINUTE_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"hour:{user_limits.id}",
            user_limits.hour_limit,
            HOUR_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"day:{user_limits.id}",
            user_limits.day_limit,
            DAY_MS,
        )
        key = await self.check_concurrent_and_increment(redis, request)
        try:
            yield
        finally:
            await self.concurrent_decrement(redis, key)

    @staticmethod
    async def check_bucket(
        redis: Redis,
        redis_rate_limit_command: str,
        key: str,
        times: int | None,
        milliseconds: int,
    ):
        if times is None:
            return
        expire = await redis.evalsha(
            redis_rate_limit_command, 1, key, str(times), str(milliseconds)
        )  # type: ignore

        if int(expire) > 0:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too Many Requests",
                headers={"Retry-After": expire},
            )

    async def check_concurrent_and_increment(
        self, redis: Redis, request: Request
    ) -> str | None:
        if not self.max_concurrent and not self.concurrent_extractor:
            return None

        if self.concurrent_extractor:
            maybe_future = self.concurrent_extractor(request)
            if iscoroutine(maybe_future):
                max_concurrent, key = await maybe_future
            else:
                max_concurrent, key = maybe_future  # type: ignore
        else:
            max_concurrent, key = self.max_concurrent, request.url.path

        current = await redis.incr(f"concurrent:{key}")
        if current > max_concurrent:
            await redis.decr(f"concurrent:{key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too Many Requests",
            )
        return key

    @staticmethod
    async def concurrent_decrement(redis: Redis, key: str | None):
        if key is None:
            return
        await redis.decr(f"concurrent:{key}")
