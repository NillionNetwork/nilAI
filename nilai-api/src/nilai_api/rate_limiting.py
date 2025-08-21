import asyncio
from asyncio import iscoroutine
from typing import Callable, Tuple, Awaitable, Annotated

from pydantic import BaseModel
from nilai_api.config import (
    WEB_SEARCH_API_RPS,
    WEB_SEARCH_API_MAX_CONCURRENT_REQUESTS,
    WEB_SEARCH_COUNT,
)

from fastapi.params import Depends
from fastapi import status, HTTPException, Request
from redis.asyncio import from_url, Redis

from nilai_api.auth import get_auth_info, AuthenticationInfo, TokenRateLimits

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


async def _extract_coroutine_result(maybe_future, request: Request):
    if iscoroutine(maybe_future):
        return await maybe_future
    else:
        return maybe_future


class UserRateLimits(BaseModel):
    subscription_holder: str
    day_limit: int | None
    hour_limit: int | None
    minute_limit: int | None
    token_rate_limit: TokenRateLimits | None
    web_search_day_limit: int | None
    web_search_hour_limit: int | None
    web_search_minute_limit: int | None


def get_user_limits(
    auth_info: Annotated[AuthenticationInfo, Depends(get_auth_info)],
) -> UserRateLimits:
    # TODO: When the only allowed strategy is NUC, we can change the apikey name to subscription_holder
    # In apikey mode, the apikey is unique as the userid.
    # In nuc mode, the apikey is associated with a subscription holder and the userid is the user
    # For NUCs we want the rate limit to be per subscription holder, not per user
    # In JWT mode, the apikey is the userid too
    # So we use the apikey as the id
    return UserRateLimits(
        subscription_holder=auth_info.user.apikey,
        day_limit=auth_info.user.ratelimit_day,
        hour_limit=auth_info.user.ratelimit_hour,
        minute_limit=auth_info.user.ratelimit_minute,
        token_rate_limit=auth_info.token_rate_limit,
        web_search_day_limit=auth_info.user.web_search_ratelimit_day,
        web_search_hour_limit=auth_info.user.web_search_ratelimit_hour,
        web_search_minute_limit=auth_info.user.web_search_ratelimit_minute,
    )


class RateLimit:
    def __init__(
        self,
        concurrent: int | None = None,
        concurrent_extractor: Callable[
            [Request], Tuple[int, str] | Awaitable[Tuple[int, str]]
        ]
        | None = None,
        web_search_extractor: Callable[[Request], bool | Awaitable[bool]] | None = None,
    ):
        """
        concurrent: Maximum number of concurrent requests allowed for a single path
        concurrent_extractor: A callable that extracts the concurrent limit and key from the request
        web_search_extractor: A callable that extracts the web_search flag from the request

        concurrent and concurrent_extractor are mutually exclusive
        """
        self.max_concurrent = concurrent
        self.concurrent_extractor = concurrent_extractor
        self.web_search_extractor = web_search_extractor

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
            f"minute:{user_limits.subscription_holder}",
            user_limits.minute_limit,
            MINUTE_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"hour:{user_limits.subscription_holder}",
            user_limits.hour_limit,
            HOUR_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"day:{user_limits.subscription_holder}",
            user_limits.day_limit,
            DAY_MS,
        )

        if (
            user_limits.token_rate_limit
        ):  # If the token rate limit is not None, we need to check it
            # We create a record in redis for the signature
            # The key is the signature
            # The value is the usage limit
            # The expiration is the time remaining in validity of the token
            # We use the time remaining to check if the token rate limit is exceeded

            for limit in user_limits.token_rate_limit.limits:
                await self.check_bucket(
                    redis,
                    redis_rate_limit_command,
                    f"token:{limit.signature}",
                    limit.usage_limit,
                    limit.ms_remaining,
                )

        if self.web_search_extractor:
            web_search_enabled = await _extract_coroutine_result(
                self.web_search_extractor(request), request
            )

            if web_search_enabled:
                allowed_rps = min(
                    WEB_SEARCH_API_RPS,
                    max(
                        1,
                        WEB_SEARCH_API_MAX_CONCURRENT_REQUESTS
                        // WEB_SEARCH_COUNT,
                    ),
                )
                await self.wait_for_bucket(
                    redis,
                    redis_rate_limit_command,
                    "global:web_search:rps",
                    allowed_rps,
                    1000,
                )
                web_search_limits = [
                    (user_limits.web_search_minute_limit, MINUTE_MS, "minute"),
                    (user_limits.web_search_hour_limit, HOUR_MS, "hour"),
                    (user_limits.web_search_day_limit, DAY_MS, "day"),
                ]

                for limit, milliseconds, time_unit in web_search_limits:
                    await self.check_bucket(
                        redis,
                        redis_rate_limit_command,
                        f"web_search_{time_unit}:{user_limits.subscription_holder}",
                        limit,
                        milliseconds,
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
                headers={"Retry-After": str(expire)},
            )

    @staticmethod
    async def wait_for_bucket(
        redis: Redis,
        redis_rate_limit_command: str,
        key: str,
        times: int | None,
        milliseconds: int,
    ):
        if times is None:
            return
        while True:
            expire = await redis.evalsha(
                redis_rate_limit_command, 1, key, str(times), str(milliseconds)
            )  # type: ignore
            if int(expire) == 0:
                return
            await asyncio.sleep((int(expire) + 50) / 1000)

    async def check_concurrent_and_increment(
        self, redis: Redis, request: Request
    ) -> str | None:
        if not self.max_concurrent and not self.concurrent_extractor:
            return None

        if self.concurrent_extractor:
            max_concurrent, key = await _extract_coroutine_result(
                self.concurrent_extractor(request), request
            )
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
