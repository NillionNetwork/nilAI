from asyncio import iscoroutine
from typing import Callable, Tuple, Awaitable, Annotated

from nilai_api.db.users import RateLimits
from pydantic import BaseModel

from fastapi.params import Depends
from fastapi import status, HTTPException, Request
from redis.asyncio import from_url, Redis

from nilai_api.auth import get_auth_info, AuthenticationInfo, TokenRateLimits

LUA_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire_time = tonumber(ARGV[2])

local current = tonumber(redis.call('get', key) or "0")
if current > 0 then
    if current + 1 > limit then
        return redis.call("PTTL", key)
    else
        redis.call("INCR", key)
        return 0
    end
else
    if expire_time > 0 then
        redis.call("SET", key, 1, "px", expire_time)
    else
        redis.call("SET", key, 1)
    end
    return 0
end
"""
MINUTE_MS = 60 * 1000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS


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
    token_rate_limit: TokenRateLimits | None
    rate_limits: RateLimits


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
        token_rate_limit=auth_info.token_rate_limit,
        rate_limits=auth_info.user.rate_limits,
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
            user_limits.rate_limits.user_rate_limit_minute,
            MINUTE_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"hour:{user_limits.subscription_holder}",
            user_limits.rate_limits.user_rate_limit_hour,
            HOUR_MS,
        )
        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"day:{user_limits.subscription_holder}",
            user_limits.rate_limits.user_rate_limit_day,
            DAY_MS,
        )

        await self.check_bucket(
            redis,
            redis_rate_limit_command,
            f"user:{user_limits.subscription_holder}",
            user_limits.rate_limits.user_rate_limit,
            0,  # No expiration for for-good rate limit
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
                await self.check_bucket(
                    redis,
                    redis_rate_limit_command,
                    f"web_search_minute:{user_limits.subscription_holder}",
                    user_limits.rate_limits.web_search_rate_limit_minute,
                    MINUTE_MS,
                )
                await self.check_bucket(
                    redis,
                    redis_rate_limit_command,
                    f"web_search_hour:{user_limits.subscription_holder}",
                    user_limits.rate_limits.web_search_rate_limit_hour,
                    HOUR_MS,
                )
                await self.check_bucket(
                    redis,
                    redis_rate_limit_command,
                    f"web_search_day:{user_limits.subscription_holder}",
                    user_limits.rate_limits.web_search_rate_limit_day,
                    DAY_MS,
                )
                await self.check_bucket(
                    redis,
                    redis_rate_limit_command,
                    f"web_search:{user_limits.subscription_holder}",
                    user_limits.rate_limits.web_search_rate_limit,
                    0,
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
