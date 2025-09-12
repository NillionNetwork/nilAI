from typing import Callable, Awaitable, Optional
from datetime import datetime, timezone

from nilai_api.db.users import UserManager, UserModel, UserData
from nilai_api.auth.jwt import validate_jwt
from nilai_api.auth.nuc import (
    validate_nuc,
    get_token_rate_limit,
    get_token_prompt_document,
)
from nilai_api.config import CONFIG
from nilai_api.auth.common import (
    PromptDocument,
    TokenRateLimits,
    AuthenticationInfo,
    AuthenticationError,
)

from enum import Enum
# All strategies must return a UserModel
# The strategies can raise any exception, which will be caught and converted to an AuthenticationError
# The exception detail will be passed to the client


# Decorator with parameter to allow a specific token to bypass the authentication
def allow_token(
    allowed_token: str | None = None,
) -> Callable[
    [Callable[[str], Awaitable[AuthenticationInfo]]],
    Callable[[str], Awaitable[AuthenticationInfo]],
]:
    """
    Decorator to allow a specific token to bypass the authentication
    If the token is not provided, the decorator will not be applied
    If the token is provided, the decorator will return an AuthenticationInfo object
    If the token is provided, the decorator will return the original function
    """

    def decorator(function) -> Callable[[str], Awaitable[AuthenticationInfo]]:
        if allowed_token is None:
            return function

        async def wrapper(token) -> AuthenticationInfo:
            if allowed_token is None:
                return await function(token)

            if token == allowed_token:
                user_model: UserModel | None = await UserManager.check_user(
                    allowed_token
                )
                if user_model is None:
                    user_model = UserModel(
                        userid=allowed_token,
                        name=allowed_token,
                        apikey=allowed_token,
                        signup_date=datetime.now(timezone.utc),
                    )
                    await UserManager.insert_user_model(user_model)

                return AuthenticationInfo(
                    user=UserData.from_sqlalchemy(user_model),
                    token_rate_limit=None,
                    prompt_document=None,
                )
            return await function(token)

        return wrapper

    return decorator


@allow_token(CONFIG.docs.token)
async def api_key_strategy(api_key: str) -> AuthenticationInfo:
    user_model: Optional[UserModel] = await UserManager.check_api_key(api_key)
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model),
            token_rate_limit=None,
            prompt_document=None,
        )
    raise AuthenticationError("Missing or invalid API key")


@allow_token(CONFIG.docs.token)
async def jwt_strategy(jwt_creds: str) -> AuthenticationInfo:
    result = validate_jwt(jwt_creds)
    user_model: Optional[UserModel] = await UserManager.check_api_key(
        result.user_address
    )
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model),
            token_rate_limit=None,
            prompt_document=None,
        )
    else:
        user_model = UserModel(
            userid=result.user_address,
            name=result.pub_key,
            apikey=result.user_address,
        )
        await UserManager.insert_user_model(user_model)
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model),
            token_rate_limit=None,
            prompt_document=None,
        )


@allow_token(CONFIG.docs.token)
async def nuc_strategy(nuc_token) -> AuthenticationInfo:
    """
    Validate a NUC token and return the user model
    """
    subscription_holder, user = validate_nuc(nuc_token)
    token_rate_limits: Optional[TokenRateLimits] = get_token_rate_limit(nuc_token)
    prompt_document: Optional[PromptDocument] = get_token_prompt_document(nuc_token)

    user_model: Optional[UserModel] = await UserManager.check_user(user)
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model),
            token_rate_limit=token_rate_limits,
            prompt_document=prompt_document,
        )

    user_model = UserModel(
        userid=user,
        name=user,
        apikey=subscription_holder,
    )
    await UserManager.insert_user_model(user_model)
    return AuthenticationInfo(
        user=UserData.from_sqlalchemy(user_model),
        token_rate_limit=token_rate_limits,
        prompt_document=prompt_document,
    )


class AuthenticationStrategy(Enum):
    API_KEY = (api_key_strategy, "API Key")
    JWT = (jwt_strategy, "JWT")
    NUC = (nuc_strategy, "NUC")

    async def __call__(self, *args, **kwargs) -> AuthenticationInfo:
        return await self.value[0](*args, **kwargs)


__all__ = ["AuthenticationStrategy"]
