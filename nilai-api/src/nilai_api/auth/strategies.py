from nilai_api.db.users import UserManager, UserModel, UserData
from nilai_api.auth.jwt import validate_jwt
from nilai_api.auth.nuc import validate_nuc, get_token_rate_limit
from nilai_api.auth.common import (
    TokenRateLimits,
    AuthenticationInfo,
    AuthenticationError,
)

from enum import Enum
# All strategies must return a UserModel
# The strategies can raise any exception, which will be caught and converted to an AuthenticationError
# The exception detail will be passed to the client


async def api_key_strategy(api_key: str) -> AuthenticationInfo:
    user_model: UserModel | None = await UserManager.check_api_key(api_key)
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model), token_rate_limit=None
        )
    raise AuthenticationError("Missing or invalid API key")


async def jwt_strategy(jwt_creds: str) -> AuthenticationInfo:
    result = validate_jwt(jwt_creds)
    user_model: UserModel | None = await UserManager.check_api_key(result.user_address)
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model), token_rate_limit=None
        )
    else:
        user_model = UserModel(
            userid=result.user_address,
            name=result.pub_key,
            apikey=result.user_address,
        )
        await UserManager.insert_user_model(user_model)
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model), token_rate_limit=None
        )


async def nuc_strategy(nuc_token) -> AuthenticationInfo:
    """
    Validate a NUC token and return the user model
    """
    subscription_holder, user = validate_nuc(nuc_token)
    token_rate_limits: TokenRateLimits | None = get_token_rate_limit(nuc_token)
    user_model: UserModel | None = await UserManager.check_user(user)
    if user_model:
        return AuthenticationInfo(
            user=UserData.from_sqlalchemy(user_model),
            token_rate_limit=token_rate_limits,
        )

    user_model = UserModel(
        userid=user,
        name=user,
        apikey=subscription_holder,
    )
    await UserManager.insert_user_model(user_model)
    return AuthenticationInfo(
        user=UserData.from_sqlalchemy(user_model), token_rate_limit=token_rate_limits
    )


class AuthenticationStrategy(Enum):
    API_KEY = (api_key_strategy, "API Key")
    JWT = (jwt_strategy, "JWT")
    NUC = (nuc_strategy, "NUC")

    async def __call__(self, *args, **kwargs) -> AuthenticationInfo:
        return await self.value[0](*args, **kwargs)


__all__ = ["AuthenticationStrategy"]
