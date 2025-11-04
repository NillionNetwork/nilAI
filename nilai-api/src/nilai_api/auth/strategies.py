from typing import Callable, Awaitable, Optional

from fastapi import HTTPException
from nilai_api.db.users import UserManager, UserModel, UserData
from nilai_api.auth.nuc import (
    validate_nuc,
    get_token_rate_limit,
    get_token_prompt_document,
)
from nilai_api.config import CONFIG
from nilai_api.auth.common import (
    PromptDocument,
    TokenRateLimits,
    AuthenticationError,
    AuthenticationInfo,
)

from nilauth_credit_middleware import (
    CreditClientSingleton,
)
from nilauth_credit_middleware.api_model import ValidateCredentialResponse


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
                user_model = UserModel(
                    user_id=allowed_token,
                    rate_limits=None,
                )
                return AuthenticationInfo(
                    user=UserData.from_sqlalchemy(user_model),
                    token_rate_limit=None,
                    prompt_document=None,
                )
            return await function(token)

        return wrapper

    return decorator


async def validate_credential(credential: str, is_public: bool) -> UserModel:
    """
    Validate a credential with nilauth credit middleware and return the user model
    """
    credit_client = CreditClientSingleton.get_client()
    try:
        validate_response: ValidateCredentialResponse = (
            await credit_client.validate_credential(credential, is_public=is_public)
        )
    except HTTPException as e:
        if e.status_code == 404:
            raise AuthenticationError(f"Credential not found: {e.detail}")
        elif e.status_code == 401:
            raise AuthenticationError(f"Credential is inactive: {e.detail}")
        else:
            raise AuthenticationError(f"Failed to validate credential: {e.detail}")

    user_model = await UserManager.check_user(validate_response.user_id)
    if user_model is None:
        user_model = UserModel(
            user_id=validate_response.user_id,
            rate_limits=None,
        )
    return user_model


@allow_token(CONFIG.docs.token)
async def api_key_strategy(api_key: str) -> AuthenticationInfo:
    user_model = await validate_credential(api_key, is_public=False)

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

    user_model = await validate_credential(subscription_holder, is_public=True)
    return AuthenticationInfo(
        user=UserData.from_sqlalchemy(user_model),
        token_rate_limit=token_rate_limits,
        prompt_document=prompt_document,
    )


class AuthenticationStrategy(Enum):
    API_KEY = (api_key_strategy, "API Key")
    NUC = (nuc_strategy, "NUC")

    async def __call__(self, *args, **kwargs) -> AuthenticationInfo:
        return await self.value[0](*args, **kwargs)


__all__ = ["AuthenticationStrategy"]
