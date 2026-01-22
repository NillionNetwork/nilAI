from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from logging import getLogger

from nilai_api.config import CONFIG
from nilai_api.auth.strategies import AuthenticationStrategy

from nuc.validate import ValidationException
from nilai_api.auth.nuc_helpers.usage import UsageLimitError

from nilai_api.auth.common import (
    AuthenticationInfo,
    AuthenticationError,
    TokenRateLimit,
    TokenRateLimits,
)

logger = getLogger(__name__)
bearer_scheme = HTTPBearer()


async def get_auth_info(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> AuthenticationInfo:
    try:
        strategy_name: str = CONFIG.auth.auth_strategy.upper()

        try:
            strategy = AuthenticationStrategy[strategy_name]
        except KeyError:  # If the strategy is not found, we raise an error
            logger.error(f"Invalid auth strategy: {strategy_name}")
            raise AuthenticationError(
                f"Server misconfiguration: invalid auth strategy: {strategy_name}"
            )

        auth_info = await strategy(credentials.credentials)
        return auth_info
    except AuthenticationError as e:
        raise e
    except ValueError as e:
        raise AuthenticationError(detail="Authentication failed: " + str(e))
    except ValidationException as e:
        raise AuthenticationError(detail="NUC validation failed: " + str(e))
    except UsageLimitError as e:
        raise AuthenticationError(detail="Usage limit error: " + str(e))
    except Exception as e:
        raise AuthenticationError(detail="Unexpected authentication error: " + str(e))


__all__ = ["get_auth_info", "AuthenticationInfo", "TokenRateLimits", "TokenRateLimit"]
