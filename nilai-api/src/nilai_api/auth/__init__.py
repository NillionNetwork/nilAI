from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from logging import getLogger

from nilai_api import config
from nilai_api.auth.jwt import validate_jwt
from nilai_api.db.users import UserManager, UserModel
from nilai_api.auth.strategies import STRATEGIES

from nuc.validate import ValidationException

logger = getLogger(__name__)
bearer_scheme = HTTPBearer()


class AuthenticationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> UserModel:
    try:
        if config.AUTH_STRATEGY not in STRATEGIES:
            logger.error(f"Invalid auth strategy: {config.AUTH_STRATEGY}")
            raise AuthenticationError("Server misconfiguration: invalid auth strategy")

        user = await STRATEGIES[config.AUTH_STRATEGY](credentials.credentials)
        if not user:
            raise AuthenticationError("Missing or invalid API key")
        await UserManager.update_last_activity(userid=user.userid)
        return user
    except AuthenticationError as e:
        raise e
    except ValueError as e:
        raise AuthenticationError(detail="Authentication failed: " + str(e))
    except ValidationException as e:
        raise AuthenticationError(detail="NUC validation failed: " + str(e))
    except Exception as e:
        raise AuthenticationError(detail="Unexpected authentication error: " + str(e))


__all__ = ["get_user", "validate_jwt"]
