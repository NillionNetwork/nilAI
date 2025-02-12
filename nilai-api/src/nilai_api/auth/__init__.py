from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from nilai_api import config
from nilai_api.auth.jwt import validate_jwt
from nilai_api.db.users import UserManager, UserModel
from nilai_api.auth.strategies import STRATEGIES

bearer_scheme = HTTPBearer()


async def get_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
) -> UserModel:
    user = await STRATEGIES[config.AUTH_STRATEGY](credentials.credentials)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )

    await UserManager.update_last_activity(userid=user.userid)
    return user


__all__ = ["get_user", "validate_jwt"]
