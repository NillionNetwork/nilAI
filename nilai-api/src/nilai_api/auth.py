from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from authlib.jose import JsonWebToken

from nilai_api.db import UserManager, UserModel
from nilai_api import config

bearer_scheme = HTTPBearer()


async def api_key_strategy(api_key):
    return await UserManager.check_api_key(api_key)


async def jwt_strategy(jwt_creds):
    jwt = JsonWebToken(["ES256"])
    raise NotImplementedError("JWT Strategy not implemented yet")
    public_key = None
    jwt.decode(jwt_creds, public_key)


STRATEGIES = {
    "api_key": api_key_strategy,
    "jwt": jwt_strategy,
}


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
