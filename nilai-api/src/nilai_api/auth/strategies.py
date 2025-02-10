from nilai_api.db import UserManager
from nilai_api.auth.jwt import validate_jwt


async def api_key_strategy(api_key):
    return await UserManager.check_api_key(api_key)


async def jwt_strategy(jwt_creds):
    result = validate_jwt(jwt_creds)
    if not result.is_valid:
        return
    return result["payload"]


STRATEGIES = {
    "api_key": api_key_strategy,
    "jwt": jwt_strategy,
}

__all__ = ["STRATEGIES"]
