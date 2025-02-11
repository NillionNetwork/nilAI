from nilai_api.db import UserManager, UserModel
from nilai_api.auth.jwt import validate_jwt


async def api_key_strategy(api_key):
    return await UserManager.check_api_key(api_key)


async def jwt_strategy(jwt_creds):
    result = validate_jwt(jwt_creds)
    print(result)
    if not result["is_valid"]:
        return None
    user_address = result["payload"].get("user_address")
    user_public_key = result["payload"].get("pub_key")
    user = await UserManager.check_api_key(user_address)
    if user:
        return user
    user = UserModel(
        userid=user_address,
        name=user_public_key,
        email=user_public_key,
        apikey=user_address,
    )
    await UserManager.insert_user_model(user)
    return user


STRATEGIES = {
    "api_key": api_key_strategy,
    "jwt": jwt_strategy,
}

__all__ = ["STRATEGIES"]
