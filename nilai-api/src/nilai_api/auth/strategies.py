from nilai_api.db.users import UserManager, UserModel
from nilai_api.auth.jwt import validate_jwt


async def api_key_strategy(api_key):
    return await UserManager.check_api_key(api_key)


async def jwt_strategy(jwt_creds):
    result = validate_jwt(jwt_creds)
    user = await UserManager.check_api_key(result.user_address)
    if user:
        return user
    user = UserModel(
        userid=result.user_address,
        name=result.pub_key,
        email=result.pub_key,
        apikey=result.user_address,
    )
    await UserManager.insert_user_model(user)
    return user


STRATEGIES = {
    "api_key": api_key_strategy,
    "jwt": jwt_strategy,
}

__all__ = ["STRATEGIES"]
