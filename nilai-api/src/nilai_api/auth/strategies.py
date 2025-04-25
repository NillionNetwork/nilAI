from nilai_api.db.users import UserManager, UserModel
from nilai_api.auth.jwt import validate_jwt
from nilai_api.auth.nuc import validate_nuc

# All strategies must return a UserModel
# The strategies can raise any exception, which will be caught and converted to an AuthenticationError
# The exception detail will be passed to the client


async def api_key_strategy(api_key) -> UserModel:
    return await UserManager.check_api_key(api_key)


async def jwt_strategy(jwt_creds) -> UserModel:
    result = validate_jwt(jwt_creds)
    user = await UserManager.check_api_key(result.user_address)
    if user:
        return user
    user = UserModel(
        userid=result.user_address,
        name=result.pub_key,
        apikey=result.user_address,
    )
    await UserManager.insert_user_model(user)
    return user


async def nuc_strategy(nuc_token) -> UserModel:
    """
    Validate a NUC token and return the user model
    """
    subscription_holder, user = validate_nuc(nuc_token)

    user_model = await UserManager.check_user(user)
    if user_model:
        return user_model

    user_model = UserModel(
        userid=user,
        name=user,
        apikey=subscription_holder,
    )
    await UserManager.insert_user_model(user_model)
    return user_model


STRATEGIES = {
    "api_key": api_key_strategy,
    "jwt": jwt_strategy,
    "nuc": nuc_strategy,
}

__all__ = ["STRATEGIES"]
