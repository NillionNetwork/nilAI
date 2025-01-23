from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from nilai_api.db import UserManager

UserManager.initialize_db()

bearer_scheme = HTTPBearer()


async def get_user(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    token = credentials.credentials
    user = await UserManager.check_api_key(token)
    if user:
        await UserManager.update_last_activity(userid=user["userid"])
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
    )
