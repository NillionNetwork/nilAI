from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from nilai.db import UserManager

UserManager.initialize_db()

api_key_header = APIKeyHeader(name="X-API-Key")


def get_user(api_key_header: str = Security(api_key_header)):
    user = UserManager.check_api_key(api_key_header)
    if user:
        return user
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
    )
