from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException, status
from nilai_api.db.users import UserData
from nuc_helpers.usage import TokenRateLimits, TokenRateLimit


class AuthenticationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthenticationInfo(BaseModel):
    user: UserData
    token_rate_limit: Optional[TokenRateLimits]


__all__ = [
    "AuthenticationError",
    "AuthenticationInfo",
    "TokenRateLimits",
    "TokenRateLimit",
]
