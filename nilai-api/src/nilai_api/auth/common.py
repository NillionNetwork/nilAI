from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from fastapi import HTTPException, status
from nilai_api.db.users import UserData


class AuthenticationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class TokenRateLimit(BaseModel):
    signature: str
    expires_at: datetime
    usage_limit: Optional[int]

    @property
    def ms_remaining(self) -> int:
        return int(
            (self.expires_at - datetime.now(timezone.utc)).total_seconds() * 1000
        )


class AuthenticationInfo(BaseModel):
    user: UserData
    token_rate_limit: Optional[TokenRateLimit]
