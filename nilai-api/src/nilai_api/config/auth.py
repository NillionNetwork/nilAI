from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AuthConfig(BaseModel):
    auth_strategy: Literal["api_key", "jwt", "nuc"] = Field(
        description="Authentication strategy"
    )
    nilauth_trusted_root_issuers: List[str] = Field(
        description="Trusted root issuers for nilauth"
    )
    auth_token: Optional[str] = Field(
        default=None, description="Auth token for testing"
    )


class DocsConfig(BaseModel):
    token: Optional[str] = Field(default=None, description="Documentation access token")
