from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AuthConfig(BaseModel):
    strategy: Literal["api_key", "jwt", "nuc"] = Field(
        default="api_key", description="Authentication strategy"
    )
    nilauth_trusted_root_issuers: List[str] = Field(
        default_factory=list, description="Trusted root issuers for nilauth"
    )


class DocsConfig(BaseModel):
    token: Optional[str] = Field(default=None, description="Documentation access token")
