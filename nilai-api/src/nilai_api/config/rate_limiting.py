from typing import Dict, Optional
from pydantic import BaseModel, Field


class RateLimitingConfig(BaseModel):
    user_rate_limit_minute: Optional[int] = Field(
        description="User requests per minute limit"
    )
    user_rate_limit_hour: Optional[int] = Field(
        description="User requests per hour limit"
    )
    user_rate_limit_day: Optional[int] = Field(
        description="User requests per day limit"
    )
    web_search_rate_limit_minute: Optional[int] = Field(
        description="Web search requests per minute limit"
    )
    web_search_rate_limit_hour: Optional[int] = Field(
        description="Web search requests per hour limit"
    )
    web_search_rate_limit_day: Optional[int] = Field(
        description="Web search requests per day limit"
    )
    model_concurrent_rate_limit: Dict[str, int] = Field(
        default_factory=lambda: {"default": 50},
        description="Model concurrent request limits",
    )
    user_rate_limit: Optional[int] = Field(
        default=None, description="User requests per day limit"
    )
    web_search_rate_limit: Optional[int] = Field(
        default=None, description="Web search requests per day limit"
    )
