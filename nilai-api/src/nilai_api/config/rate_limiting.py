from typing import Dict, Optional
from pydantic import BaseModel, Field


class RateLimitingConfig(BaseModel):
    user_rate_limit_minute: Optional[int] = Field(
        default=100, description="User requests per minute limit"
    )
    user_rate_limit_hour: Optional[int] = Field(
        default=1000, description="User requests per hour limit"
    )
    user_rate_limit_day: Optional[int] = Field(
        default=10000, description="User requests per day limit"
    )
    web_search_rate_limit_minute: Optional[int] = Field(
        default=1, description="Web search requests per minute limit"
    )
    web_search_rate_limit_hour: Optional[int] = Field(
        default=3, description="Web search requests per hour limit"
    )
    web_search_rate_limit_day: Optional[int] = Field(
        default=72, description="Web search requests per day limit"
    )
    model_concurrent_rate_limit: Dict[str, int] = Field(
        default_factory=dict, description="Model concurrent request limits"
    )
