from typing import Optional
from pydantic import BaseModel, Field


class WebSearchSettings(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Brave Search API key")
    api_path: str = Field(
        default="https://api.search.brave.com/res/v1/web/search",
        description="Search API endpoint",
    )
    count: int = Field(default=3, description="Number of search results")
    lang: str = Field(default="en", description="Search language")
    country: str = Field(default="us", description="Search country")
    timeout: float = Field(default=20.0, description="Request timeout in seconds")
    max_concurrent_requests: int = Field(
        default=20, description="Maximum concurrent requests"
    )
    rps: Optional[int] = Field(default=20, description="Requests per second limit")
