"""Host and infrastructure configuration settings."""

import os
from pydantic import BaseModel, Field


def to_bool(value: str) -> bool:
    """Convert a string to a boolean."""
    return value.lower() in ("true", "1", "t", "y", "yes")


class HostSettings(BaseModel):
    """Infrastructure and service host configuration."""

    url: str = Field(default="http://localhost:8000", description="URL of the service")
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis URL for discovery service"
    )
    gunicorn_workers: int = Field(default=10, description="Number of gunicorn workers")


# Global host settings instance
SETTINGS: HostSettings = HostSettings(
    url=os.getenv("SVC_URL", "http://localhost:8000"),
    redis_url=os.getenv("DISCOVERY_URL", "redis://localhost:6379"),
    gunicorn_workers=int(os.getenv("NILAI_GUNICORN_WORKERS", 10)),
)
