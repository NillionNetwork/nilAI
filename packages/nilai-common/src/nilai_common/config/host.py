"""Host and infrastructure configuration settings."""

import os
from pydantic import BaseModel, Field


def to_bool(value: str) -> bool:
    """Convert a string to a boolean."""
    return value.lower() in ("true", "1", "t", "y", "yes")


class HostSettings(BaseModel):
    """Infrastructure and service host configuration."""

    host: str = Field(default="localhost", description="Host of the service")
    port: int = Field(default=8000, description="Port of the service")
    discovery_url: str = Field(
        default="redis://redis:6379",
        description="Redis URL of the discovery service (preferred)",
    )
    gunicorn_workers: int = Field(default=10, description="Number of gunicorn workers")


# Global host settings instance
SETTINGS: HostSettings = HostSettings(
    host=str(os.getenv("SVC_HOST", "localhost")),
    port=int(os.getenv("SVC_PORT", 8000)),
    discovery_url=str(os.getenv("DISCOVERY_URL", "redis://redis:6379")),
    gunicorn_workers=int(os.getenv("NILAI_GUNICORN_WORKERS", 10)),
)
