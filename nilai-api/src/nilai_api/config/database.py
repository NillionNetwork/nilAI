from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    user: str = Field(description="Database user")
    password: str = Field(description="Database password")
    host: str = Field(description="Database host")
    port: int = Field(description="Database port")
    db: str = Field(description="Database name")


class DiscoveryConfig(BaseModel):
    host: str = Field(default="localhost", description="Redis host for discovery")
    port: int = Field(default=6379, description="Redis port for discovery")


class RedisConfig(BaseModel):
    url: str = Field(description="Redis URL for rate limiting")
