from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    user: str = Field(description="Database user")
    password: str = Field(description="Database password")
    host: str = Field(description="Database host")
    port: int = Field(description="Database port")
    db: str = Field(description="Database name")


class DiscoveryConfig(BaseModel):
    url: str = Field(description="Redis URL for discovery")


class RedisConfig(BaseModel):
    url: str = Field(description="Redis URL for rate limiting")
