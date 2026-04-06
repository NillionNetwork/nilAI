from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    user: str = Field(default="", description="Database user")
    password: str = Field(default="", description="Database password")
    host: str = Field(default="", description="Database host")
    port: int = Field(default=5432, description="Database port")
    db: str = Field(default="", description="Database name")


class DiscoveryConfig(BaseModel):
    url: str = Field(
        default="redis://localhost:6379",
        description="Redis URL for discovery (preferred default)",
    )


class RedisConfig(BaseModel):
    url: str = Field(
        default="redis://localhost:6379", description="Redis URL for rate limiting"
    )
