from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="nilai_users", description="Database name")


class EtcdConfig(BaseModel):
    host: str = Field(default="localhost", description="ETCD host")
    port: int = Field(default=2379, description="ETCD port")


class RedisConfig(BaseModel):
    url: str = Field(default="redis://localhost:6379", description="Redis URL")
