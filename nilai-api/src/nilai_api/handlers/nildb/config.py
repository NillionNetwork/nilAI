import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

from secretvaults.common.types import Uuid

load_dotenv()


class NilDBConfig(BaseModel):
    NILCHAIN_URL: str = Field(..., description="The URL of the Nilchain")
    NILAUTH_URL: str = Field(..., description="The URL of the Nilauth")
    NODES: list[str] = Field(..., description="The URLs of the Nildb nodes")
    BUILDER_PRIVATE_KEY: str = Field(..., description="The private key of the builder")
    COLLECTION: Uuid = Field(..., description="The ID of the collection")


def get_required_env_var(name: str) -> str:
    """Get a required environment variable, raising an error if not set."""
    value: Optional[str] = os.getenv(name, None)
    if value is None:
        raise ValueError(f"Required environment variable {name} is not set")
    return value


# Validate environment variables at import time
CONFIG = NilDBConfig(
    NILCHAIN_URL=get_required_env_var("NILDB_NILCHAIN_URL"),
    NILAUTH_URL=get_required_env_var("NILDB_NILAUTH_URL"),
    NODES=get_required_env_var("NILDB_NODES").split(","),
    BUILDER_PRIVATE_KEY=get_required_env_var("NILDB_BUILDER_PRIVATE_KEY"),
    COLLECTION=Uuid(get_required_env_var("NILDB_COLLECTION")),
)


__all__ = ["CONFIG"]
