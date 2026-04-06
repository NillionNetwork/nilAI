from typing import List
from pydantic import BaseModel, Field, field_validator
from secretvaults.common.types import Uuid


class NilDBConfig(BaseModel):
    nilchain_url: str = Field(default="", description="The URL of the Nilchain")
    nilauth_url: str = Field(default="", description="The URL of the Nilauth")
    nodes: List[str] = Field(
        default_factory=list, description="The URLs of the Nildb nodes"
    )
    builder_private_key: str = Field(
        default="", description="The private key of the builder"
    )
    collection: Uuid = Field(
        default=Uuid("00000000-0000-0000-0000-000000000000"),
        description="The ID of the collection",
    )

    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, v):
        if isinstance(v, str):
            return v.split(",")
        return v

    @field_validator("collection", mode="before")
    @classmethod
    def parse_collection(cls, v):
        if isinstance(v, str):
            return Uuid(v)
        return v
