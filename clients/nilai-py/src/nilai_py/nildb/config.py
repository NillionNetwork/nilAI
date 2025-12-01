import os
import enum

from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field, field_validator
from secretvaults.common.types import Uuid


class NilDBConfig(BaseModel):
    nilchain_url: str = Field(..., description="The URL of the Nilchain")
    nilauth_url: str = Field(..., description="The URL of the Nilauth")
    nodes: List[str] = Field(..., description="The URLs of the Nildb nodes")
    collection: Uuid = Field(..., description="The ID of the collection")

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


class NilDBCollection(enum.Enum):
    SANDBOX = "e035f44e-9fb4-4560-b707-b9325c11207c"
    PRODUCTION = "e035f44e-9fb4-4560-b707-b9325c11207c"


load_dotenv()

# Initialize configuration from environment variables or defaults
DefaultNilDBConfig = NilDBConfig(
    nilchain_url=os.getenv(
        "NILDB_NILCHAIN_URL", "http://rpc.testnet.nilchain-rpc-proxy.nilogy.xyz"
    ),
    nilauth_url=os.getenv(
        "NILDB_NILAUTH_URL", "https://nilauth.sandbox.app-cluster.sandbox.nilogy.xyz"
    ),
    nodes=os.getenv(
        "NILDB_NODES",
        "https://nildb-stg-n1.nillion.network,https://nildb-stg-n2.nillion.network,https://nildb-stg-n3.nillion.network",
    ).split(","),
    collection=Uuid(os.getenv("NILDB_COLLECTION", NilDBCollection.SANDBOX.value)),
)

print(
    f"Using NilDB Configuration:\n Nilchain URL: {DefaultNilDBConfig.nilchain_url}\n  Nilauth URL: {DefaultNilDBConfig.nilauth_url}\n  Nodes: {DefaultNilDBConfig.nodes}\n  Collection ID: {DefaultNilDBConfig.collection}"
)
