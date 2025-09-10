from typing import Literal
from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    environment: Literal["testnet", "mainnet", "ci"] = Field(
        default="mainnet", description="The environment to use"
    )
