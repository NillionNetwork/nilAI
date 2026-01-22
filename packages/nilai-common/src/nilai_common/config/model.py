"""Model-specific configuration settings."""

import os
from pydantic import BaseModel, Field

from .host import to_bool


class ModelSettings(BaseModel):
    """Model retry and timeout configuration."""

    num_retries: int = Field(default=30, ge=-1, description="Number of retries")
    timeout: int = Field(default=10, ge=1, description="Timeout in seconds")


class ModelCapabilities(BaseModel):
    """Model capability flags."""

    tool_support: bool = Field(default=False, description="Tool support flag")
    multimodal_support: bool = Field(
        default=False, description="Multimodal support flag"
    )


# Global model settings instance
MODEL_SETTINGS: ModelSettings = ModelSettings(
    num_retries=int(os.getenv("MODEL_NUM_RETRIES", 30)),
    timeout=int(os.getenv("MODEL_RETRY_TIMEOUT", 10)),
)

# Global model capabilities instance
MODEL_CAPABILITIES: ModelCapabilities = ModelCapabilities(
    tool_support=to_bool(os.getenv("TOOL_SUPPORT", "False")),
    multimodal_support=to_bool(os.getenv("MULTIMODAL_SUPPORT", "False")),
)
