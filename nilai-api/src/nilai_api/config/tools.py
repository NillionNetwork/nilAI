from typing import List
from pydantic import BaseModel, Field


class ToolsConfig(BaseModel):
    implemented_tools: List[str] = Field(
        default_factory=lambda: ["execute_python"],
        description="List of tool names that are implemented and can be executed",
    )
