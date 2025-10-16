from typing import Set
from pydantic import BaseModel, Field


class ToolsConfig(BaseModel):
    implemented_tools: Set[str] = Field(
        default={"execute_python"},
        description="Set of tool names that are implemented and can be executed"
    )

