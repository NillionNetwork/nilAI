from pydantic import BaseModel, ConfigDict
from typing import TypeAlias

PromptDelegationRequest: TypeAlias = str


class PromptDelegationToken(BaseModel):
    """Delegation token model"""

    model_config = ConfigDict(validate_assignment=True)

    token: str
    did: str
