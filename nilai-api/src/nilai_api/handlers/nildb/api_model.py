from pydantic import BaseModel, ConfigDict


class PromptDelegationRequest(BaseModel):
    """Prompt Delegation Token Request model"""

    model_config = ConfigDict(validate_assignment=True)
    used_id: str


class PromptDelegationToken(BaseModel):
    """Delegation token model"""

    model_config = ConfigDict(validate_assignment=True)

    token: str
    did: str
