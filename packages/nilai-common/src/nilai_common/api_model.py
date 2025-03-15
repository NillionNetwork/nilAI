import uuid
from enum import Enum
from typing import Dict, List, Optional, Literal, Iterable
from uuid import UUID


from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenaAIChoice
from openai.types.chat.chat_completion import CompletionUsage
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field, model_validator


__all__ = [
    "Message",
    "Choice",
    "ChatRequest",
    "CompletionUsage",
    "SignedChatCompletion",
    "AttestationResponse",
    "ModelMetadata",
    "ModelEndpoint",
    "HealthCheckResponse",
]


class Message(ChatCompletionMessage):
    role: Literal["system", "user", "assistant", "tool"]


class Choice(OpenaAIChoice):
    pass

class SecretVaultPayload(BaseModel):
    org_did: str
    secret_key: str
    inject_from: Optional[UUID] = None
    filter_: Optional[Dict] = Field(None, alias="filter")
    save_to: Optional[UUID] = None

    @model_validator(mode='after')
    def check_required_fields(self):
        if not self.save_to and not (self.inject_from and self.filter):
            raise ValueError("Either 'save_to' must be provided or both 'inject_from' and 'filter' must be provided")
        return self


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    nilrag: Optional[dict] = {}
    secret_vault: Optional[SecretVaultPayload] = None


class SignedChatCompletion(ChatCompletion):
    signature: str
    secret_vault: List[str] = []


class AttestationResponse(BaseModel):
    verifying_key: str  # PEM encoded public key
    cpu_attestation: str  # Base64 encoded CPU attestation
    gpu_attestation: str  # Base64 encoded GPU attestation


class AllowedModelRoles(str, Enum):
    WORKER = "worker"
    GENERATION = "generation"
    REASONING = "reasoning"

class ModelMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    description: str
    author: str
    license: str
    source: str
    role: AllowedModelRoles
    supported_features: List[str]
    tool_support: bool


class ModelEndpoint(BaseModel):
    url: str
    metadata: ModelMetadata


class HealthCheckResponse(BaseModel):
    status: str
    uptime: str
