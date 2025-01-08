import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    nilrag: Optional[dict] = {}


class ChoiceChunkContent(BaseModel):
    content: Optional[str]


class ChoiceChunk(BaseModel):
    """Choice chunk."""

    index: int
    delta: ChoiceChunkContent


class ChatCompletionChunk(BaseModel):
    """Chat completion chunk."""

    choices: List[ChoiceChunk]


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str
    logprobs: Optional[dict]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    signature: str


class AttestationResponse(BaseModel):
    verifying_key: str  # PEM encoded public key
    cpu_attestation: str  # Base64 encoded CPU attestation
    gpu_attestation: str  # Base64 encoded GPU attestation


class ModelMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    description: str
    author: str
    license: str
    source: str
    supported_features: List[str]


class ModelEndpoint(BaseModel):
    url: str
    metadata: ModelMetadata


class HealthCheckResponse(BaseModel):
    status: str
    uptime: str
