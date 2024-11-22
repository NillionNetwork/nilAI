from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]

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
    verifying_key: str      # PEM encoded public key
    cpu_attestation: str    # Base64 encoded CPU attestation
    gpu_attestation: str    # Base64 encoded GPU attestation
    
class Model(BaseModel):
    id: str
    name: str
    description: str
    author: str
    license: str
    source: str
