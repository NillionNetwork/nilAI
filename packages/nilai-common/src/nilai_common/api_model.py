import uuid

from typing import Annotated, Iterable, List, Literal, Optional, Union

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat import ChatCompletionToolParam
from openai.types.chat.chat_completion import Choice as OpenaAIChoice
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from pydantic import BaseModel, Field


class Message(ChatCompletionMessage):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[
        str,
        List[
            Union[
                ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
            ]
        ],
    ]


class Choice(OpenaAIChoice):
    pass


class Source(BaseModel):
    source: str
    content: str


class SearchResult(BaseModel):
    title: str
    body: str
    url: str


class WebSearchEnhancedMessages(BaseModel):
    messages: List[Message]
    sources: List[Source]


class WebSearchContext(BaseModel):
    """Prompt and sources obtained from a web search."""

    prompt: str
    sources: List[Source]


class ChatRequest(BaseModel):
    model: str
    messages: List[Message] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=5.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=100000)
    stream: Optional[bool] = False
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    nilrag: Optional[dict] = {}
    web_search: Optional[bool] = Field(
        default=False,
        description="Enable web search to enhance context with current information",
    )


class SignedChatCompletion(ChatCompletion):
    signature: str
    sources: Optional[List[Source]] = Field(
        default=None, description="Sources used for web search when enabled"
    )


class ModelMetadata(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    description: str
    author: str
    license: str
    source: str
    supported_features: List[str]
    tool_support: bool
    multimodal_support: bool = False


class ModelEndpoint(BaseModel):
    url: str
    metadata: ModelMetadata


class HealthCheckResponse(BaseModel):
    status: str
    uptime: str


Nonce = Annotated[
    str,
    Field(
        max_length=64,
        min_length=64,
        description="The nonce to be used for the attestation",
    ),
]

AMDAttestationToken = Annotated[
    str, Field(description="The attestation token from AMD's attestation service")
]

NVAttestationToken = Annotated[
    str, Field(description="The attestation token from NVIDIA's attestation service")
]


class AttestationReport(BaseModel):
    nonce: Nonce
    verifying_key: Annotated[str, Field(description="PEM encoded public key")]
    cpu_attestation: AMDAttestationToken
    gpu_attestation: NVAttestationToken
