from __future__ import annotations

import uuid
from typing import Annotated, List, Optional, Any
from pydantic import BaseModel, Field

class ResultContent(BaseModel):
    text: str
    truncated: bool = False


class Source(BaseModel):
    source: str
    content: str


class SearchResult(BaseModel):
    title: str
    body: str
    url: str
    content: ResultContent | None = None

    def as_source(self) -> "Source":
        text = self.content.text if self.content else self.body
        return Source(source=self.url, content=text)

    def model_post_init(self, __context) -> None:
        if self.content is None and isinstance(self.body, str) and self.body:
            self.content = ResultContent(text=self.body)


class Topic(BaseModel):
    topic: str
    needs_search: bool = Field(..., alias="needs_search")


class TopicResponse(BaseModel):
    topics: List[Topic]


class TopicQuery(BaseModel):
    topic: str
    query: str


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

