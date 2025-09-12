from __future__ import annotations

import uuid
from typing import (
    Annotated,
    Iterable,
    List,
    Optional,
    Any,
    cast,
    TypeAlias,
    Literal,
    Union,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion import Choice as OpenaAIChoice
from pydantic import BaseModel, Field


# ---------- Aliases from the OpenAI SDK ----------
ImageContent: TypeAlias = ChatCompletionContentPartImageParam
TextContent: TypeAlias = ChatCompletionContentPartTextParam
Message: TypeAlias = ChatCompletionMessageParam  # SDK union of message shapes


# ---------- Domain-specific objects for web search ----------
class ResultContent(BaseModel):
    text: str
    truncated: bool = False


class Choice(OpenaAIChoice):
    pass


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
        # Auto-derive structured fields when not provided
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


# ---------- Helpers ----------
def _extract_text_from_content(content: Any) -> Optional[str]:
    """
    - If content is a str -> return it (stripped) if non-empty.
    - If content is a list of content parts -> concatenate 'text' parts.
    - Else -> None.
    """
    if isinstance(content, str):
        s = content.strip()
        return s or None
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        if parts:
            return "\n".join(parts)
    return None


# ---------- Adapter over the raw SDK message ----------
class MessageAdapter(BaseModel):
    """Thin wrapper around an OpenAI ChatCompletionMessageParam with convenience methods."""

    raw: Message

    @property
    def role(self) -> str:
        return cast(str, self.raw.get("role"))

    @role.setter
    def role(
        self,
        value: Literal["developer", "user", "system", "assistant", "tool", "function"],
    ) -> None:
        if not isinstance(value, str):
            raise TypeError("role must be a string")
        # Update the underlying SDK message dict
        # Cast to Any to bypass TypedDict restrictions
        cast(Any, self.raw)["role"] = value

    @property
    def content(self) -> Any:
        return self.raw.get("content")

    @content.setter
    def content(self, value: Any) -> None:
        # Update the underlying SDK message dict
        # Cast to Any to bypass TypedDict restrictions
        cast(Any, self.raw)["content"] = value

    @staticmethod
    def new_message(
        role: Literal["developer", "user", "system", "assistant", "tool", "function"],
        content: Union[str, List[Any]],
    ) -> Message:
        message: Message = cast(Message, {"role": role, "content": content})
        return message

    @staticmethod
    def new_completion_message(content: str) -> ChatCompletionMessage:
        message: ChatCompletionMessage = cast(
            ChatCompletionMessage, {"role": "assistant", "content": content}
        )
        return message

    def is_text_part(self) -> bool:
        return _extract_text_from_content(self.content) is not None

    def is_multimodal_part(self) -> bool:
        c = self.content
        if c is None:  # tool calling message
            return False
        if isinstance(c, str):
            return False

        for part in c:
            if isinstance(part, dict) and part.get("type") in (
                "image_url",
                "input_image",
            ):
                return True
        return False

    def extract_text(self) -> Optional[str]:
        return _extract_text_from_content(self.content)

    def to_openai_param(self) -> Message:
        # Return the original dict for API calls.
        return self.raw


def adapt_messages(msgs: List[Message]) -> List[MessageAdapter]:
    return [MessageAdapter(raw=m) for m in msgs]


# ---------- Your additional containers ----------
class WebSearchEnhancedMessages(BaseModel):
    messages: List[Message]
    sources: List[Source]


class WebSearchContext(BaseModel):
    """Prompt and sources obtained from a web search."""

    prompt: str
    sources: List[Source]


# ---------- Request/response models ----------
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

    def model_post_init(self, __context) -> None:
        # Process messages after model initialization
        for i, msg in enumerate(self.messages):
            content = msg.get("content")
            if (
                content is not None
                and hasattr(content, "__iter__")
                and hasattr(content, "__next__")
            ):
                # Convert iterator to list in place
                cast(Any, msg)["content"] = list(content)

    @property
    def adapted_messages(self) -> List[MessageAdapter]:
        return adapt_messages(self.messages)

    def get_last_user_query(self) -> Optional[str]:
        """
        Returns the latest non-empty user text (plain or from content parts),
        or None if not found.
        """
        for m in reversed(self.adapted_messages):
            if m.role == "user" and m.is_text_part():
                return m.extract_text()
        return None

    def has_multimodal_content(self) -> bool:
        """True if any message contains an image content part."""
        return any([m.is_multimodal_part() for m in self.adapted_messages])

    def ensure_system_content(self, system_content: str) -> None:
        """Ensure the conversation starts with a system message containing the given content.

        This method directly mutates the `self.messages` list in place.

        Logic cases:
        1. Empty message list: Insert new system message at the beginning
        2. First message is not system: Insert new system message at the beginning
        3. First message is system: Merge content with existing system message
           - String content: Append with separator
           - List content: Add new text part to the list
        """
        msgs = self.messages

        if not msgs:
            msgs.insert(
                0, MessageAdapter.new_message(role="system", content=system_content)
            )
            return

        first_message = msgs[0]

        if first_message.get("role") != "system":
            msgs.insert(
                0, MessageAdapter.new_message(role="system", content=system_content)
            )
            return

        existing_text = MessageAdapter(raw=first_message).extract_text() or ""
        content = first_message.get("content")

        if content is None or isinstance(content, str):
            first_message["content"] = (
                existing_text + ("\n\n" if existing_text else "") + system_content
            )
        elif isinstance(content, list):
            prefix = "\n\n" if existing_text else ""
            content.append({"type": "text", "text": prefix + system_content})


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


# ---------- Attestation ----------
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
