from __future__ import annotations

from typing import (
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

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat.chat_completion import Choice as OpenaAIChoice
from pydantic import BaseModel, Field

from nilai_common.api_models.common_model import Source

ChatToolFunction: TypeAlias = Function
ImageContent: TypeAlias = ChatCompletionContentPartImageParam
TextContent: TypeAlias = ChatCompletionContentPartTextParam
Message: TypeAlias = ChatCompletionMessageParam


class Choice(OpenaAIChoice):
    pass


def _extract_text_from_content(content: Any) -> Optional[str]:
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


class MessageAdapter(BaseModel):
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
        cast(Any, self.raw)["role"] = value

    @property
    def content(self) -> Any:
        return self.raw.get("content")

    @content.setter
    def content(self, value: Any) -> None:
        cast(Any, self.raw)["content"] = value

    @staticmethod
    def new_message(
        role: Literal["developer", "user", "system", "assistant", "tool", "function"],
        content: Union[str, List[Any]],
    ) -> Message:
        message: Message = cast(Message, {"role": role, "content": content})
        return message

    @staticmethod
    def new_tool_message(
        name: str,
        content: str,
        tool_call_id: str,
    ) -> Message:
        message: Message = cast(
            Message,
            {
                "role": "tool",
                "name": name,
                "content": content,
                "tool_call_id": tool_call_id,
            },
        )
        return message

    @staticmethod
    def new_assistant_tool_call_message(
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> Message:
        return cast(
            Message,
            {
                "role": "assistant",
                "tool_calls": [tc.model_dump(exclude_unset=True) for tc in tool_calls],
                "content": None,
            },
        )

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
        if c is None:
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
        return self.raw


def adapt_messages(msgs: List[Message]) -> List[MessageAdapter]:
    return [MessageAdapter(raw=m) for m in msgs]


class WebSearchEnhancedMessages(BaseModel):
    messages: List[Message]
    sources: List[Source]


class WebSearchContext(BaseModel):
    prompt: str
    sources: List[Source]


class ChatRequest(BaseModel):
    model: str
    messages: List[Message] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=100000)
    stream: Optional[bool] = False
    tools: Optional[Iterable[ChatCompletionToolParam]] = None
    tool_choice: Optional[Union[str, dict]] = "auto"
    nilrag: Optional[dict] = {}
    web_search: Optional[bool] = Field(
        default=False,
        description="Enable web search to enhance context with current information",
    )

    def model_post_init(self, __context) -> None:
        for i, msg in enumerate(self.messages):
            content = msg.get("content")
            if (
                content is not None
                and hasattr(content, "__iter__")
                and hasattr(content, "__next__")
            ):
                cast(Any, msg)["content"] = list(content)

    @property
    def adapted_messages(self) -> List[MessageAdapter]:
        return adapt_messages(self.messages)

    def get_last_user_query(self) -> Optional[str]:
        for m in reversed(self.adapted_messages):
            if m.role == "user" and m.is_text_part():
                return m.extract_text()
        return None

    def has_multimodal_content(self) -> bool:
        return any([m.is_multimodal_part() for m in self.adapted_messages])

    def ensure_system_content(self, system_content: str) -> None:
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
