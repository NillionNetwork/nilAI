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

from pydantic import BaseModel, Field

# Imports from the responses endpoint types
from openai.types.responses.response import Response, ToolChoice  
from openai.types.responses.tool import Tool  
from openai.types.responses.function_tool import FunctionTool  
from openai.types.responses.response_input_item_param import ResponseInputItemParam  
from openai.types.responses.response_output_message import ResponseOutputMessage  
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall

# Assuming this is a shared custom model in your project
from nilai_common.api_models.common_model import Source


# --- Type Aliases for Clarity ---
ResponseFunctionTool: TypeAlias = FunctionTool
InputItem: TypeAlias = ResponseInputItemParam


def _extract_text_from_input_content(content: Any) -> Optional[str]:
    """Extracts text from an input item's content field."""
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


class InputItemAdapter(BaseModel):
    """
    An adapter for a ResponseInputItemParam, providing convenient methods
    for creating and inspecting input items for the Response API.
    """
    raw: InputItem

    @property
    def type(self) -> str:
        return cast(str, self.raw.get("type"))
        
    @property
    def role(self) -> Optional[str]:
        return cast(str, self.raw.get("role"))

    @property
    def content(self) -> Any:
        return self.raw.get("content")

    # Corrected Version
    @staticmethod
    def new_message_item(role: Literal["user", "assistant"], content: str) -> InputItem:
        """Creates a new, simple message-based input item with a role."""
        return cast(InputItem, {"type": "message", "role": role, "content": content})

    @staticmethod
    def new_image_item(
        image_url: str,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> InputItem:
        """Creates a new image input item from a URL."""
        content = {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
        return cast(InputItem, {"type": "image", "content": [content]})

    @staticmethod
    def new_tool_output_item(tool_call_id: str, output: str) -> InputItem:
        """Creates a new tool output item to provide results back to the model."""
        content = {"tool_call_id": tool_call_id, "output": output}
        return cast(InputItem, {"type": "tool_outputs", "content": [content]})

    def is_text_part(self) -> bool:
        """Checks if the input item contains text content."""
        return self.type == "message" and self.role in ("user", "assistant")

    def is_multimodal_part(self) -> bool:
        """Checks if the input item is an image or other non-text type."""
        return self.type in ("image", "audio", "file")

    def extract_text(self) -> Optional[str]:
        """Extracts the text from a text input item."""
        if self.is_text_part() and isinstance(self.content, list):
             for part in self.content:
                 if part.get("type") == "text":
                     return part.get("text")
        return None

    def to_openai_param(self) -> InputItem:
        """Returns the raw dictionary for the API request."""
        return self.raw


def adapt_input_items(items: List[InputItem]) -> List[InputItemAdapter]:
    """Wraps a list of raw input items in the InputItemAdapter."""
    return [InputItemAdapter(raw=item) for item in items]


class WebSearchEnhancedInput(BaseModel):
    """Model for storing input items enhanced with web search sources."""
    input_items: List[InputItem]
    sources: List[Source]


class WebSearchContext(BaseModel):
    """Context for performing a web search."""
    prompt: str
    sources: List[Source]


class ResponseRequest(BaseModel):
    """
    Represents a request to the Responses endpoint, mirroring the structure
    of the chat completions request.
    """
    model: str
    input: List[InputItem] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    stream: Optional[bool] = False
    tools: Optional[Iterable[Tool]] = None
    tool_choice: Optional[ToolChoice] = "auto"
    instructions: Optional[str] = Field(
        default=None,
        description="System-level instructions for the model."
    )
    web_search: Optional[bool] = Field(
        default=False,
        description="Enable web search to enhance context with current information.",
    )

    def model_post_init(self, __context) -> None:
        for i, item in enumerate(self.input):
            content = item.get("content")
            if (
                content is not None
                and hasattr(content, "__iter__")
                and hasattr(content, "__next__")
            ):
                cast(Any, item)["content"] = list(content)

    @property
    def adapted_input_items(self) -> List[InputItemAdapter]:
        return adapt_input_items(self.input)

    def get_last_user_query(self) -> Optional[str]:
        """Finds the last text-based input from the list of items."""
        for item in reversed(self.adapted_input_items):
            if item.is_text_part():
                return item.extract_text()
        return None

    def has_multimodal_content(self) -> bool:
        """Checks if any input item is multimodal (e.g., an image)."""
        return any(item.is_multimodal_part() for item in self.adapted_input_items)

    def ensure_instructions(self, new_instructions: str) -> None:
        """
        Ensures the request has the specified instructions, appending them
        if instructions already exist.
        """
        if not self.instructions:
            self.instructions = new_instructions
        else:
            self.instructions += "\n\n" + new_instructions


class SignedResponse(Response):
    """
    An extension of the official Response object to include a signature
    and a list of sources used to generate the response.
    """
    signature: str
    sources: Optional[List[Source]] = Field(
        default=None, description="Sources used for web search when enabled"
    )