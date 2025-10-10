from __future__ import annotations

from typing import Iterable, List, Optional, Union
from typing_extensions import Literal, TypeAlias

from pydantic import BaseModel, Field

from openai.types.responses import Response, ResponseInputParam, ToolParam, ResponseFunctionToolCall
from openai.types.responses.response_includable import ResponseIncludable
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.tool_choice_options import ToolChoiceOptions
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from openai.types.responses.tool_choice_types_param import ToolChoiceTypesParam
from openai.types.responses.tool_choice_function_param import ToolChoiceFunctionParam
from openai.types.responses.tool_choice_mcp_param import ToolChoiceMcpParam
from openai.types.responses.tool_choice_custom_param import ToolChoiceCustomParam
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_conversation_param import ResponseConversationParam
from openai.types.shared_params.metadata import Metadata
from openai.types.shared_params.reasoning import Reasoning
from openai.types.shared_params.responses_model import ResponsesModel

from .common_model import Source

ToolChoice: TypeAlias = Union[
    ToolChoiceOptions,
    ToolChoiceAllowedParam,
    ToolChoiceTypesParam,
    ToolChoiceFunctionParam,
    ToolChoiceMcpParam,
    ToolChoiceCustomParam,
]

Conversation: TypeAlias = Union[str, ResponseConversationParam]

class StreamOptions(BaseModel):
    include_obfuscation: Optional[bool] = None


class ResponseRequest(BaseModel):
    model: ResponsesModel

    input: Union[str, ResponseInputParam]
    instructions: Optional[str] = None

    stream: Optional[bool] = Field(
        default=False,
        description=(
            "If true, the response will stream using SSE. Matches the official "
            "ResponseCreateParamsStreaming vs NonStreaming union behavior."
        ),
    )
    stream_options: Optional[StreamOptions] = None

    tools: Optional[Iterable[ToolParam]] = None
    tool_choice: Optional[ToolChoice] = None
    parallel_tool_calls: Optional[bool] = None
    max_tool_calls: Optional[int] = Field(default=None, ge=0)

    background: Optional[bool] = None
    conversation: Optional[Conversation] = None
    previous_response_id: Optional[str] = None
    include: Optional[List[ResponseIncludable]] = None
    prompt: Optional[ResponsePromptParam] = None

    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)

    text: Optional[ResponseTextConfigParam] = None

    reasoning: Optional[Reasoning] = None

    prompt_cache_key: Optional[str] = None
    safety_identifier: Optional[str] = None
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None
    store: Optional[bool] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    user: Optional[str] = None

    metadata: Optional[Metadata] = None

    web_search: Optional[bool] = Field(
        default=False,
        description="Enable web search to enhance context with current information",
    )

    class Config:
        arbitrary_types_allowed = True

    def has_multimodal_content(self) -> bool:  
        if isinstance(self.input, str):  
            return False  
    
        if isinstance(self.input, list):  
            for item in self.input:  
                if isinstance(item, dict):  
                    if item.get("type") == "input_image":  
                        return True  
                    
                    content = item.get("content")  
                    if isinstance(content, list):  
                        for part in content:  
                            if isinstance(part, dict) and part.get("type") == "input_image":  
                                return True  
        return False

    def get_last_user_query(self) -> Optional[str]:
        if isinstance(self.input, str):
            return self.input

        if isinstance(self.input, list):
            for item in reversed(self.input):
                if isinstance(item, dict):
                    role = item.get("role")
                    if role == "user":
                        content = item.get("content")
                        if isinstance(content, str):
                            return content.strip() or None
                        elif isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "input_text":
                                    text = part.get("text")
                                    if isinstance(text, str) and text.strip():
                                        text_parts.append(text.strip())
                            if text_parts:
                                return "\n".join(text_parts)
        return None

    def ensure_instructions(self, additional_instructions: str) -> None:
        if self.instructions:
            self.instructions = self.instructions + "\n\n" + additional_instructions
        else:
            self.instructions = additional_instructions


class WebSearchEnhancedInput(BaseModel):
    input: Union[str, ResponseInputParam]
    instructions: Optional[str]
    sources: List[Source]


class SignedResponse(Response):
    """
    An extension of the official Response object.
    """
    signature: str
    sources: Optional[List[Source]] = Field(
        default=None, description="Sources used for web search when enabled"
    )
