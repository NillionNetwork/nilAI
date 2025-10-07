# nilai_common/api_models/responses_model.py

from __future__ import annotations
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field
from openai.types.responses import Response, ResponseCompletedEvent, ResponseInputParam, ToolParam
from openai.types.shared import Metadata

from openai.types.responses.response_create_params import ToolChoice

from .common_model import Source


class ResponseRequest(BaseModel):
    """
    A Pydantic model for validating incoming requests to the /v1/responses endpoint.
    """
    model: str
    input: Union[str, ResponseInputParam]
    instructions: Optional[str] = None
    stream: Optional[bool] = False
    tools: Optional[List[ToolParam]] = None
    tool_choice: Optional[ToolChoice] = None

    web_search: bool = Field(default=False, exclude=True)

    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Optional[Metadata] = None

    class Config:
        arbitrary_types_allowed = True

    def has_multimodal_content(self) -> bool:
        if isinstance(self.input, str):
            return False
        
        if isinstance(self.input, list):
            for item in self.input:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in ("image_url", "input_image", "image"):
                        return True
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") in (
                                "image_url",
                                "input_image",
                                "image",
                            ):
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
                                if isinstance(part, dict) and part.get("type") == "text":
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