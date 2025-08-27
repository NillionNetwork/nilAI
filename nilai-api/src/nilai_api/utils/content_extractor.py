from typing import Union, List, Optional
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from nilai_common import Message


def extract_text_content(
    content: Union[
        str,
        List[
            Union[
                ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
            ]
        ],
    ],
) -> str:
    """
    Extract text content from a message content field.

    Args:
        content: Either a string or a list of content parts

    Returns:
        str: The extracted text content, or empty string if no text content found
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        for part in content:
            if part["type"] == "text":
                return part["text"]
    return ""


def has_multimodal_content(messages: List[Message]) -> bool:
    """Check if any message contains multimodal content with image_url type."""
    for msg in messages:
        content = getattr(msg, "content", None)
        if hasattr(content, '__iter__') and not isinstance(content, str):
            content_list = list(content)
            for part in content_list:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def get_last_user_query(messages: List[Message]) -> Optional[str]:
    """
    Walk from the end to find the most recent user-authored content, and
    extract text from either a string or a multimodal content list.
    """
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "user":
            content = getattr(msg, "content", None)
            if content is not None:
                try:
                    text = extract_text_content(content)
                except Exception:
                    text = None
                if text:
                    return text.strip()
    return None
