from typing import Union, List
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)


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
            if part.type == "text":
                return part.text
    return ""
