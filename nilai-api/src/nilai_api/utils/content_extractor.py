from typing import Union, List, Optional, Iterable, cast, Any
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
from openai.types.chat import ChatCompletionMessageParam


def _iter_parts(content):
    if content is None or isinstance(content, str):
        return ()
    if isinstance(content, dict):
        return (content,)
    try:
        iter(content)
        return content
    except TypeError:
        return ()


def _is_multimodal_part(part) -> bool:
    if isinstance(part, dict):
        if "image_url" in part or "input_audio" in part or "file" in part:
            return True
        t = part.get("type")
        return t in {"image_url", "input_audio", "file"}
    if (
        hasattr(part, "image_url")
        or hasattr(part, "input_audio")
        or hasattr(part, "file")
    ):
        return True
    t = getattr(part, "type", None)
    return t in {"image_url", "input_audio", "file"}


def extract_text_content(
    content: Union[
        str,
        Iterable[
            Union[
                ChatCompletionContentPartTextParam,
                ChatCompletionContentPartImageParam,
                dict,
            ]
        ],
        None,
    ],
) -> str:
    if isinstance(content, str):
        return content
    for part in _iter_parts(content):
        if isinstance(part, dict):
            if part.get("type") == "text":  # type: ignore
                txt = part.get("text")  # type: ignore
                if isinstance(txt, str):
                    return txt
        else:
            if getattr(part, "type", None) == "text":
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    return txt
    return ""


def has_multimodal_content(messages: List[ChatCompletionMessageParam]) -> bool:
    for m in messages:
        for part in _iter_parts(m.get("content")):
            if _is_multimodal_part(part):
                return True
    return False


def get_last_user_query(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if content is not None:
                try:
                    text = extract_text_content(content)  # type: ignore
                except Exception:
                    text = None
                if text:
                    return text.strip()
    return None
