from dataclasses import dataclass
from typing import List, Optional, Any
from fastapi import HTTPException
from nilai_common import Message

@dataclass(frozen=True)
class MultimodalCheck:
    has_multimodal: bool
    error: Optional[str] = None


def _extract_url(image_url_field: Any) -> Optional[str]:
    """
    Support both object-with-attr and dict-like shapes.
    Returns the URL string or None.
    """
    if image_url_field is None:
        return None

    url = getattr(image_url_field, "url", None)
    if url is not None:
        return url
    if isinstance(image_url_field, dict):
        return image_url_field.get("url")
    return None


def multimodal_check(messages: List[Message]) -> MultimodalCheck:
    """
    Single-pass check:
      - detect if any part is type=='image_url'
      - validate that image_url.url exists and is a base64 data URL
    Returns:
      MultimodalCheck(has_multimodal: bool, error: Optional[str])
    """
    has_mm = False

    for m in messages:
        content = getattr(m, "content", None) or []
        for item in content:
            if getattr(item, "type", None) == "image_url":
                has_mm = True
                iu = getattr(item, "image_url", None)
                url = _extract_url(iu)
                if not url:
                    return MultimodalCheck(True, "image_url.url is required for image_url parts")
                if not (url.startswith("data:image/") and ";base64," in url):
                    return MultimodalCheck(True, "Only base64 data URLs are allowed for images (data:image/...;base64,...)")

    return MultimodalCheck(has_mm, None)


def has_multimodal_content(messages: List[Message], precomputed: Optional[MultimodalCheck] = None) -> bool:
    """
    Check if any message contains multimodal content (image_url parts).
    
    Args:
        messages: List of messages to check
        precomputed: Optional precomputed result from multimodal_check() to avoid re-iterating
        
    Returns:
        True if any message contains image_url parts, False otherwise
    """
    res = precomputed or multimodal_check(messages)
    return res.has_multimodal


def validate_multimodal_content(messages: List[Message], precomputed: Optional[MultimodalCheck] = None) -> None:
    """
    Validate that multimodal content (image_url parts) follows the required format.
    
    Args:
        messages: List of messages to validate
        precomputed: Optional precomputed result from multimodal_check() to avoid re-iterating
        
    Raises:
        HTTPException(400): When image_url parts don't have required URL or use invalid format
                           (only base64 data URLs are allowed: data:image/...;base64,...)
    """
    res = precomputed or multimodal_check(messages)
    if res.error:
        raise HTTPException(status_code=400, detail=res.error)
