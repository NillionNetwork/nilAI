import asyncio
import logging
from typing import List

from duckduckgo_search import DDGS
from fastapi import HTTPException, status

from nilai_common.api_model import Source
from nilai_common import Message
from nilai_common.api_model import EnhancedMessages, WebSearchContext

logger = logging.getLogger(__name__)


def perform_web_search_sync(query: str) -> WebSearchContext:
    """Synchronously query DuckDuckGo and build a contextual prompt.

    The function sends *query* to DuckDuckGo, extracts the first three text results,
    formats them in a single prompt, and returns that prompt together with the
    metadata (URL and snippet) of every result.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Web search requested with an empty query",
        )

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=3, region="us-en"))

        if not raw_results:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Web search failed, service currently unavailable",
            )

        snippets: List[str] = []
        sources: List[Source] = []

        for result in raw_results:
            if result.get("title") and result.get("body"):
                title = result["title"]
                body = result["body"][:500]
                snippets.append(f"{title}: {body}")
                sources.append(Source(source=result["href"], content=body))

        prompt = (
            "You have access to the following current information from web search:\n"
            + "\n".join(snippets)
        )

        return WebSearchContext(prompt=prompt, sources=sources)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error performing web search: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search failed, service currently unavailable",
        ) from exc


async def get_web_search_context(query: str) -> WebSearchContext:
    """Non-blocking wrapper around *perform_web_search_sync*."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, perform_web_search_sync, query)


async def enhance_messages_with_web_search(
    messages: List[Message], query: str
) -> EnhancedMessages:
    ctx = await get_web_search_context(query)
    enhanced = [Message(role="system", content=ctx.prompt)] + messages
    return EnhancedMessages(messages=enhanced, sources=ctx.sources)


async def handle_web_search(req_messages: List[Message]) -> EnhancedMessages:
    """Handle web search for the given messages.

    Only the last user message is used as the query.
    """

    user_query = ""
    for message in reversed(req_messages):
        if message.role == "user":
            user_query = message.content
            break

    if not user_query:
        return EnhancedMessages(messages=req_messages, sources=[])
    try:
        return await enhance_messages_with_web_search(req_messages, user_query)
    except Exception:
        return EnhancedMessages(messages=req_messages, sources=[])
