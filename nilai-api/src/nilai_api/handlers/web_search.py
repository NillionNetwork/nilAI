import asyncio
import logging
from typing import List

from ddgs import DDGS
from fastapi import HTTPException, status

from nilai_common.api_model import Source
from nilai_common import Message
from nilai_common.api_model import EnhancedMessages, WebSearchContext

logger = logging.getLogger(__name__)


def perform_web_search_sync(query: str) -> WebSearchContext:
    """Synchronously query Brave and build a contextual prompt.

    The function sends *query* to Brave, extracts the first three text results,
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
            raw_results = list(
                ddgs.text(query=query, max_results=3, region="us-en", backend="brave")
            )

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
                sources.append(
                    Source(
                        source=result.get("href", result.get("url", "")), content=body
                    )
                )

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


async def generate_search_query_from_llm(
    user_message: str, model_name: str, client
) -> str:
    system_prompt = """You are given a user's prompt and your task is to write a short, straight to the point search query that will retrieve the best information to answer it. No punctuation, no newlines, no formatting, no explanation. Do not answer the query but simply write a good query for web search. Only output the query."""
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]
    req = {
        "model": model_name,
        "messages": [m.model_dump() for m in messages],
        "max_tokens": 150,
    }
    response = await client.chat.completions.create(**req)
    logger.info(
        f"For {user_message}, Generated search query: {response.choices[0].message.content.strip()}"
    )
    return response.choices[0].message.content.strip()


async def handle_web_search(
    req_messages: List[Message], model_name: str, client
) -> EnhancedMessages:
    user_query = ""
    for message in reversed(req_messages):
        if message.role == "user":
            user_query = message.content
            break
    if not user_query:
        return EnhancedMessages(messages=req_messages, sources=[])
    try:
        concise_query = await generate_search_query_from_llm(
            user_query, model_name, client
        )
        return await enhance_messages_with_web_search(req_messages, concise_query)
    except Exception:
        return EnhancedMessages(messages=req_messages, sources=[])
