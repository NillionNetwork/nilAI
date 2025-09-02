import logging
import re
from functools import lru_cache
from typing import List, Dict, Any

import httpx
from fastapi import HTTPException, status

from nilai_api.config import WEB_SEARCH_SETTINGS
from nilai_common.api_model import (
    ChatRequest,
    Message,
    MessageAdapter,
    SearchResult,
    Source,
    WebSearchEnhancedMessages,
    WebSearchContext,
)

logger = logging.getLogger(__name__)

_BRAVE_API_HEADERS = {
    "Api-Version": "2023-10-11",
    "Accept": "application/json",
}

_BRAVE_API_PARAMS_BASE = {
    "summary": 1,
    "count": WEB_SEARCH_SETTINGS.count,
    "country": WEB_SEARCH_SETTINGS.country,
    "lang": WEB_SEARCH_SETTINGS.lang,
}


@lru_cache(maxsize=1)
def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared HTTP client instance to avoid creating it on every request.

    Returns:
        An AsyncClient configured with timeouts and connection limits
    """
    return httpx.AsyncClient(
        timeout=WEB_SEARCH_SETTINGS.timeout,
        limits=httpx.Limits(
            max_connections=WEB_SEARCH_SETTINGS.max_concurrent_requests
        ),
    )


async def _make_brave_api_request(query: str) -> Dict[str, Any]:
    """Make an API request to the Brave Search API.

    Args:
        query: The search query string to execute

    Returns:
        Dict containing the raw API response data

    Raises:
        HTTPException: If API key is missing or API request fails
    """
    if not WEB_SEARCH_SETTINGS.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Missing BRAVE_SEARCH_API key in environment",
        )

    q = " ".join(query.split())
    q = " ".join(q.split()[:50])[:400]

    params = {**_BRAVE_API_PARAMS_BASE, "q": q}
    headers = {
        **_BRAVE_API_HEADERS,
        "X-Subscription-Token": WEB_SEARCH_SETTINGS.api_key,
    }

    client = _get_http_client()
    logger.info("Brave API request start")
    logger.debug(
        "Brave API params assembled q_len=%d country=%s lang=%s count=%s",
        len(q),
        params.get("country"),
        params.get("lang"),
        params.get("count"),
    )
    resp = await client.get(
        WEB_SEARCH_SETTINGS.api_path, headers=headers, params=params
    )

    if resp.status_code >= 400:
        logger.error("Brave API error: %s - %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search failed, service currently unavailable",
        )

    try:
        data = resp.json()
        logger.info("Brave API request success")
        logger.debug(
            "Brave API response keys=%s",
            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
        return data
    except Exception:
        logger.exception("Failed to parse Brave API JSON")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web search failed: invalid response from provider",
        )


def _parse_brave_results(data: Dict[str, Any]) -> List[SearchResult]:
    """Parse raw Brave Search API results into SearchResult objects.

    Args:
        data: Raw API response data

    Returns:
        List of SearchResult objects with title, body and URL
    """
    web_block = data.get("web", {}) if isinstance(data, dict) else {}
    raw_results = web_block.get("results", []) if isinstance(web_block, dict) else []

    results: List[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "")[:200]
        body = item.get("description") or item.get("snippet") or item.get("body", "")
        url = item.get("url") or item.get("link") or item.get("href", "")

        if title and body and url:
            results.append(SearchResult(title=title, body=body, url=url))
    logger.debug("Parsed brave results count=%d", len(results))
    return results


def _sanitize_query(query: str) -> str:
    if not query:
        return ""
    query = query.strip().strip("'\"`")
    query = re.sub(r"[.!?]+$", "", query)
    query = re.sub(r"\s+", " ", query)
    return query.strip()


async def perform_web_search_async(query: str) -> WebSearchContext:
    """Perform an asynchronous web search using the Brave Search API.

    Args:
        query: The search query string to execute

    Returns:
        WebSearchContext containing formatted search results and source information

    Raises:
        HTTPException: If query is empty, no results found, or API errors occur
    """
    query_sanitized = _sanitize_query(query)
    if not query_sanitized:
        logger.warning("Empty or invalid query after sanitization")
        return WebSearchContext(prompt="", sources=[])

    logger.info("Web search start")
    logger.debug("Web search sanitized query: %s", query_sanitized)

    try:
        data = await _make_brave_api_request(query_sanitized)
        results = _parse_brave_results(data)
    except HTTPException:
        logger.exception("Brave API request failed")
        return WebSearchContext(prompt=query, sources=[])

    if not results:
        logger.warning("No web results found for query: %s", query_sanitized)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No web search results found",
        )

    logger.info("Web search results ready count=%d", len(results))
    lines = [
        f"[{idx}] {r.title}\nURL: {r.url}\nSnippet: {r.body}"
        for idx, r in enumerate(results, start=1)
    ]
    prompt = "\n".join(lines)
    sources = [Source(source=r.url, content=r.body) for r in results]

    return WebSearchContext(prompt=prompt, sources=sources)


async def enhance_messages_with_web_search(
    messages: List[Message], query: str
) -> WebSearchEnhancedMessages:
    """Enhance a list of messages with web search context.

    Args:
        messages: List of conversation messages to enhance
        query: Search query to retrieve web search results for

    Returns:
        WebSearchEnhancedMessages containing the original messages with web search
        context prepended as a system message, along with source information
    """
    ctx = await perform_web_search_async(query)
    query_source = Source(source="web_search_query", content=query)

    web_search_content = (
        f'You have access to the following web search results for the query: "{query}"\n\n'
        "Use this information to provide accurate and up-to-date answers. "
        "Cite the sources when appropriate.\n\n"
        "Web Search Results:\n"
        f"{ctx.prompt}\n\n"
        "Please provide a comprehensive answer based on the search results above."
    )

    enhanced: List[Message] = [
        MessageAdapter.new_message(role="system", content=web_search_content)
    ]

    for msg in messages:
        adapted_message = MessageAdapter(raw=msg)
        enhanced.append(adapted_message.to_openai_param())

    return WebSearchEnhancedMessages(
        messages=enhanced,
        sources=[query_source] + ctx.sources,
    )


async def generate_search_query_from_llm(
    user_message: str, model_name: str, client
) -> str:
    """
    Use the LLM to produce a concise, high-recall search query.
    """
    system_prompt = (
        "You are given a user question. Generate a concise web search query that will best retrieve information "
        "to answer the question. If the user’s question is already optimal, repeat it exactly.\n"
        "- Do not add assumptions not present in the question.\n"
        "- Do not answer the question.\n"
        "- The query must contain at least 10 words.\n"
        "Output only the search query. Never answer the user's question (it will likely always be a question inputed by the user).\n\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    req = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 150,
    }

    logger.info("Generate search query start model=%s", model_name)
    logger.debug(
        "User message len=%d", len(user_message) if isinstance(user_message, str) else 0
    )
    try:
        response = await client.chat.completions.create(**req)
    except Exception as exc:
        logger.exception("LLM call failed")
        raise RuntimeError(f"Failed to generate search query: {exc}") from exc

    try:
        choices = getattr(response, "choices", None) or []
        msg = choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
    except Exception as exc:
        logger.exception("Invalid LLM response structure")
        raise RuntimeError(f"Invalid response structure from LLM: {exc}") from exc

    if not content:
        logger.error("LLM returned empty search query")
        raise RuntimeError("LLM returned an empty search query")

    logger.info("Generate search query success")
    logger.debug("Generated query len=%d", len(content))
    return content


async def handle_web_search(
    req_messages: ChatRequest, model_name: str, client
) -> WebSearchEnhancedMessages:
    """Handle web search enhancement for a conversation.

    Extracts the most recent user message, generates an optimized search query
    using an LLM, and enhances the conversation with web search results.

    Args:
        req_messages: List of conversation messages to process
        model_name: Name of the LLM model to use for query generation
        client: LLM client instance for making API calls

    Returns:
        WebSearchEnhancedMessages with web search context added, or original
        messages if no user query is found or search fails
    """
    logger.info("Handle web search start")
    logger.debug(
        "Handle web search messages_in=%d model=%s",
        len(req_messages.messages),
        model_name,
    )
    user_query = req_messages.get_last_user_query()
    if not user_query:
        logger.info("No user query found")
        return WebSearchEnhancedMessages(messages=req_messages.messages, sources=[])

    try:
        concise_query = await generate_search_query_from_llm(
            user_query, model_name, client
        )
        logger.info("Enhancing messages with web search context")
        return await enhance_messages_with_web_search(
            req_messages.messages, concise_query
        )

    except HTTPException:
        logger.exception("Web search provider error")
        return WebSearchEnhancedMessages(messages=req_messages.messages, sources=[])

    except Exception:
        logger.exception("Unexpected error during web search handling")
        return WebSearchEnhancedMessages(messages=req_messages.messages, sources=[])
