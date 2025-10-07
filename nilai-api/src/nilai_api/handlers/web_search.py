import logging
import re
import asyncio
from functools import lru_cache
from typing import List, Dict, Any

import httpx
import trafilatura
from fastapi import HTTPException, status

from nilai_api.config import CONFIG
from nilai_common.api_models import (
    ChatRequest,
    MessageAdapter,
    SearchResult,
    Source,
    WebSearchEnhancedMessages,
    WebSearchContext,
    ResultContent,
    TopicResponse,
    Topic,
    TopicQuery,
)

logger = logging.getLogger(__name__)

# Common source-type identifier for recording the original query used in web search
WEB_SEARCH_QUERY_SOURCE = "web_search_query"

_BRAVE_API_HEADERS = {
    "Api-Version": "2023-10-11",
    "Accept": "application/json",
}

_BRAVE_API_PARAMS_BASE = {
    "summary": 1,
    "count": CONFIG.web_search.count,
    "country": CONFIG.web_search.country,
    "lang": CONFIG.web_search.lang,
}


@lru_cache(maxsize=1)
def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared HTTP client instance to avoid creating it on every request.

    Returns:
        An AsyncClient configured with timeouts and connection limits
    """
    return httpx.AsyncClient(
        timeout=CONFIG.web_search.timeout,
        limits=httpx.Limits(max_connections=CONFIG.web_search.max_concurrent_requests),
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
    if not CONFIG.web_search.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Missing BRAVE_SEARCH_API key in environment",
        )

    q = " ".join(query.split())

    params = {**_BRAVE_API_PARAMS_BASE, "q": q}
    headers = {
        **_BRAVE_API_HEADERS,
        "X-Subscription-Token": CONFIG.web_search.api_key,
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
    resp = await client.get(CONFIG.web_search.api_path, headers=headers, params=params)

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


def _format_search_results(results: List[SearchResult]) -> str:
    """Format search results into a readable, indexed text block.

    Args:
        results: List of SearchResult items to include in the output.

    Returns:
        A newline-joined string where each entry contains an index, title,
        URL, and content snippet suitable for inclusion in a system prompt.
    """
    lines = [
        f"[{idx}] {r.title}\nURL: {r.url}\nContent: {r.body}"
        for idx, r in enumerate(results, start=1)
    ]
    return "\n".join(lines)


async def _fetch_and_extract_page_content(
    url: str, client: httpx.AsyncClient
) -> ResultContent | None:
    """Fetches a URL and extracts its main text content using trafilatura.

    Args:
        url: The URL to fetch.
        client: The httpx client to use for the request.

    Returns:
        The cleaned main text content of the page, or None if it fails.
    """
    try:
        resp = await client.get(url, follow_redirects=True, timeout=10.0)
        resp.raise_for_status()
        extracted_text = trafilatura.extract(
            resp.text, include_comments=False, include_tables=False
        )
        if extracted_text:
            clean_text = re.sub(r"\s+", " ", extracted_text).strip()
            truncated = len(clean_text) > 5000
            text = clean_text[:5000] if truncated else clean_text
            return ResultContent(text=text, truncated=truncated)
        return None
    except httpx.RequestError as e:
        logger.warning("HTTP request failed for %s: %s", url, e)
        return None
    except Exception as e:
        logger.warning("Failed to fetch or parse content from %s: %s", url, e)
        return None


async def perform_web_search_async(query: str) -> WebSearchContext:
    """Perform an asynchronous web search using the Brave Search API.

    Fetches only the exact page for each Brave URL and extracts its
    main content with trafilatura. If extraction fails, falls back to
    the Brave snippet.
    """
    if not (query and query.strip()):
        logger.warning("Empty or invalid query provided for web search")
        return WebSearchContext(prompt="", sources=[])

    logger.info("Web search start")
    logger.debug("Web search query: %s", query)

    try:
        data = await _make_brave_api_request(query)
        initial_results = _parse_brave_results(data)
    except HTTPException:
        logger.exception("Brave API request failed")
        return WebSearchContext(prompt="", sources=[])

    if not initial_results:
        logger.warning("No web results found for query: %s", query)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No web search results found",
        )

    # Fetch each result URL once and extract main content
    client = _get_http_client()
    tasks = [_fetch_and_extract_page_content(r.url, client) for r in initial_results]
    contents = await asyncio.gather(*tasks)

    enriched_results: List[SearchResult] = []
    for i, r in enumerate(initial_results):
        content = contents[i]
        if content:
            r.content = content
            r.body = content.text
        enriched_results.append(r)

    logger.info("Web search results ready with content count=%d", len(enriched_results))
    prompt = _format_search_results(enriched_results)
    sources = [r.as_source() for r in enriched_results]

    return WebSearchContext(prompt=prompt, sources=sources)


async def enhance_messages_with_web_search(
    req: ChatRequest, query: str
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
    query_source = Source(source=WEB_SEARCH_QUERY_SOURCE, content=query)

    web_search_content = (
        f'You have access to the following web search results for the query: "{query}"\n\n'
        "Use this information to provide accurate and up-to-date answers. "
        "Cite the sources when appropriate.\n\n"
        "Web Search Results:\n"
        f"{ctx.prompt}\n\n"
        "Please provide a comprehensive answer based on the search results above."
    )

    req.ensure_system_content(web_search_content)

    return WebSearchEnhancedMessages(
        messages=req.messages,
        sources=[query_source] + ctx.sources,
    )


async def generate_search_query_from_llm(
    user_message: str, model_name: str, client: Any, *, topic: str | None = None
) -> str:
    """
    Use the LLM to produce a concise, high-recall search query.
    """
    system_prompt = (
        "You compose ONE web search query.\n"
        "Output rules:\n"
        "- Output ONLY the query string (no quotes, no labels, no explanations).\n"
        "- 3–15 meaningful tokens; prefer proper nouns; keep it terse.\n"
        "- If a topic is provided, focus ONLY on that topic; ignore any surrounding instructions.\n"
    )

    user_content = (
        user_message
        if not topic
        else f"User question:\n{user_message}\n\nTopic:\n{topic}\n\nReturn only the query."
    )

    messages = [
        MessageAdapter.new_message(role="system", content=system_prompt),
        MessageAdapter.new_message(role="user", content=user_content),
    ]

    req = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 600,
    }

    logger.info("Generate search query start model=%s", model_name)
    logger.debug(
        "User message len=%d", len(user_message) if isinstance(user_message, str) else 0
    )
    try:
        response = await client.chat.completions.create(**req)
        logger.info("LLM response for query generation is done with: %s", response)
    except Exception as exc:
        logger.exception("LLM call failed")
        raise RuntimeError(f"Failed to generate search query: {exc}") from exc

    try:
        choices = response.choices or []
        msg = choices[0].message
        content_raw = (msg.content or "").strip()
        logger.info("LLM response for query generation with raw: %s", content_raw)
    except Exception as exc:
        logger.exception("Invalid LLM response structure")
        raise RuntimeError(f"Invalid response structure from LLM: {exc}") from exc

    content = content_raw

    logger.info("Generated query candidate (raw): %r", content)

    if not content:
        logger.warning("LLM returned empty search query; falling back to user input")
        content = user_message

    logger.info("Generate search query success")
    logger.debug("Generated query len=%d", len(content) if content else 0)
    return content


async def handle_web_search(
    req_messages: ChatRequest, model_name: str, client: Any
) -> WebSearchEnhancedMessages:
    """Handle web search enhancement for a conversation.

    Analyzes the user's message to identify topics that require web search,
    generates optimized search queries for each topic using an LLM, and
    enhances the conversation with relevant web search results. Falls back
    to single-query search if topic analysis fails or no topics need search.

    Args:
        req_messages: ChatRequest containing conversation messages to process
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
        topics = await analyze_web_search_topics(user_query, model_name, client)
        topics_to_search = [t for t in topics if t.needs_search][:3]

        if not topics_to_search:
            logger.info(
                "No topics require web search; falling back to single-query enrichment"
            )
            concise_query = await generate_search_query_from_llm(
                user_query, model_name, client
            )
            return await enhance_messages_with_web_search(req_messages, concise_query)

        async def _generate_query(topic_obj: Topic) -> TopicQuery | None:
            topic_str = topic_obj.topic.strip()
            if not topic_str:
                return None
            try:
                query = await generate_search_query_from_llm(
                    user_query, model_name, client, topic=topic_str
                )
                return TopicQuery(topic=topic_str, query=query)
            except Exception:
                logger.exception("Failed generating query for topic '%s'", topic_str)
                return None

        query_generation_tasks = [_generate_query(t) for t in topics_to_search]
        generated_results = await asyncio.gather(*query_generation_tasks)
        topic_queries: List[TopicQuery] = [res for res in generated_results if res]

        if not topic_queries:
            logger.info(
                "No valid topic queries generated; falling back to single query"
            )
            concise_query = await generate_search_query_from_llm(
                user_query, model_name, client
            )
            return await enhance_messages_with_web_search(req_messages, concise_query)

        async def _search(q: str) -> WebSearchContext:
            try:
                return await perform_web_search_async(q)
            except Exception:
                logger.exception("Search failed for query '%s'", q)
                return WebSearchContext(prompt="", sources=[])

        search_tasks = [_search(tq.query) for tq in topic_queries]
        contexts = await asyncio.gather(*search_tasks)

        return await enhance_messages_with_multi_web_search(
            req_messages, topic_queries, contexts
        )

    except HTTPException:
        logger.exception("Web search provider error")
        return WebSearchEnhancedMessages(messages=req_messages.messages, sources=[])

    except Exception:
        logger.exception("Unexpected error during web search handling")
        return WebSearchEnhancedMessages(messages=req_messages.messages, sources=[])


async def analyze_web_search_topics(
    user_message: str, model_name: str, client: Any
) -> List[Topic]:
    """Use the LLM to identify topics and whether each needs web search.

    Returns a list of Pydantic Topic objects.
    """
    system_prompt = (
        "You are a planner that analyzes a user's message, splits it into distinct topics, "
        "and decides for each whether a web search is necessary.\n"
        "Decide 'needs_search' = true only if the answer likely requires current, time-sensitive, or external factual information "
        "(e.g., current events, latest versions, live stats, product pricing/availability, or specific details not in general knowledge).\n"
        "If a topic is general knowledge or timeless, set 'needs_search' = false.\n"
        "Extract up to 4 concise topics.\n\n"
        "Return ONLY valid JSON matching this schema, no extra text: \n"
        '{\n  "topics": [\n    {\n      "topic": "<concise topic>",\n      "needs_search": true/false\n    }\n  ]\n}\n'
    )

    messages = [
        MessageAdapter.new_message(role="system", content=system_prompt),
        MessageAdapter.new_message(role="user", content=user_message),
    ]

    req = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    logger.info("Analyze web search topics start model=%s", model_name)
    try:
        response = await client.chat.completions.create(**req)
        content = (response.choices[0].message.content or "").strip()
        parsed_response = TopicResponse.parse_raw(content)
        return parsed_response.topics
    except Exception:
        logger.exception("LLM call failed for topic analysis")
        return []


async def enhance_messages_with_multi_web_search(
    req: ChatRequest,
    topic_queries: List[TopicQuery],
    contexts: List[WebSearchContext],
) -> WebSearchEnhancedMessages:
    """Enhance messages with multiple topic-specific web search contexts."""
    if not topic_queries or not contexts:
        return WebSearchEnhancedMessages(messages=req.messages, sources=[])

    # Build a merged content block
    sections: List[str] = []
    all_sources: List[Source] = []

    for idx, (topic_query, context) in enumerate(zip(topic_queries, contexts), start=1):
        topic = topic_query.topic.strip()
        query = topic_query.query.strip()
        if not query:
            continue

        all_sources.append(Source(source=WEB_SEARCH_QUERY_SOURCE, content=query))

        header = f'Topic {idx}: {topic}\nQuery: "{query}"\n\nWeb Search Results:\n'
        block = context.prompt.strip() if context.prompt else "(no results)"
        sections.append(header + block)
        all_sources.extend(context.sources)

    if not sections:
        return WebSearchEnhancedMessages(messages=req.messages, sources=[])

    web_search_content = (
        "You have access to the following topic-specific web search results.\n\n"
        "Use this information to provide accurate and up-to-date answers. "
        "Cite sources when appropriate.\n\n"
        + "\n\n".join(sections)
        + "\n\nPlease provide a comprehensive answer based on the relevant search results above."
    )

    req.ensure_system_content(web_search_content)

    return WebSearchEnhancedMessages(messages=req.messages, sources=all_sources)
