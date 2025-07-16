import asyncio
import logging
from typing import List, Optional, Tuple, Dict
from duckduckgo_search import DDGS
from nilai_common import Message

logger = logging.getLogger(__name__)


async def enhance_messages_with_web_search(
    messages: List[Message], query: str
) -> Tuple[List[Message], List[Dict[str, str]]]:
    """
    Enhance the conversation context with web search results.

    Args:
        messages: Original conversation messages
        query: The search query to perform

    Returns:
        Tuple of (enhanced messages with web search context, list of sources used)
    """
    if not query:
        return messages, []

    loop = asyncio.get_event_loop()
    search_results, sources = await loop.run_in_executor(
        None, lambda: perform_web_search_sync(query)
    )

    if not search_results:
        return messages, []

    web_context = "Based on recent web search results:\n" + "\n".join(search_results)

    user_query = ""
    for message in reversed(messages):
        if message.role == "user":
            user_query = message.content
            break

    enhanced_system_message = Message(
        role="system",
        content=f"You have access to the following current information from web search:\n{web_context}\n\nPlease use this information to provide accurate and up-to-date responses to the user's query: {user_query}\n\nYou must add between brackets the source of the information you are using to answer the user's query.",
    )

    enhanced_messages = []
    system_messages_added = False

    for message in messages:
        enhanced_messages.append(message)
        if message.role == "system" and not system_messages_added:
            enhanced_messages.append(enhanced_system_message)
            system_messages_added = True

    if not system_messages_added:
        enhanced_messages.insert(0, enhanced_system_message)

    return enhanced_messages, sources


def perform_web_search_sync(query: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Synchronous version of web search for use with run_in_executor.

    Args:
        query: The search query to perform

    Returns:
        Tuple of (list of search result snippets, list of sources with metadata)
    """
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=3, region="us-en"))

            results = []
            sources = []

            for result in search_results:
                if result.get("title") and result.get("body"):
                    title = result["title"]
                    body = result["body"][:500]
                    results.append(f"{title}: {body}")

                    source_info = {
                        "title": title,
                        "url": result["href"],
                        "snippet": body,
                        "type": "web_search",
                    }
                    sources.append(source_info)

            return results, sources

    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return [], []


async def handle_web_search(
    req_messages: List[Message],
) -> Tuple[List[Message], List[Dict[str, str]]]:
    """
    Handle web search functionality for chat requests.

    Args:
        req_messages: Original request messages

    Returns:
        Tuple of (enhanced messages with web search context if enabled, list of sources used)
    """
    user_query = ""
    for message in reversed(req_messages):
        if message.role == "user":
            user_query = message.content
            break

    if not user_query:
        return req_messages, []

    try:
        enhanced_messages, sources = await enhance_messages_with_web_search(
            req_messages, user_query
        )
        return enhanced_messages, sources
    except Exception as e:
        logger.error(f"Error in web search handler: {e}")
        return req_messages, []
