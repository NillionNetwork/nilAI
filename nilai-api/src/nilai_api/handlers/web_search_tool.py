from __future__ import annotations

import logging
from typing import Any

from nilai_common import ChatRequest, WebSearchToolOutcome

from .web_search import handle_web_search


async def handle_web_search_tool_call(
    req: ChatRequest,
    model_name: str,
    client: Any,
    logger: logging.Logger,
) -> WebSearchToolOutcome:
    """Leverage the same pipeline as the `web_search=True` flag via tool call.

    - Runs the full web search pipeline (topic analysis, query generation, multi-search).
    - Injects system content into `req.messages` using existing helpers.
    - Returns the injected content as tool output, with sources for traceability.
    """
    result = await handle_web_search(req, model_name, client)

    # Compose tool content using the exact system content that was injected.
    tool_content = result.system_content or "(web_search) Context injected."

    logger.info(
        "[chat] web_search tool completed sources=%d", len(result.sources or [])
    )
    return WebSearchToolOutcome(content=tool_content, sources=result.sources)

