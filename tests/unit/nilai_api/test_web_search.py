import pytest
from unittest.mock import patch
from fastapi import HTTPException
from nilai_api.handlers.web_search import (
    perform_web_search_async,
    enhance_messages_with_web_search,
)
from nilai_common import MessageAdapter
from nilai_common.api_model import (
    WebSearchContext,
    Source,
)


@pytest.mark.asyncio
async def test_perform_web_search_async_success():
    """Test successful web search with proper source validation"""
    mock_data = {
        "web": {
            "results": [
                {
                    "title": "Latest AI Developments",
                    "description": "OpenAI announces GPT-5 with improved capabilities.",
                    "url": "https://example.com/ai1",
                },
                {
                    "title": "AI Breakthrough in Robotics",
                    "description": "New neural network architecture improves robot learning.",
                    "url": "https://example.com/ai2",
                },
            ]
        }
    }

    with (
        patch("nilai_api.config.CONFIG.web_search.api_key", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            return_value=mock_data,
        ),
    ):
        ctx = await perform_web_search_async("AI developments")

        assert ctx.sources is not None
        assert len(ctx.sources) == 2
        assert ctx.sources[0].source == "https://example.com/ai1"
        assert (
            ctx.sources[0].content
            == "OpenAI announces GPT-5 with improved capabilities."
        )
        assert ctx.sources[1].source == "https://example.com/ai2"
        assert (
            ctx.sources[1].content
            == "New neural network architecture improves robot learning."
        )


@pytest.mark.asyncio
async def test_perform_web_search_async_no_results():
    """Test web search with no results returns 404"""
    mock_data = {"web": {"results": []}}

    with (
        patch("nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            return_value=mock_data,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await perform_web_search_async("nonexistent query")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_perform_web_search_async_concurrent_queries():
    """Test multiple concurrent web search queries"""
    mock_data_1 = {
        "web": {
            "results": [
                {
                    "title": "AI News",
                    "description": "Latest developments in artificial intelligence.",
                    "url": "https://example.com/ai-news",
                }
            ]
        }
    }

    mock_data_2 = {
        "web": {
            "results": [
                {
                    "title": "Machine Learning",
                    "description": "Advances in machine learning algorithms.",
                    "url": "https://example.com/ml",
                }
            ]
        }
    }

    with (
        patch("nilai_api.config.CONFIG.web_search.api_key", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            side_effect=[mock_data_1, mock_data_2],
        ),
    ):
        import asyncio

        # Run two concurrent web searches
        results = await asyncio.gather(
            perform_web_search_async("AI news"),
            perform_web_search_async("Machine learning"),
        )

        # Verify both searches completed successfully
        assert len(results) == 2

        # Check first result
        assert results[0].sources is not None
        assert len(results[0].sources) == 1
        assert results[0].sources[0].source == "https://example.com/ai-news"
        assert (
            results[0].sources[0].content
            == "Latest developments in artificial intelligence."
        )

        # Check second result
        assert results[1].sources is not None
        assert len(results[1].sources) == 1
        assert results[1].sources[0].source == "https://example.com/ml"
        assert (
            results[1].sources[0].content == "Advances in machine learning algorithms."
        )


@pytest.mark.asyncio
async def test_enhance_messages_with_web_search():
    """Test message enhancement with web search results and source validation"""
    original_messages = [
        MessageAdapter.new_message(
            role="system", content="You are a helpful assistant"
        ),
        MessageAdapter.new_message(role="user", content="What is the latest AI news?"),
    ]

    with patch("nilai_api.handlers.web_search.perform_web_search_async") as mock_search:
        mock_search.return_value = WebSearchContext(
            prompt="[1] Latest AI Developments\nURL: https://example.com\nSnippet: OpenAI announces GPT-5",
            sources=[
                Source(source="https://example.com", content="OpenAI announces GPT-5")
            ],
        )

        enhanced = await enhance_messages_with_web_search(original_messages, "AI news")

        assert len(enhanced.messages) == 3
        assert enhanced.messages[0]["role"] == "system"
        assert "Latest AI Developments" in str(enhanced.messages[0]["content"])
        assert enhanced.sources is not None
        assert len(enhanced.sources) == 2
        assert enhanced.sources[0].source == "web_search_query"
        assert enhanced.sources[0].content == "AI news"
        assert enhanced.sources[1].source == "https://example.com"
        assert enhanced.sources[1].content == "OpenAI announces GPT-5"
