import pytest
from unittest.mock import patch, MagicMock
from nilai_api.handlers.web_search import (
    perform_web_search_sync,
    get_web_search_context,
    enhance_messages_with_web_search,
    handle_web_search,
)
from nilai_common import Message
from nilai_common.api_model import WebSearchContext, WebSearchEnhancedMessages


def test_perform_web_search_sync_success():
    """Test successful web search with mock response"""
    mock_search_results = [
        {
            "title": "Latest AI Developments",
            "body": "OpenAI announces GPT-5 with improved capabilities and better performance across various tasks.",
            "href": "https://example.com/ai1",
        },
        {
            "title": "AI Breakthrough in Robotics",
            "body": "New neural network architecture improves robot learning efficiency by 40% in recent studies.",
            "href": "https://example.com/ai2",
        },
    ]

    with patch("nilai_api.handlers.web_search.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = mock_search_results
        mock_instance.news.return_value = []

        ctx = perform_web_search_sync("AI developments")

        assert len(ctx.sources) == 2
        assert "GPT-5" in ctx.prompt
        assert "40%" in ctx.prompt
        assert ctx.sources[0].source == "https://example.com/ai1"
        assert ctx.sources[1].source == "https://example.com/ai2"


def test_perform_web_search_sync_no_results():
    """Test web search with no results"""
    with patch("nilai_api.handlers.web_search.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = []
        mock_instance.news.return_value = []

        with pytest.raises(Exception):
            _ = perform_web_search_sync("nonexistent query")


def test_perform_web_search_sync_fallback_to_news():
    """Test web search fallback to news when text search returns no results"""
    mock_news_results = [
        {
            "title": "Breaking AI News",
            "body": "Major breakthrough in artificial intelligence research announced today.",
            "href": "https://example.com/news1",
        }
    ]

    with patch("nilai_api.handlers.web_search.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = []
        mock_instance.news.return_value = mock_news_results

        with pytest.raises(Exception):
            _ = perform_web_search_sync("AI news")


@pytest.mark.asyncio
async def test_enhance_messages_with_web_search():
    """Test message enhancement with web search results"""
    original_messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What is the latest AI news?"),
    ]

    with patch("nilai_api.handlers.web_search.perform_web_search_sync") as mock_search:
        mock_search.return_value = WebSearchContext(
            prompt="Latest AI Developments: OpenAI announces GPT-5\nAI Breakthrough: New neural network improves efficiency by 40%",
            sources=[],
        )

        enhanced = await enhance_messages_with_web_search(original_messages, "AI news")

        assert len(enhanced.messages) == 3
        assert enhanced.messages[0].role == "system"
        assert "Latest AI Developments" in str(enhanced.messages[0].content)
        assert enhanced.sources == []


@pytest.mark.asyncio
async def test_handle_web_search():
    """Test web search handler with user messages"""
    messages = [
        Message(role="user", content="Tell me about current events"),
    ]
    with (
        patch(
            "nilai_api.handlers.web_search.enhance_messages_with_web_search"
        ) as mock_enhance,
        patch(
            "nilai_api.handlers.web_search.generate_search_query_from_llm"
        ) as mock_generate_query,
    ):
        mock_enhance.return_value = WebSearchEnhancedMessages(
            messages=[Message(role="system", content="Enhanced context")] + messages,
            sources=[],
        )
        mock_generate_query.return_value = "Tell me about current events"
        dummy_client = MagicMock()
        enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
        mock_generate_query.assert_called_once()
        mock_enhance.assert_called_once_with(messages, "Tell me about current events")
        assert len(enhanced.messages) == len(messages) + 1
        assert enhanced.sources == []


@pytest.mark.asyncio
async def test_handle_web_search_no_user_message():
    """Test web search handler with no user message"""
    messages = [
        Message(role="assistant", content="Hello! How can I help you?"),
    ]
    dummy_client = MagicMock()
    enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
    assert enhanced.messages == messages
    assert enhanced.sources == []


@pytest.mark.asyncio
async def test_handle_web_search_exception_handling():
    """Test web search handler exception handling"""
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What's the weather like?"),
    ]
    with (
        patch(
            "nilai_api.handlers.web_search.enhance_messages_with_web_search"
        ) as mock_enhance,
        patch(
            "nilai_api.handlers.web_search.generate_search_query_from_llm"
        ) as mock_generate_query,
    ):
        mock_enhance.side_effect = Exception("Search service unavailable")
        mock_generate_query.return_value = "What's the weather like?"
        dummy_client = MagicMock()
        enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
        mock_generate_query.assert_called_once()
        assert enhanced.messages == messages
        assert enhanced.sources == []


@pytest.mark.asyncio
async def test_get_web_search_context_async_wrapper():
    with patch("nilai_api.handlers.web_search.perform_web_search_sync") as mock_sync:
        mock_sync.return_value = WebSearchContext(prompt="info", sources=[])
        ctx = await get_web_search_context("query")
        assert ctx.prompt == "info"
        mock_sync.assert_called_once()
