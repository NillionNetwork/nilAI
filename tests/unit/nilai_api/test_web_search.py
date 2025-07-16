import pytest
from unittest.mock import patch, MagicMock
from nilai_api.handlers.web_search import (
    perform_web_search_sync,
    enhance_messages_with_web_search,
    handle_web_search,
)
from nilai_common import Message


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

        results, sources = perform_web_search_sync("AI developments")

        assert len(results) == 2
        assert "Latest AI Developments" in results[0]
        assert "AI Breakthrough in Robotics" in results[1]
        assert "GPT-5" in results[0]
        assert "40%" in results[1]
        assert sources[0]["title"] == "Latest AI Developments"
        assert sources[1]["title"] == "AI Breakthrough in Robotics"


def test_perform_web_search_sync_no_results():
    """Test web search with no results"""
    with patch("nilai_api.handlers.web_search.DDGS") as mock_ddgs:
        mock_instance = MagicMock()
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        mock_instance.text.return_value = []
        mock_instance.news.return_value = []

        results, sources = perform_web_search_sync("nonexistent query")

        assert results == []
        assert sources == []


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

        results, sources = perform_web_search_sync("AI news")

        assert results == []
        assert sources == []


@pytest.mark.asyncio
async def test_enhance_messages_with_web_search():
    """Test message enhancement with web search results"""
    original_messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What is the latest AI news?"),
    ]

    with patch("nilai_api.handlers.web_search.perform_web_search_sync") as mock_search:
        mock_search.return_value = (
            [
                "Latest AI Developments: OpenAI announces GPT-5",
                "AI Breakthrough: New neural network improves efficiency by 40%",
            ],
            [],
        )

        enhanced_messages, sources = await enhance_messages_with_web_search(
            original_messages, "AI news"
        )

        assert len(enhanced_messages) == 3
        assert enhanced_messages[0].role == "system"
        assert enhanced_messages[0].content == "You are a helpful assistant"
        assert enhanced_messages[1].role == "system"
        assert "Latest AI Developments" in enhanced_messages[1].content
        assert enhanced_messages[2].role == "user"
        assert enhanced_messages[2].content == "What is the latest AI news?"
        assert sources == []


@pytest.mark.asyncio
async def test_handle_web_search():
    """Test web search handler with user messages"""
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="Tell me about current events"),
    ]

    with patch(
        "nilai_api.handlers.web_search.enhance_messages_with_web_search"
    ) as mock_enhance:
        mock_enhance.return_value = (
            messages + [Message(role="system", content="Enhanced context")],
            [],
        )

        result, sources = await handle_web_search(messages)

        mock_enhance.assert_called_once_with(messages, "Tell me about current events")
        assert len(result) == 3
        assert sources == []


@pytest.mark.asyncio
async def test_handle_web_search_no_user_message():
    """Test web search handler with no user message"""
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="assistant", content="Hello! How can I help you?"),
    ]

    result, sources = await handle_web_search(messages)

    assert result == messages
    assert sources == []


@pytest.mark.asyncio
async def test_handle_web_search_exception_handling():
    """Test web search handler exception handling"""
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What's the weather like?"),
    ]

    with patch(
        "nilai_api.handlers.web_search.enhance_messages_with_web_search"
    ) as mock_enhance:
        mock_enhance.side_effect = Exception("Search service unavailable")

        result, sources = await handle_web_search(messages)

        assert result == messages
        assert sources == []
