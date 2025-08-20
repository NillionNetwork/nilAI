import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException
from nilai_api.handlers.web_search import (
    perform_web_search_async,
    enhance_messages_with_web_search,
    handle_web_search,
    generate_search_query_from_llm,
    _parse_brave_results,
    _make_brave_api_request,
)
from nilai_common import Message
from nilai_common.api_model import (
    WebSearchContext,
    WebSearchEnhancedMessages,
    Source,
)


@pytest.mark.asyncio
async def test_make_brave_api_request_success():
    """Test successful Brave API request"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"web": {"results": []}}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with (
        patch("nilai_api.handlers.web_search.BRAVE_SEARCH_API", "test-key"),
        patch(
            "nilai_api.handlers.web_search.brave_rate_limiter.acquire",
            new_callable=AsyncMock,
        ),
        patch(
            "nilai_api.handlers.web_search._get_http_client", return_value=mock_client
        ),
    ):
        result = await _make_brave_api_request("test query")

        assert result == {"web": {"results": []}}
        mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_make_brave_api_request_server_error():
    """Test server error handling"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Server Error"

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response

    with (
        patch("nilai_api.handlers.web_search.BRAVE_SEARCH_API", "test-key"),
        patch(
            "nilai_api.handlers.web_search.brave_rate_limiter.acquire",
            new_callable=AsyncMock,
        ),
        patch(
            "nilai_api.handlers.web_search._get_http_client", return_value=mock_client
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _make_brave_api_request("test query")

    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_perform_web_search_async_success():
    """Test successful web search with mock response"""
    mock_data = {
        "web": {
            "results": [
                {
                    "title": "Latest AI Developments",
                    "description": "OpenAI announces GPT-5 with improved capabilities and better performance across various tasks.",
                    "url": "https://example.com/ai1",
                },
                {
                    "title": "AI Breakthrough in Robotics",
                    "description": "New neural network architecture improves robot learning efficiency by 40% in recent studies.",
                    "url": "https://example.com/ai2",
                },
            ]
        }
    }

    with (
        patch("nilai_api.handlers.web_search.BRAVE_SEARCH_API", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            return_value=mock_data,
        ),
    ):
        ctx = await perform_web_search_async("AI developments")

        assert ctx.sources is not None
        assert len(ctx.sources) > 0
        assert "GPT-5" in ctx.prompt
        assert "40%" in ctx.prompt
        assert "[1]" in ctx.prompt
        assert "[2]" in ctx.prompt


@pytest.mark.asyncio
async def test_perform_web_search_async_no_results():
    """Test web search with no results returns 404"""
    mock_data = {"web": {"results": []}}

    with (
        patch("nilai_api.handlers.web_search.BRAVE_SEARCH_API", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            return_value=mock_data,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await perform_web_search_async("nonexistent query")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_perform_web_search_async_empty_query():
    """Test web search with empty query"""
    with pytest.raises(HTTPException) as exc_info:
        await perform_web_search_async("")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_perform_web_search_async_no_api_key():
    """Test web search without API key"""
    with patch("nilai_api.handlers.web_search.BRAVE_SEARCH_API", None):
        with pytest.raises(HTTPException) as exc_info:
            await perform_web_search_async("test query")

        assert exc_info.value.status_code == 500


def test_parse_brave_results():
    """Test parsing of Brave API results"""
    mock_data = {
        "web": {
            "results": [
                {
                    "title": "Test Title",
                    "description": "Test description",
                    "url": "https://example.com",
                },
                {
                    "title": "Test Title 2",
                    "snippet": "Test snippet",
                    "link": "https://example2.com",
                },
                {
                    "title": "Test Title 3",
                    "body": "Test body",
                    "href": "https://example3.com",
                },
                {
                    "title": "",
                    "description": "No title",
                    "url": "https://example4.com",
                },
                {
                    "title": "No URL",
                    "description": "No URL provided",
                },
            ]
        }
    }

    results = _parse_brave_results(mock_data)

    assert len(results) == 3
    assert results[0].title == "Test Title"
    assert results[0].body == "Test description"
    assert results[0].url == "https://example.com"
    assert results[1].title == "Test Title 2"
    assert results[1].body == "Test snippet"
    assert results[1].url == "https://example2.com"
    assert results[2].title == "Test Title 3"
    assert results[2].body == "Test body"
    assert results[2].url == "https://example3.com"


@pytest.mark.asyncio
async def test_enhance_messages_with_web_search():
    """Test message enhancement with web search results"""
    original_messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What is the latest AI news?"),
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
        assert enhanced.messages[0].role == "system"
        assert "Latest AI Developments" in str(enhanced.messages[0].content)
        assert enhanced.sources is not None
        assert len(enhanced.sources) > 0


@pytest.mark.asyncio
async def test_generate_search_query_from_llm_success():
    """Test successful LLM query generation"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "  generated search query  "

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("nilai_api.handlers.web_search.logger") as mock_logger:
        result = await generate_search_query_from_llm(
            "test message", "test-model", mock_client
        )

        assert result == "generated search query"
        mock_logger.debug.assert_called_once()


@pytest.mark.asyncio
async def test_generate_search_query_from_llm_empty_response():
    """Test LLM query generation with empty response"""
    mock_response = MagicMock()
    mock_response.choices = []

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(RuntimeError, match="LLM returned an empty search query"):
        await generate_search_query_from_llm("test message", "test-model", mock_client)


@pytest.mark.asyncio
async def test_generate_search_query_from_llm_exception():
    """Test LLM query generation with exception"""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    with pytest.raises(RuntimeError, match="Failed to generate search query"):
        await generate_search_query_from_llm("test message", "test-model", mock_client)


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
        expected_sources = [
            Source(source="search_query", content="Tell me about current events"),
            Source(source="https://example.com/news", content="Current events summary"),
        ]
        mock_enhance.return_value = WebSearchEnhancedMessages(
            messages=[Message(role="system", content="Enhanced context")] + messages,
            sources=expected_sources,
        )
        mock_generate_query.return_value = "Tell me about current events"
        dummy_client = MagicMock()
        enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
        mock_generate_query.assert_called_once()
        mock_enhance.assert_called_once_with(messages, "Tell me about current events")
        assert len(enhanced.messages) == len(messages) + 1
        assert enhanced.sources is not None
        assert len(enhanced.sources) > 0


@pytest.mark.asyncio
async def test_handle_web_search_no_user_message():
    """Test web search handler with no user message"""
    messages = [
        Message(role="assistant", content="Hello! How can I help you?"),
    ]
    dummy_client = MagicMock()
    enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
    assert enhanced.messages == messages
    assert enhanced.sources is not None
    assert len(enhanced.sources) == 0


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
        patch("nilai_api.handlers.web_search.logger") as mock_logger,
    ):
        mock_enhance.side_effect = Exception("Search service unavailable")
        mock_generate_query.return_value = "What's the weather like?"
        dummy_client = MagicMock()
        enhanced = await handle_web_search(messages, "dummy-model", dummy_client)
        mock_generate_query.assert_called_once()
        mock_logger.warning.assert_called_once()
        assert enhanced.messages == messages
        assert enhanced.sources is not None
        assert len(enhanced.sources) == 0
