import pytest
from unittest.mock import patch, AsyncMock, Mock
from fastapi import HTTPException
from nilai_api.handlers.web_search import (
    perform_web_search_async,
    enhance_messages_with_web_search,
    _make_brave_api_request,
    _get_brave_rate_limiter,
    _send_brave_api_request,
)
from nilai_common import MessageAdapter, ChatRequest
from nilai_common.api_models import (
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
        patch("nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"),
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
        patch("nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"),
        patch(
            "nilai_api.handlers.web_search._make_brave_api_request",
            side_effect=[mock_data_1, mock_data_2],
        ),
    ):
        import asyncio

        results = await asyncio.gather(
            perform_web_search_async("AI news"),
            perform_web_search_async("Machine learning"),
        )
        assert len(results) == 2
        assert results[0].sources is not None
        assert len(results[0].sources) == 1
        assert results[0].sources[0].source == "https://example.com/ai-news"
        assert (
            results[0].sources[0].content
            == "Latest developments in artificial intelligence."
        )
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
    req = ChatRequest(model="dummy", messages=original_messages)

    with patch("nilai_api.handlers.web_search.perform_web_search_async") as mock_search:
        mock_search.return_value = WebSearchContext(
            prompt="[1] Latest AI Developments\nURL: https://example.com\nSnippet: OpenAI announces GPT-5",
            sources=[
                Source(source="https://example.com", content="OpenAI announces GPT-5")
            ],
        )

        enhanced = await enhance_messages_with_web_search(req, "AI news")

        assert len(enhanced.messages) == 2
        assert enhanced.messages[0]["role"] == "system"
        assert "Latest AI Developments" in str(enhanced.messages[0]["content"])
        assert enhanced.sources is not None
        assert len(enhanced.sources) == 2
        assert enhanced.sources[0].source == "web_search_query"
        assert enhanced.sources[0].content == "AI news"
        assert enhanced.sources[1].source == "https://example.com"
        assert enhanced.sources[1].content == "OpenAI announces GPT-5"


@pytest.mark.asyncio
async def test_make_brave_api_request_respects_config_rps():
    _get_brave_rate_limiter.cache_clear()
    try:
        limiter_mock = Mock()
        limiter_mock.has_capacity.return_value = True
        limiter_ctx = AsyncMock()
        limiter_mock.__aenter__ = limiter_ctx.__aenter__
        limiter_mock.__aexit__ = limiter_ctx.__aexit__
        send_mock = AsyncMock(return_value={"status": "ok"})
        with (
            patch(
                "nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"
            ),
            patch("nilai_api.handlers.web_search.CONFIG.web_search.rps", 3),
            patch(
                "nilai_api.handlers.web_search.AsyncLimiter",
                return_value=limiter_mock,
            ) as limiter_cls,
            patch("nilai_api.handlers.web_search._send_brave_api_request", send_mock),
        ):
            result1 = await _make_brave_api_request(" sample ")
            result2 = await _make_brave_api_request("sample")
    finally:
        _get_brave_rate_limiter.cache_clear()
    assert result1 == {"status": "ok"}
    assert result2 == {"status": "ok"}
    limiter_cls.assert_called_once_with(max_rate=3, time_period=1.0)
    assert limiter_mock.has_capacity.call_count == 2
    assert send_mock.await_count == 2


@pytest.mark.asyncio
async def test_make_brave_api_request_rate_limit_rejection():
    _get_brave_rate_limiter.cache_clear()
    try:
        limiter_mock = Mock()
        limiter_mock.has_capacity.return_value = False
        limiter_ctx = AsyncMock()
        limiter_mock.__aenter__ = limiter_ctx.__aenter__
        limiter_mock.__aexit__ = limiter_ctx.__aexit__
        send_mock = AsyncMock()
        with (
            patch(
                "nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"
            ),
            patch("nilai_api.handlers.web_search.CONFIG.web_search.rps", 2),
            patch(
                "nilai_api.handlers.web_search.AsyncLimiter",
                return_value=limiter_mock,
            ),
            patch("nilai_api.handlers.web_search._send_brave_api_request", send_mock),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _make_brave_api_request("sample")
    finally:
        _get_brave_rate_limiter.cache_clear()
    assert exc_info.value.status_code == 429
    assert "2 requests/second" in exc_info.value.detail
    assert send_mock.await_count == 0


@pytest.mark.asyncio
async def test_make_brave_api_request_no_limiter_when_rps_disabled():
    _get_brave_rate_limiter.cache_clear()
    try:
        send_mock = AsyncMock(return_value={"status": "ok"})
        with (
            patch(
                "nilai_api.handlers.web_search.CONFIG.web_search.api_key", "test-key"
            ),
            patch("nilai_api.handlers.web_search.CONFIG.web_search.rps", 0),
            patch("nilai_api.handlers.web_search._send_brave_api_request", send_mock),
            patch("nilai_api.handlers.web_search.AsyncLimiter") as limiter_cls,
        ):
            result = await _make_brave_api_request("q")
    finally:
        _get_brave_rate_limiter.cache_clear()
    assert result == {"status": "ok"}
    limiter_cls.assert_not_called()
    assert send_mock.await_count == 1


@pytest.mark.asyncio
async def test_make_brave_api_request_missing_api_key():
    _get_brave_rate_limiter.cache_clear()
    try:
        with (
            patch("nilai_api.handlers.web_search.CONFIG.web_search.api_key", None),
            patch("nilai_api.handlers.web_search.CONFIG.web_search.rps", 3),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _make_brave_api_request("q")
    finally:
        _get_brave_rate_limiter.cache_clear()
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_send_brave_api_request_handles_provider_429():
    client = AsyncMock()

    class Resp429:
        status_code = 429
        text = "Too Many Requests"

        def json(self):
            return {}

    client.get.return_value = Resp429()
    with (
        patch("nilai_api.handlers.web_search.CONFIG.web_search.api_key", "k"),
        patch("nilai_api.handlers.web_search.CONFIG.web_search.api_path", "http://x"),
        patch("nilai_api.handlers.web_search._get_http_client", return_value=client),
    ):
        with pytest.raises(HTTPException) as exc:
            await _send_brave_api_request("q")
    assert exc.value.status_code == 429
