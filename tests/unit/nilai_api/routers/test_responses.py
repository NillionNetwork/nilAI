import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from nilai_api.routers.responses import response_completion
from nilai_common import ResponseRequest, MessageAdapter


@pytest.mark.asyncio
async def test_response_to_chat_request_string_input():
    req = ResponseRequest(
        model="test-model",
        input="Hello",
        instructions="You are a helpful assistant.",
        temperature=0.5,
        max_tokens=100,
    )
    
    chat_req = req.to_chat_request()
    
    assert chat_req.model == "test-model"
    assert len(chat_req.messages) == 2
    assert chat_req.messages[0]["role"] == "system"
    assert chat_req.messages[0]["content"] == "You are a helpful assistant."
    assert chat_req.messages[1]["role"] == "user"
    assert chat_req.messages[1]["content"] == "Hello"
    assert chat_req.temperature == 0.5
    assert chat_req.max_tokens == 100


@pytest.mark.asyncio
async def test_response_to_chat_request_messages_input():
    req = ResponseRequest(
        model="test-model",
        input=[
            {"role": "user", "content": "Hello"}
        ],
        instructions="You are helpful.",
        temperature=0.7,
    )
    
    chat_req = req.to_chat_request()
    
    assert len(chat_req.messages) == 2
    assert chat_req.messages[0]["role"] == "system"
    assert chat_req.messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_response_to_chat_request_no_instructions():
    req = ResponseRequest(
        model="test-model",
        input="Hello",
    )
    
    chat_req = req.to_chat_request()
    
    assert len(chat_req.messages) == 1
    assert chat_req.messages[0]["role"] == "user"
    assert chat_req.messages[0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_response_has_multimodal_content_string():
    req = ResponseRequest(
        model="test-model",
        input="Hello",
    )
    
    assert req.has_multimodal_content() == False


@pytest.mark.asyncio
async def test_response_has_multimodal_content_with_image():
    req = ResponseRequest(
        model="test-model",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                ],
            }
        ],
    )
    
    assert req.has_multimodal_content() == True


@pytest.mark.asyncio
async def test_response_max_output_tokens_alias():
    req = ResponseRequest(
        model="test-model",
        input="Hello",
        max_output_tokens=150,
    )
    
    assert req.max_tokens == 150
    
    chat_req = req.to_chat_request()
    assert chat_req.max_tokens == 150

