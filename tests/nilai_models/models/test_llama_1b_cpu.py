from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nilai_common import ChatRequest, ChatResponse, Message
from nilai_models.models.llama_1b_cpu.llama_1b_cpu import Llama1BCpu

from tests import response as RESPONSE


@pytest.fixture
def llama_model(mocker):
    """Fixture to provide a Llama1BCpu instance for testing."""
    model = Llama1BCpu()
    mocker.patch.object(model, "chat_completion", new_callable=AsyncMock)
    return model


@pytest.mark.asyncio
async def test_chat_completion_valid_request(llama_model):
    """Test chat completion with a valid request."""
    req = ChatRequest(
        model="bartowski/Llama-3.2-1B-Instruct-GGUF",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is your name?"),
        ],
    )
    llama_model.chat_completion.return_value = RESPONSE
    response = await llama_model.chat_completion(req)
    assert isinstance(response, ChatResponse)
    assert response.model == "test-model"
    assert response.choices is not None


@pytest.mark.asyncio
async def test_chat_completion_empty_messages(llama_model):
    """Test chat completion with an empty messages list."""
    req = ChatRequest(
        model="bartowski/Llama-3.2-1B-Instruct-GGUF",
        messages=[],
    )
    llama_model.chat_completion.side_effect = HTTPException(
        status_code=400, detail="The 'messages' field is required."
    )
    with pytest.raises(HTTPException) as exc_info:
        await llama_model.chat_completion(req)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "The 'messages' field is required."


@pytest.mark.asyncio
async def test_chat_completion_missing_model(llama_model):
    """Test chat completion with a missing model field."""
    req = ChatRequest(
        model="",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is your name?"),
        ],
    )
    llama_model.chat_completion.side_effect = HTTPException(
        status_code=400, detail="The 'model' field is required."
    )
    with pytest.raises(HTTPException) as exc_info:
        await llama_model.chat_completion(req)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "The 'model' field is required."
