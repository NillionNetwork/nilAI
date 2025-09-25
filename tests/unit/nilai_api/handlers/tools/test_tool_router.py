import json
from typing import Any, Dict, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ChatToolFunction,
)

from nilai_common import MessageAdapter, Usage, ChatRequest
from nilai_api.handlers.tools import tool_router


@pytest.mark.asyncio
async def test_route_and_execute_tool_call_invokes_code_execution(mocker):
    # Arrange: tool call to execute Python code
    tc = ChatCompletionMessageToolCall(
        id="call_123",
        type="function",
        function=ChatToolFunction(
            name="execute_python", arguments=json.dumps({"code": "print(6*7)"})
        ),
    )

    # Patch the async code execution to return a computed value
    mock_exec = mocker.patch(
        "nilai_api.handlers.tools.code_execution.execute_python",
        new_callable=AsyncMock,
        return_value="42",
    )

    # Act
    tool_msg = await tool_router.route_and_execute_tool_call(tc)

    # Assert
    mock_exec.assert_awaited_once_with("print(6*7)")
    assert tool_msg["role"] == "tool"
    # Cast to a plain dict to access optional fields not present
    # on the strict ChatCompletionToolMessageParam TypedDict
    tool_msg_dict = cast(Dict[str, Any], tool_msg)
    assert tool_msg_dict["name"] == "execute_python"
    assert tool_msg_dict["tool_call_id"] == "call_123"
    assert tool_msg_dict["content"] == "42"


def make_completion_message_with_tool_call(code: str) -> ChatCompletion:
    # Helper to create a ChatCompletion whose message requests a tool call
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_abc",
                type="function",
                function=ChatToolFunction(
                    name="execute_python", arguments=json.dumps({"code": code})
                ),
            )
        ],
    )
    return ChatCompletion(
        id="cmpl_tool_1",
        object="chat.completion",
        model="meta-llama/Llama-3.2-1B-Instruct",
        created=123456,
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.asyncio
async def test_handle_tool_workflow_executes_and_uses_result(mocker):
    # Arrange: chat request declaring execute_python tool
    req = ChatRequest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            MessageAdapter.new_message(
                role="system",
                content=(
                    "You can use a code execution tool to perform calculations "
                    "accurately. If arithmetic is needed, call the tool."
                ),
            ),
            MessageAdapter.new_message(role="user", content="What is 6*7?"),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute small Python code snippets and return their output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to run",
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
    )

    # First completion triggers a tool call
    first_response = make_completion_message_with_tool_call("print(6*7)")

    # Patch code execution to return the correct value
    mock_exec = mocker.patch(
        "nilai_api.handlers.tools.code_execution.execute_python",
        new_callable=AsyncMock,
        return_value="42",
    )

    # Stub AsyncOpenAI client: the follow-up completion should use the tool output
    final_message = ChatCompletionMessage(role="assistant", content="The answer is 42.")
    final_completion = ChatCompletion(
        id="cmpl_final_1",
        object="chat.completion",
        model=req.model,
        created=123457,
        choices=[Choice(index=0, message=final_message, finish_reason="stop")],
        usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
    )

    mock_chat_completions = MagicMock()
    mock_chat_completions.create = AsyncMock(return_value=final_completion)
    mock_chat = MagicMock()
    mock_chat.completions = mock_chat_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    # Act
    second, prompt_tokens, completion_tokens = await tool_router.handle_tool_workflow(
        mock_client, req, list(req.messages), first_response
    )

    # Assert: code execution used and final answer returned
    mock_exec.assert_awaited_once_with("print(6*7)")
    assert second.choices[0].message.content == "The answer is 42."
    # Aggregated usage is the sum of both calls
    assert first_response.usage is not None
    assert final_completion.usage is not None
    expected_prompt = (
        first_response.usage.prompt_tokens + final_completion.usage.prompt_tokens
    )
    expected_completion = (
        first_response.usage.completion_tokens
        + final_completion.usage.completion_tokens
    )
    assert prompt_tokens == expected_prompt
    assert completion_tokens == expected_completion


def test_extract_tool_calls_from_json_content():
    # Arrange: assistant content encodes a function call in JSON
    content = json.dumps(
        {
            "function": {
                "name": "execute_python",
                "parameters": {"code": "print(2+3)"},
            }
        }
    )
    msg = ChatCompletionMessage(role="assistant", content=content)

    # Act
    tool_calls = tool_router.extract_tool_calls_from_response_message(msg)

    # Assert
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc.type == "function"
    assert tc.function.name == "execute_python"
    parsed_args = json.loads(tc.function.arguments or "{}")
    assert parsed_args == {"code": "print(2+3)"}
