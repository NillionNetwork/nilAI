import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from nilai_api.handlers.tools import responses_tool_router
from nilai_common import (
    ResponseRequest,
    ResponseFunctionToolCallParam,
)
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseUsage,
    FunctionToolParam,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)


@pytest.mark.asyncio
async def test_route_and_execute_tool_call_invokes_code_execution(mocker):
    tool_call: ResponseFunctionToolCallParam = {
        "type": "function_call",
        "call_id": "call_123",
        "name": "execute_python",
        "arguments": json.dumps({"code": "print(6*7)"}),
    }

    mock_exec = mocker.patch(
        "nilai_api.handlers.tools.responses_tool_router.code_execution.execute_python",
        new_callable=AsyncMock,
        return_value="42",
    )

    result = await responses_tool_router.route_and_execute_tool_call(tool_call)

    mock_exec.assert_awaited_once_with("print(6*7)")
    assert result.type == "function_call_output"
    assert result.call_id == "call_123"
    output_str = result.output if isinstance(result.output, str) else "{}"
    payload = json.loads(output_str)
    assert payload == {"result": "42"}


def make_response_usage(prompt: int, completion: int) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=prompt,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=completion,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=prompt + completion,
    )


def make_tool_call_response(code: str) -> Response:
    return Response(
        id="resp_tool",
        object="response",
        model="openai/gpt-oss-20b",
        created_at=123456.0,
        status="completed",
        output=[
            ResponseFunctionToolCall(
                id="call_abc",
                type="function_call",
                call_id="call_abc",
                name="execute_python",
                arguments=json.dumps({"code": code}),
                status="completed",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=make_response_usage(prompt=10, completion=5),
    )


@pytest.mark.asyncio
async def test_handle_responses_tool_workflow_executes_and_uses_result(mocker):
    tool: FunctionToolParam = {
        "type": "function",
        "name": "execute_python",
        "description": "Execute small Python code snippets.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
        "strict": None,
    }

    req = ResponseRequest(
        model="openai/gpt-oss-20b",
        input=[
            {
                "role": "user",
                "type": "message",
                "content": [
                    {"type": "input_text", "text": "What is 6*7?"},
                ],
            }
        ],
        tools=[tool],
    )

    first_response = make_tool_call_response("print(6*7)")

    mock_exec = mocker.patch(
        "nilai_api.handlers.tools.responses_tool_router.code_execution.execute_python",
        new_callable=AsyncMock,
        return_value="42",
    )

    second_response = Response(
        id="resp_final",
        object="response",
        model=req.model,
        created_at=123457.0,
        status="completed",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=make_response_usage(prompt=7, completion=2),
    )

    mock_client = MagicMock()
    mock_responses = MagicMock()
    mock_responses.create = AsyncMock(return_value=second_response)
    mock_client.responses = mock_responses

    (
        final,
        prompt_tokens,
        completion_tokens,
    ) = await responses_tool_router.handle_responses_tool_workflow(
        mock_client,
        req,
        cast(Any, req.input),
        first_response,
    )

    mock_exec.assert_awaited_once_with("print(6*7)")
    assert final == second_response
    assert first_response.usage is not None
    assert second_response.usage is not None
    assert (
        prompt_tokens
        == first_response.usage.input_tokens + second_response.usage.input_tokens
    )
    assert (
        completion_tokens
        == first_response.usage.output_tokens + second_response.usage.output_tokens
    )


def test_extract_function_tool_calls_from_response():
    response = make_tool_call_response("print(2+3)")

    tool_calls = responses_tool_router.extract_function_tool_calls_from_response(
        response
    )

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["type"] == "function_call"
    assert tc["name"] == "execute_python"
    assert tc["call_id"] == "call_abc"
    args = json.loads(tc["arguments"] or "{}")
    assert args == {"code": "print(2+3)"}
