from __future__ import annotations

import json
from typing import List

from openai.types.chat import ChatCompletionMessageToolCall

from nilai_common import Message, MessageAdapter

from . import code_execution

import logging
logger = logging.getLogger(__name__)


async def route_and_execute_tool_call(tool_call: ChatCompletionMessageToolCall) -> Message:
    """Route a single tool call to its implementation and return a tool message.

    The returned message is a dict compatible with OpenAI's ChatCompletionMessageParam
    with role="tool".
    """
    func_name = tool_call.function.name
    arguments = tool_call.function.arguments or "{}"

    if func_name == "execute_python":
        # arguments is a JSON string
        try:
            args = json.loads(arguments)
        except Exception:
            args = {}
        code = args.get("code", "")
        result = await code_execution.execute_python(code)
        logger.info(f"[tool] execute_python result: {result}")
        return MessageAdapter.new_tool_message(
            name="execute_python",
            content=result,
            tool_call_id=tool_call.id,
        )

    # Unknown tool: return an error message to the model
    return MessageAdapter.new_tool_message(
        name=func_name,
        content=f"Tool '{func_name}' not implemented",
        tool_call_id=tool_call.id,
    )


async def process_tool_calls(tool_calls: List[ChatCompletionMessageToolCall]) -> List[Message]:
    msgs: List[Message] = []
    for tc in tool_calls:
        msg = await route_and_execute_tool_call(tc)
        msgs.append(msg)
    return msgs

