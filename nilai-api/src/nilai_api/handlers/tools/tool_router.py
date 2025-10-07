from __future__ import annotations

import json
import uuid
from typing import List, Optional, Tuple, cast
from nilai_common import (
    Message,
    MessageAdapter,
    ChatRequest,
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatToolFunction,
)

from . import code_execution
from openai import AsyncOpenAI

import logging

logger = logging.getLogger(__name__)


async def route_and_execute_tool_call(
    tool_call: ChatCompletionMessageToolCall,
) -> Message:
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


async def process_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCall],
) -> List[Message]:
    """Process a list of tool calls and return their corresponding tool messages.

    Routes each tool call to its implementation and collects the results as
    tool messages that can be appended to the conversation history.
    """
    msgs: List[Message] = []
    for tc in tool_calls:
        msg = await route_and_execute_tool_call(tc)
        msgs.append(msg)
    return msgs


def extract_tool_calls_from_response_message(
    response_message: ChatCompletionMessage,
) -> List[ChatCompletionMessageToolCall]:
    """Return tool calls from a ChatCompletionMessage, parsing content if needed.

    Many models may emit function-calling either via the structured `tool_calls`
    field or encode it as JSON in the assistant `content`. This helper returns a
    normalized list of `ChatCompletionMessageToolCall` objects, using a
    best-effort parse of the content when `tool_calls` is empty.
    """
    if response_message.tool_calls:
        return cast(List[ChatCompletionMessageToolCall], response_message.tool_calls)

    try:
        adapter = MessageAdapter(
            raw=cast(
                Message,
                response_message.model_dump(exclude_unset=True),
            )
        )
        content: Optional[str] = adapter.extract_text()
    except Exception:
        content = response_message.content

    if not content:
        return []

    try:
        data = json.loads(content)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []

    # Support multiple possible schemas
    fn = data.get("function")
    if isinstance(fn, dict) and "name" in fn:
        name = fn.get("name")
        args = fn.get("parameters", {})
    else:
        # Fallbacks for other schemas
        name = data.get("name") or data.get("tool") or data.get("function_name")
        raw_args = data.get("arguments")
        try:
            args = (
                (json.loads(raw_args) if isinstance(raw_args, str) else raw_args)
                or data.get("parameters", {})
                or {}
            )
        except Exception:
            args = data.get("parameters", {}) or {}

    if not isinstance(name, str) or not name:
        return []

    try:
        tool_call = ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4()}",
            type="function",
            function=ChatToolFunction(name=name, arguments=json.dumps(args)),
        )
    except Exception:
        return []

    return [tool_call]


async def handle_tool_workflow(
    client: AsyncOpenAI,
    req: ChatRequest,
    current_messages: List[Message],
    first_response: ChatCompletion,
) -> Tuple[ChatCompletion, int, int]:
    """Execute tool workflow if requested and return final completion and usage.

    - Extracts tool calls from the first response (structured or JSON in content)
    - Executes tools and appends tool messages
    - Runs a follow-up completion providing tool outputs
    - Returns the final ChatCompletion and aggregated usage (prompt, completion)
    """
    logger.info("[tools] evaluating tool workflow for response")

    prompt_tokens = first_response.usage.prompt_tokens if first_response.usage else 0
    completion_tokens = (
        first_response.usage.completion_tokens if first_response.usage else 0
    )

    response_message = first_response.choices[0].message
    tool_calls = extract_tool_calls_from_response_message(response_message)
    logger.info(f"[tools] extracted tool_calls: {tool_calls}")

    if not tool_calls:
        return first_response, 0, 0

    assistant_tool_call_msg = MessageAdapter.new_assistant_tool_call_message(tool_calls)
    current_messages = [*current_messages, assistant_tool_call_msg]

    tool_messages = await process_tool_calls(tool_calls)
    current_messages.extend(tool_messages)

    request_kwargs = {
        "model": req.model,
        "messages": current_messages,  # type: ignore[arg-type]
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "tool_choice": "none",
    }

    logger.info("[tools] performing follow-up completion with tool outputs")
    second: ChatCompletion = await client.chat.completions.create(**request_kwargs)  # type: ignore
    if second.usage:
        prompt_tokens += second.usage.prompt_tokens
        completion_tokens += second.usage.completion_tokens

    return second, prompt_tokens, completion_tokens
