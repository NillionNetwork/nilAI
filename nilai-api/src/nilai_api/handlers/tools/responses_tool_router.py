from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple, Union, cast

from openai import AsyncOpenAI

from openai.types.responses import (
    Response, 
    ResponseFunctionToolCall, 
    ResponseFunctionToolCallOutputItem
)

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_item import ResponseOutputItem

from nilai_common import ResponseRequest

from . import code_execution

logger = logging.getLogger(__name__)


async def route_and_execute_tool_call(
    tool_call: ResponseFunctionToolCall,
) -> ResponseFunctionToolCallOutputItem: # <--- Update return type hint
    func_name = tool_call.name
    arguments = tool_call.arguments or "{}"

    if func_name == "execute_python":
        try:
            args = json.loads(arguments)
        except Exception:
            args = {}
        code = args.get("code", "")
        result = await code_execution.execute_python(code)
        logger.info(f"[responses_tool] execute_python result: {result}")
        # Return the correct object type
        return ResponseFunctionToolCallOutputItem(tool_call_id=tool_call.call_id, output=result)

    # Return the correct object type for the "not implemented" case
    return ResponseFunctionToolCallOutputItem(
        tool_call_id=tool_call.call_id,
        output=f"Tool '{func_name}' not implemented",
    )



async def process_tool_calls(
    tool_calls: List[ResponseFunctionToolCall],
) -> List[ResponseFunctionToolCallOutputItem]: # <--- Update this type hint
    results: List[ResponseFunctionToolCallOutputItem] = []
    for tc in tool_calls:
        result = await route_and_execute_tool_call(tc)
        results.append(result)
    return results


def extract_function_tool_calls_from_response(
    response: Response,
) -> List[ResponseFunctionToolCall]:
    """Extracts all function tool call items from a Response object's output."""
    if not response.output:
        return []

    # Use isinstance for a robust, type-safe check.
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


async def handle_responses_tool_workflow(
    client: AsyncOpenAI,
    req: ResponseRequest,
    input_items: Union[str, ResponseInputParam],
    first_response: Response,
) -> Tuple[Response, int, int]:
    logger.info("[responses_tool] evaluating tool workflow for response")

    prompt_tokens = first_response.usage.input_tokens if first_response.usage else 0
    completion_tokens = (
        first_response.usage.output_tokens if first_response.usage else 0
    )

    tool_calls = extract_function_tool_calls_from_response(first_response)
    logger.info(f"[responses_tool] extracted tool_calls: {tool_calls}")

    if not tool_calls:
        return first_response, 0, 0

    tool_results = await process_tool_calls(tool_calls)

    new_input_items: List = []
    if isinstance(input_items, str):
        new_input_items = [{"role": "user", "content": input_items}]
    elif isinstance(input_items, list):
        new_input_items = list(input_items)

    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments or "{}",
                },
            }
            for tc in tool_calls
        ],
    }
    new_input_items.append(assistant_message)

    new_input_items.extend(tool_results)

    request_kwargs = {
        "model": req.model,
        "input": new_input_items,
        "instructions": req.instructions,
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
        "tool_choice": "none",
    }

    logger.info("[responses_tool] performing follow-up response with tool outputs")
    second: Response = await client.responses.create(**request_kwargs)
    if second.usage:
        prompt_tokens += second.usage.input_tokens
        completion_tokens += second.usage.output_tokens

    return second, prompt_tokens, completion_tokens

