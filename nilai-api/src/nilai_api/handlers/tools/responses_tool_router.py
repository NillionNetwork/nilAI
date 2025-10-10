from __future__ import annotations

import json
import asyncio
import logging
from typing import List, Tuple, Union

from openai import AsyncOpenAI
from nilai_common import (
    ResponseRequest,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
    FunctionCallOutput,
)

# Assuming a code execution module exists
from . import code_execution

logger = logging.getLogger(__name__)


async def route_and_execute_tool_call(
    tool_call: ResponseFunctionToolCall,
) -> FunctionCallOutput:
    """
    Executes a function tool call and returns a correctly formatted FunctionCallOutput
    object.
    """
    func_name = tool_call.name
    arguments = tool_call.arguments or "{}"

    output_json_string = json.dumps({"error": f"Tool '{func_name}' not implemented"})

    if func_name == "execute_python":
        try:
            args = json.loads(arguments)
            code = args.get("code", "")
            result = await code_execution.execute_python(code)

            output_json_string = json.dumps({"result": str(result).strip()})
            logger.info(f"[responses_tool] execute_python result: {result.strip()}")
        except json.JSONDecodeError:
            output_json_string = json.dumps(
                {"error": "Invalid JSON in tool call arguments."}
            )
        except Exception as e:
            output_json_string = json.dumps({"error": f"Error executing tool: {e}"})

    return FunctionCallOutput(
        call_id=tool_call.call_id,
        output=output_json_string,
        type="function_call_output",
    )


async def process_tool_calls(
    tool_calls: List[ResponseFunctionToolCall],
) -> List[FunctionCallOutput]:
    """Processes a list of tool calls and returns them as FunctionCallOutput objects."""
    tasks = [route_and_execute_tool_call(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks)
    return results


def extract_function_tool_calls_from_response(
    response: Response,
) -> List[ResponseFunctionToolCall]:
    """Extracts all function tool call items from a Response object's output."""
    if not response.output:
        return []
    return [
        item for item in response.output if isinstance(item, ResponseFunctionToolCall)
    ]


async def handle_responses_tool_workflow(
    client: AsyncOpenAI,
    req: ResponseRequest,
    input_items: Union[str, List[ResponseInputParam]],
    first_response: Response,
) -> Tuple[Response, int, int]:
    """
    Manages a tool-use workflow by rebuilding the conversation
    history with the full context for each turn.
    """
    logger.info("[responses_tool] evaluating tool workflow for response")

    prompt_tokens = first_response.usage.input_tokens if first_response.usage else 0
    completion_tokens = (
        first_response.usage.output_tokens if first_response.usage else 0
    )

    tool_calls = extract_function_tool_calls_from_response(first_response)
    logger.info(f"[responses_tool] extracted tool_calls: {tool_calls}")

    if not tool_calls:
        return first_response, prompt_tokens, completion_tokens

    tool_results = await process_tool_calls(tool_calls)
    logger.info(f"[responses_tool] tool_results: {tool_results}")

    new_input_items: List[ResponseInputParam] = []
    if isinstance(input_items, str):
        new_input_items.append({"role": "user", "content": input_items})
    elif isinstance(input_items, list):
        new_input_items = list(input_items)

    new_input_items.extend(tool_calls)
    new_input_items.extend(tool_results)

    request_kwargs = {
        "model": req.model,
        "input": new_input_items,
        "instructions": req.instructions,
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
        "tool_choice": "auto",
    }
    
    if req.tools:
        request_kwargs["tools"] = req.tools

    logger.info("[responses_tool] performing follow-up response with tool outputs")
    second_response: Response = await client.responses.create(**request_kwargs)

    if second_response.usage:
        prompt_tokens += second_response.usage.input_tokens
        completion_tokens += second_response.usage.output_tokens

    return second_response, prompt_tokens, completion_tokens