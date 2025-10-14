from __future__ import annotations

import json
import asyncio
import logging
import uuid
from typing import List, Tuple, Union, cast

from openai import AsyncOpenAI
from nilai_common import (
    ResponseRequest,
    Response,
    ResponseFunctionToolCall,
    FunctionCallOutput,
    ResponseInputItemParam,
    EasyInputMessageParam,
    ResponseFunctionToolCallParam,
)

from . import code_execution

logger = logging.getLogger(__name__)

AVAILABLE_TOOLS = {"execute_python"}


async def route_and_execute_tool_call(
    tool_call: ResponseFunctionToolCallParam,
) -> FunctionCallOutput:
    """Route and execute a single tool call, returning the result as a FunctionCallOutput.

    Currently supports:
    - execute_python: Executes Python code in a sandbox environment

    For unknown tools, returns an error message in the output field.

    Args:
        tool_call: Tool call parameter containing name, arguments, and call_id

    Returns:
        FunctionCallOutput object with execution result or error message
    """
    func_name = tool_call["name"]
    arguments = tool_call["arguments"] or "{}"

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
        id=str(uuid.uuid4()),
        call_id=tool_call["call_id"],
        output=output_json_string,
        type="function_call_output",
    )


async def process_tool_calls(
    tool_calls: List[ResponseFunctionToolCallParam],
) -> List[FunctionCallOutput]:
    """Process multiple tool calls concurrently using asyncio.gather.

    Executes all tool calls in parallel for optimal performance.
    Unknown tools will return error messages in their output.

    Args:
        tool_calls: List of tool call parameters to execute

    Returns:
        List of FunctionCallOutput objects with execution results
    """
    tasks = [route_and_execute_tool_call(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)


def extract_function_tool_calls_from_response(
    response: Response,
) -> List[ResponseFunctionToolCallParam]:
    """Extract all function tool calls from a Response object's output.

    Filters the response output for ResponseFunctionToolCall items and converts
    them to the parameter format required for tool execution.

    Args:
        response: Response object from the model containing potential tool calls

    Returns:
        List of ResponseFunctionToolCallParam objects ready for execution
    """
    if not response.output:
        return []
    return [
        ResponseFunctionToolCallParam(
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
            type="function_call",
        )
        for item in response.output
        if isinstance(item, ResponseFunctionToolCall)
    ]


def check_if_all_tools_available(
    tool_calls: List[ResponseFunctionToolCallParam],
) -> bool:
    """Check if all requested tools are available in the registry.

    Validates that each tool call references a tool that exists in AVAILABLE_TOOLS.
    Logs a warning for the first unavailable tool encountered.

    Args:
        tool_calls: List of tool calls to validate

    Returns:
        True if all tools are available, False if any tool is unavailable
    """
    for tool_call in tool_calls:
        if tool_call["name"] not in AVAILABLE_TOOLS:
            logger.warning(
                f"[responses_tool] Tool '{tool_call['name']}' not available, shortcutting workflow"
            )
            return False
    return True


async def handle_responses_tool_workflow(
    client: AsyncOpenAI,
    req: ResponseRequest,
    input_items: Union[str, List[ResponseInputItemParam]],
    first_response: Response,
) -> Tuple[Response, int, int]:
    """Handle the complete tool workflow for responses API.

    This function manages the multi-turn tool execution flow:
    1. Extracts tool calls from the model's first response
    2. Validates all tools are available in the registry
    3. If any tool is unavailable, shortcuts the workflow and returns the first response
    4. If all tools are available, executes them concurrently
    5. Constructs a new input with original messages + tool calls + tool results
    6. Makes a follow-up API call with tool results
    7. Returns the final response with aggregated token usage

    Args:
        client: AsyncOpenAI client for making API calls
        req: Original request parameters
        input_items: Original input messages (string or list)
        first_response: Initial response from the model containing tool calls

    Returns:
        Tuple of (final_response, total_prompt_tokens, total_completion_tokens)
    """
    logger.info("[responses_tool] evaluating tool workflow for response")

    prompt_tokens = first_response.usage.input_tokens if first_response.usage else 0
    completion_tokens = (
        first_response.usage.output_tokens if first_response.usage else 0
    )

    tool_calls = extract_function_tool_calls_from_response(first_response)
    logger.info(f"[responses_tool] extracted tool_calls: {tool_calls}")

    if not tool_calls or not check_if_all_tools_available(tool_calls):
        return first_response, prompt_tokens, completion_tokens

    tool_results = await process_tool_calls(tool_calls)
    logger.info(f"[responses_tool] tool_results: {tool_results}")

    new_input_items: List[ResponseInputItemParam] = []
    if isinstance(input_items, str):
        new_input_items.append(
            EasyInputMessageParam(
                role="user",
                content=input_items,
                type="message",
            )
        )
    elif isinstance(input_items, list):
        new_input_items = list(input_items)

    if first_response.output:
        new_input_items.extend(
            [
                cast(
                    ResponseInputItemParam,
                    item.model_dump(exclude_unset=True, mode="json"),
                )
                for item in first_response.output
            ]
        )

    new_input_items.extend(
        [
            cast(
                ResponseInputItemParam,
                result.model_dump(exclude_unset=True, mode="json"),
            )
            for result in tool_results
        ]
    )

    request_kwargs = {
        "model": req.model,
        "input": new_input_items,
        "instructions": req.instructions,
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_output_tokens": req.max_output_tokens,
    }

    logger.info("[responses_tool] performing follow-up completion with tool outputs")
    second_response: Response = await client.responses.create(**request_kwargs)

    if second_response.usage:
        prompt_tokens += second_response.usage.input_tokens
        completion_tokens += second_response.usage.output_tokens

    return second_response, prompt_tokens, completion_tokens
