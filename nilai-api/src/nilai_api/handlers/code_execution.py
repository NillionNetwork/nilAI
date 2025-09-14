from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, List

from nilai_common import Message, ChatRequest, CodeExecutionResult
from nilai_api.handlers.web_search_tool import handle_web_search_tool_call

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

try:
    from e2b_code_interpreter import Sandbox  # type: ignore
except Exception:  # pragma: no cover - if not installed, feature remains inactive
    Sandbox = None  # type: ignore


async def handle_code_execution_tool_calls_if_any(
    response: ChatCompletion,
    current_messages: List[Message],
    req: ChatRequest,
    client: AsyncOpenAI,
    logger: logging.Logger,
) -> CodeExecutionResult:
    """Handle non-stream tool calls for code execution and return the final response.

    - Detects assistant tool calls from the initial response.
    - Executes supported tools (currently: execute_python via e2b sandbox).
    - Appends tool outputs to the conversation.
    - Performs a follow-up completion call and returns a CodeExecutionResult
      containing the final response and usage deltas.
    """

    # Default: no follow-up call, no extra usage
    prompt_tokens_delta: int = 0
    completion_tokens_delta: int = 0
    final_openai_response: ChatCompletion = response

    # Extract tool calls if any
    try:
        first_choice = response.choices[0] if response.choices else None
        tool_calls = first_choice.message.tool_calls if first_choice else None  # type: ignore[attr-defined]
    except Exception:
        tool_calls = None

    if not tool_calls:
        return CodeExecutionResult(
            response=final_openai_response,
            prompt_tokens_delta=prompt_tokens_delta,
            completion_tokens_delta=completion_tokens_delta,
        )

    logger.info(
        f"[chat] tool_calls detected count={len(tool_calls)}"
    )

    follow_up_messages = list(current_messages)

    # Append assistant message with tool call metadata to preserve function-call context
    try:
        assistant_param: Any = {
            "role": "assistant",
            "content": first_choice.message.content,  # type: ignore[attr-defined]
            "tool_calls": [tc.model_dump(exclude_none=True, by_alias=True) for tc in tool_calls],
        }
        follow_up_messages.append(assistant_param)  # type: ignore
    except Exception as e:
        logger.error(
            f"[chat] failed to prepare assistant tool_calls message error={e}"
        )

    # Execute supported tool calls
    extra_sources = []
    for tc in tool_calls:
        try:
            func = getattr(tc, "function", None)
            name = getattr(func, "name", None)
            args_json = getattr(func, "arguments", "{}")
            tool_call_id = getattr(tc, "id", None)

            if name == "execute_python":
                if Sandbox is None:
                    raise RuntimeError(
                        "e2b-code-interpreter is not installed. Install to enable code execution."
                    )
                # Parse code
                try:
                    args = json.loads(args_json) if isinstance(args_json, str) else args_json
                    code = args.get("code", "")
                except Exception:
                    code = ""

                def _run_in_sandbox(src: str) -> str:
                    with Sandbox.create() as sandbox:  # type: ignore
                        execution = sandbox.run_code(src)
                        return getattr(execution, "text", "")

                result: str = await asyncio.to_thread(_run_in_sandbox, code)

                follow_up_messages.append(
                    {
                        "role": "tool",
                        "name": "execute_python",
                        "content": result,
                        "tool_call_id": tool_call_id,
                    }
                )
            elif name == "web_search":
                # Use the same pipeline as the `web_search=True` flag.
                try:
                    outcome = await handle_web_search_tool_call(
                        req=req, model_name=req.model, client=client, logger=logger
                    )
                    extra_sources.extend(outcome.sources)
                    follow_up_messages.append(
                        {
                            "role": "tool",
                            "name": "web_search",
                            "content": outcome.content,
                            "tool_call_id": tool_call_id,
                        }
                    )
                except Exception as e:
                    logger.error("[chat] web_search tool pipeline error: %s", e)
                    follow_up_messages.append(
                        {
                            "role": "tool",
                            "name": "web_search",
                            "content": "(web_search) Error: pipeline failed.",
                            "tool_call_id": tool_call_id,
                        }
                    )
            else:
                logger.info(
                    f"[chat] unsupported tool name tool={name}, skipping execution"
                )
        except Exception as e:
            logger.error(f"[chat] tool execution failed error={e}")

    # Follow-up completion call using tool outputs
    request_kwargs = {
        "model": req.model,
        "messages": follow_up_messages,  # type: ignore
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.tools:
        request_kwargs["tools"] = req.tools  # type: ignore

    final_openai_response = await client.chat.completions.create(**request_kwargs)  # type: ignore
    try:
        if getattr(final_openai_response, "usage", None):
            prompt_tokens_delta = final_openai_response.usage.prompt_tokens
            completion_tokens_delta = final_openai_response.usage.completion_tokens
    except Exception:
        pass

    return CodeExecutionResult(
        response=final_openai_response,
        prompt_tokens_delta=prompt_tokens_delta,
        completion_tokens_delta=completion_tokens_delta,
        extra_sources=extra_sources,
    )
