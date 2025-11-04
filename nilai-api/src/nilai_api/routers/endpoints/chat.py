import json
import logging
import time
import uuid
from base64 import b64encode
from typing import AsyncGenerator, Optional, Union, List, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.config import CONFIG
from nilai_api.crypto import sign_message
from nilai_api.credit import LLMMeter, LLMUsage
from nilai_api.db.logs import QueryLogManager, QueryLogContext
from nilai_api.db.users import UserManager
from nilai_api.handlers.nildb.handler import get_prompt_from_nildb
from nilai_api.handlers.nilrag import handle_nilrag
from nilai_api.handlers.tools.tool_router import handle_tool_workflow
from nilai_api.handlers.web_search import handle_web_search
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state

from nilai_common import (
    ChatRequest,
    MessageAdapter,
    SignedChatCompletion,
    Source,
)
from nilauth_credit_middleware import MeteringContext

logger = logging.getLogger(__name__)

chat_completion_router = APIRouter()


async def chat_completion_concurrent_rate_limit(request: Request) -> Tuple[int, str]:
    body = await request.json()
    try:
        chat_request = ChatRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    key = f"chat:{chat_request.model}"
    limit = CONFIG.rate_limiting.model_concurrent_rate_limit.get(
        chat_request.model,
        CONFIG.rate_limiting.model_concurrent_rate_limit.get("default", 50),
    )
    return limit, key


async def chat_completion_web_search_rate_limit(request: Request) -> bool:
    """Extract web_search flag from request body for rate limiting."""
    body = await request.json()
    try:
        chat_request = ChatRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    return bool(chat_request.web_search)


@chat_completion_router.post("/v1/chat/completions", tags=["Chat"], response_model=None)
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                MessageAdapter.new_message(
                    role="system", content="You are a helpful assistant."
                ),
                MessageAdapter.new_message(role="user", content="What is your name?"),
            ],
        )
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _rate_limit=Depends(
        RateLimit(
            concurrent_extractor=chat_completion_concurrent_rate_limit,
            web_search_extractor=chat_completion_web_search_rate_limit,
        )
    ),
    auth_info: AuthenticationInfo = Depends(get_auth_info),
    meter: MeteringContext = Depends(LLMMeter),
    log_ctx: QueryLogContext = Depends(QueryLogContext),
) -> Union[SignedChatCompletion, StreamingResponse]:
    """
    Generate a chat completion response from the AI model.

    - **req**: Chat completion request containing messages and model specifications
    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Full chat response with model output, usage statistics, and cryptographic signature

    ### Request Requirements
    - Must include non-empty list of messages
    - Must specify a model
    - Supports multiple message formats (system, user, assistant)
    - Optional web_search parameter to enhance context with current information

    ### Response Components
    - Model-generated text completion
    - Token usage metrics
    - Cryptographically signed response for verification

    ### Processing Steps
    1. Validate input request parameters
    2. If web_search is enabled, perform web search and enhance context
    3. Prepare messages for model processing
    4. Generate AI model response
    5. Track and update token usage
    6. Cryptographically sign the response

    ### Web Search Feature
    When web_search=True, the system will:
    - Extract the user's query from the last user message
    - Perform a web search using Brave API
    - Enhance the conversation context with current information
    - Add search results as a system message for better responses

    ### Potential HTTP Errors
    - **400 Bad Request**:
      - Missing messages list
      - No model specified
    - **500 Internal Server Error**:
      - Model fails to generate a response

    ### Example
    ```python
    # Generate a chat completion with web search
    request = ChatRequest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is your name?"}
        ],
    )
    response = await chat_completion(request, user)
    """
    # Initialize log context early so we can log any errors
    log_ctx.set_user(auth_info.user.user_id)
    log_ctx.set_lockid(meter.lock_id)
    model_name = req.model
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

    try:
        if len(req.messages) == 0:
            raise HTTPException(
                status_code=400,
                detail="Request contained 0 messages",
            )
        endpoint = await state.get_model(model_name)
        if endpoint is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name {model_name}, check /v1/models for options",
            )

        # Now we have a valid model, set it in log context
        log_ctx.set_model(model_name)

        if not endpoint.metadata.tool_support and req.tools:
            raise HTTPException(
                status_code=400,
                detail="Model does not support tool usage, remove tools from request",
            )

        has_multimodal = req.has_multimodal_content()
        logger.info(f"[chat] has_multimodal: {has_multimodal}")
        if has_multimodal and (
            not endpoint.metadata.multimodal_support or req.web_search
        ):
            raise HTTPException(
                status_code=400,
                detail="Model does not support multimodal content, remove image inputs from request",
            )

        model_url = endpoint.url + "/v1/"

        logger.info(
            f"[chat] start request_id={request_id} user={auth_info.user.user_id} model={model_name} stream={req.stream} web_search={bool(req.web_search)} tools={bool(req.tools)} multimodal={has_multimodal} url={model_url}"
        )
        log_ctx.set_request_params(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            was_streamed=req.stream or False,
            was_multimodal=has_multimodal,
            was_nildb=bool(auth_info.prompt_document),
            was_nilrag=bool(req.nilrag),
        )

        client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")
        if auth_info.prompt_document:
            try:
                nildb_prompt: str = await get_prompt_from_nildb(
                    auth_info.prompt_document
                )
                req.messages.insert(
                    0, MessageAdapter.new_message(role="system", content=nildb_prompt)
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unable to extract prompt from nilDB: {str(e)}",
                )

        if req.nilrag:
            logger.info(f"[chat] nilrag start request_id={request_id}")
            t_nilrag = time.monotonic()
            await handle_nilrag(req)
            logger.info(
                f"[chat] nilrag done request_id={request_id} duration_ms={(time.monotonic() - t_nilrag) * 1000:.0f}"
            )

        messages = req.messages
        sources: Optional[List[Source]] = None

        if req.web_search:
            logger.info(f"[chat] web_search start request_id={request_id}")
            t_ws = time.monotonic()
            web_search_result = await handle_web_search(req, model_name, client)
            messages = web_search_result.messages
            sources = web_search_result.sources
            logger.info(
                f"[chat] web_search done request_id={request_id} sources={len(sources) if sources else 0} duration_ms={(time.monotonic() - t_ws) * 1000:.0f}"
            )
            logger.info(f"[chat] web_search messages: {messages}")

        if req.stream:

            async def chat_completion_stream_generator() -> AsyncGenerator[str, None]:
                t_call = time.monotonic()
                prompt_token_usage = 0
                completion_token_usage = 0

                try:
                    logger.info(f"[chat] stream start request_id={request_id}")

                    log_ctx.start_model_timing()

                    request_kwargs = {
                        "model": req.model,
                        "messages": messages,
                        "stream": True,
                        "top_p": req.top_p,
                        "temperature": req.temperature,
                        "max_tokens": req.max_tokens,
                        "extra_body": {
                            "stream_options": {
                                "include_usage": True,
                                "continuous_usage_stats": False,
                            }
                        },
                    }
                    if req.tools:
                        request_kwargs["tools"] = req.tools

                    response = await client.chat.completions.create(**request_kwargs)

                    async for chunk in response:
                        if chunk.usage is not None:
                            prompt_token_usage = chunk.usage.prompt_tokens
                            completion_token_usage = chunk.usage.completion_tokens

                        payload = chunk.model_dump(exclude_unset=True)

                        if chunk.usage is not None and sources:
                            payload["sources"] = [
                                s.model_dump(mode="json") for s in sources
                            ]

                        yield f"data: {json.dumps(payload)}\n\n"

                    log_ctx.end_model_timing()
                    meter.set_response(
                        {
                            "usage": LLMUsage(
                                prompt_tokens=prompt_token_usage,
                                completion_tokens=completion_token_usage,
                                web_searches=len(sources) if sources else 0,
                            )
                        }
                    )
                    log_ctx.set_usage(
                        prompt_tokens=prompt_token_usage,
                        completion_tokens=completion_token_usage,
                        web_search_calls=len(sources) if sources else 0,
                    )
                    background_tasks.add_task(log_ctx.commit)
                    logger.info(
                        "[chat] stream done request_id=%s prompt_tokens=%d completion_tokens=%d "
                        "duration_ms=%.0f total_ms=%.0f",
                        request_id,
                        prompt_token_usage,
                        completion_token_usage,
                        (time.monotonic() - t_call) * 1000,
                        (time.monotonic() - t_start) * 1000,
                    )

                except Exception as e:
                    logger.error(
                        "[chat] stream error request_id=%s error=%s", request_id, e
                    )
                    log_ctx.set_error(error_code=500, error_message=str(e))
                    await log_ctx.commit()
                    yield f"data: {json.dumps({'error': 'stream_failed', 'message': str(e)})}\n\n"

            return StreamingResponse(
                chat_completion_stream_generator(),
                media_type="text/event-stream",
            )

        current_messages = messages
        request_kwargs = {
            "model": req.model,
            "messages": current_messages,  # type: ignore
            "top_p": req.top_p,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.tools:
            request_kwargs["tools"] = req.tools  # type: ignore
            request_kwargs["tool_choice"] = req.tool_choice

        logger.info(f"[chat] call start request_id={request_id}")
        logger.info(f"[chat] call message: {current_messages}")
        t_call = time.monotonic()
        log_ctx.start_model_timing()
        response = await client.chat.completions.create(**request_kwargs)  # type: ignore
        log_ctx.end_model_timing()
        logger.info(
            f"[chat] call done request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
        )
        logger.info(f"[chat] call response: {response}")

        # Handle tool workflow fully inside tools.router
        log_ctx.start_tool_timing()
        (
            final_completion,
            agg_prompt_tokens,
            agg_completion_tokens,
        ) = await handle_tool_workflow(client, req, current_messages, response)
        log_ctx.end_tool_timing()
        logger.info(f"[chat] call final_completion: {final_completion}")
        model_response = SignedChatCompletion(
            **final_completion.model_dump(),
            signature="",
            sources=sources,
        )

        logger.info(
            f"[chat] model_response request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
        )

        if model_response.usage is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model response does not contain usage statistics",
            )

        if agg_prompt_tokens or agg_completion_tokens:
            total_prompt_tokens = response.usage.prompt_tokens
            total_completion_tokens = response.usage.completion_tokens

            total_prompt_tokens += agg_prompt_tokens
            total_completion_tokens += agg_completion_tokens

            model_response.usage.prompt_tokens = total_prompt_tokens
            model_response.usage.completion_tokens = total_completion_tokens
            model_response.usage.total_tokens = (
                total_prompt_tokens + total_completion_tokens
            )

        # Update token usage in DB
        meter.set_response(
            {
                "usage": LLMUsage(
                    prompt_tokens=model_response.usage.prompt_tokens,
                    completion_tokens=model_response.usage.completion_tokens,
                    web_searches=len(sources) if sources else 0,
                )
            }
        )

        # Log query with context
        tool_calls_count = 0
        if final_completion.choices and final_completion.choices[0].message.tool_calls:
            tool_calls_count = len(final_completion.choices[0].message.tool_calls)

        log_ctx.set_usage(
            prompt_tokens=model_response.usage.prompt_tokens,
            completion_tokens=model_response.usage.completion_tokens,
            tool_calls=tool_calls_count,
            web_search_calls=len(sources) if sources else 0,
        )
        # Use background task for successful requests to avoid blocking response
        background_tasks.add_task(log_ctx.commit)

        # Sign the response
        response_json = model_response.model_dump_json()
        signature = sign_message(state.private_key, response_json)
        model_response.signature = b64encode(signature).decode()

        logger.info(
            f"[chat] done request_id={request_id} prompt_tokens={model_response.usage.prompt_tokens} completion_tokens={model_response.usage.completion_tokens} total_ms={(time.monotonic() - t_start) * 1000:.0f}"
        )
        return model_response
    except HTTPException as e:
        # Extract error code from HTTPException, default to status code
        error_code = e.status_code
        error_message = str(e.detail) if e.detail else str(e)
        logger.error(
            f"[chat] HTTPException request_id={request_id} user={auth_info.user.user_id} "
            f"model={model_name} error_code={error_code} error={error_message}",
            exc_info=True,
        )

        # Only log server errors (5xx) to database to prevent DoS attacks via client errors
        # Client errors (4xx) are logged to application logs only
        if error_code >= 500:
            # Set model if not already set (e.g., for validation errors before model validation)
            if log_ctx.model is None:
                log_ctx.set_model(model_name)
            log_ctx.set_error(error_code=error_code, error_message=error_message)
            await log_ctx.commit()
        # For 4xx errors, we skip DB logging - they're logged above via logger.error()
        # This prevents DoS attacks where attackers send many invalid requests

        raise
    except Exception as e:
        # Catch any other unexpected exceptions
        error_message = str(e)
        logger.error(
            f"[chat] unexpected error request_id={request_id} user={auth_info.user.user_id} "
            f"model={model_name} error={error_message}",
            exc_info=True,
        )
        # Set model if not already set
        if log_ctx.model is None:
            log_ctx.set_model(model_name)
        log_ctx.set_error(error_code=500, error_message=error_message)
        await log_ctx.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {error_message}",
        )