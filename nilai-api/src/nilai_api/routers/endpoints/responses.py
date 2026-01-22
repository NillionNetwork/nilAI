import json
import logging
import time
import uuid
from base64 import b64encode
from typing import AsyncGenerator, Optional, Union, List, Tuple

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    status,
    Request,
    BackgroundTasks,
)
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.config import CONFIG
from nilai_api.crypto import sign_message
from nilai_api.credit import LLMMeter, LLMUsage
from nilai_api.db.logs import QueryLogContext
from nilai_api.handlers.nildb.handler import get_prompt_from_nildb

# from nilai_api.handlers.nilrag import handle_nilrag_for_responses
from nilai_api.handlers.tools.responses_tool_router import (
    handle_responses_tool_workflow,
)
from nilai_api.handlers.web_search import handle_web_search_for_responses
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state

from nilai_common import ResponseRequest, SignedResponse, Source, ResponseCompletedEvent
from nilauth_credit_middleware import MeteringContext

logger = logging.getLogger(__name__)

responses_router = APIRouter()


async def responses_concurrent_rate_limit(request: Request) -> Tuple[int, str]:
    """Rate limit extractor for concurrent requests to the responses endpoint."""
    body = await request.json()
    try:
        response_request = ResponseRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    key = f"responses:{response_request.model}"
    limit = CONFIG.rate_limiting.model_concurrent_rate_limit.get(
        response_request.model,
        CONFIG.rate_limiting.model_concurrent_rate_limit.get("default", 50),
    )
    return limit, key


async def responses_web_search_rate_limit(request: Request) -> bool:
    """Extracts web_search flag from the request body for rate limiting."""
    body = await request.json()
    try:
        response_request = ResponseRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    return bool(response_request.web_search)


@responses_router.post(
    "/v1/responses", tags=["Responses"], response_model=SignedResponse
)
async def create_response(
    req: ResponseRequest = Body(
        {
            "model": "openai/gpt-oss-20b",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "What is your name?"},
            ],
            "stream": False,
            "web_search": False,
        }
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _rate_limit=Depends(
        RateLimit(
            concurrent_extractor=responses_concurrent_rate_limit,
            web_search_extractor=responses_web_search_rate_limit,
        )
    ),
    auth_info: AuthenticationInfo = Depends(get_auth_info),
    meter: MeteringContext = Depends(LLMMeter),
    log_ctx: QueryLogContext = Depends(QueryLogContext),
) -> Union[SignedResponse, StreamingResponse]:
    """
    Generate a response from the AI model using the Responses API.

    This endpoint provides a more flexible and powerful way to interact with models,
    supporting complex inputs and a structured event stream.

    - **req**: Response request containing input and model specifications
    - **Returns**: Full response with model output, usage statistics, and cryptographic signature

    ### Request Requirements
    - Must include non-empty input (string or structured input)
    - Must specify a model
    - Supports optional instructions to guide the model's behavior
    - Optional web_search parameter to enhance context with current information

    ### Response Components
    - Model-generated text completion
    - Token usage metrics
    - Cryptographically signed response for verification

    ### Processing Steps
    1. Validate input request parameters
    2. If web_search is enabled, perform web search and enhance context
    3. Prepare input for model processing
    4. Generate AI model response
    5. Track and update token usage
    6. Cryptographically sign the response

    ### Web Search Feature
    When web_search=True, the system will:
    - Extract the user's query from the input
    - Perform a web search using Brave API
    - Enhance the context with current information
    - Add search results to instructions for better responses

    ### Potential HTTP Errors
    - **400 Bad Request**:
      - Missing or empty input
      - No model specified
    - **500 Internal Server Error**:
      - Model fails to generate a response
    """
    # Initialize log context early so we can log any errors
    log_ctx.set_user(auth_info.user.user_id)
    log_ctx.set_lockid(meter.lock_id)
    model_name = req.model
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

    try:
        if not req.input:
            raise HTTPException(
                status_code=400,
                detail="Request 'input' field cannot be empty.",
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
        if has_multimodal and (
            not endpoint.metadata.multimodal_support or req.web_search
        ):
            raise HTTPException(
                status_code=400,
                detail="Model does not support multimodal content, remove image inputs from request",
            )

        model_url = endpoint.url + "/v1/"

        logger.info(
            f"[responses] start request_id={request_id} user={auth_info.user.user_id} model={model_name} stream={req.stream} web_search={bool(req.web_search)} tools={bool(req.tools)} multimodal={has_multimodal} url={model_url}"
        )
        log_ctx.set_request_params(
            temperature=req.temperature,
            max_tokens=req.max_output_tokens,
            was_streamed=req.stream or False,
            was_multimodal=has_multimodal,
            was_nildb=bool(auth_info.prompt_document),
            was_nilrag=False,
        )

        client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")
        if auth_info.prompt_document:
            try:
                nildb_prompt: str = await get_prompt_from_nildb(
                    auth_info.prompt_document
                )
                req.ensure_instructions(nildb_prompt)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Unable to extract prompt from nilDB: {str(e)}",
                )

        input_items = req.input
        instructions = req.instructions
        sources: Optional[List[Source]] = None

        if req.web_search:
            logger.info(f"[responses] web_search start request_id={request_id}")
            t_ws = time.monotonic()
            web_search_result = await handle_web_search_for_responses(
                req, model_name, client
            )
            input_items = web_search_result.input
            instructions = web_search_result.instructions
            sources = web_search_result.sources
            logger.info(
                f"[responses] web_search done request_id={request_id} sources={len(sources) if sources else 0} duration_ms={(time.monotonic() - t_ws) * 1000:.0f}"
            )

        if req.stream:

            async def response_stream_generator() -> AsyncGenerator[str, None]:
                t_call = time.monotonic()
                prompt_token_usage = 0
                completion_token_usage = 0

                try:
                    logger.info(f"[responses] stream start request_id={request_id}")
                    log_ctx.start_model_timing()

                    request_kwargs = {
                        "model": req.model,
                        "input": input_items,
                        "instructions": instructions,
                        "stream": True,
                        "top_p": req.top_p,
                        "temperature": req.temperature,
                        "max_output_tokens": req.max_output_tokens,
                        "extra_body": {
                            "stream_options": {
                                "include_usage": True,
                                "continuous_usage_stats": False,
                            }
                        },
                    }
                    if req.tools:
                        request_kwargs["tools"] = req.tools

                    stream = await client.responses.create(**request_kwargs)

                    async for event in stream:
                        payload = event.model_dump(exclude_unset=True)

                        if isinstance(event, ResponseCompletedEvent):
                            if event.response and event.response.usage:
                                usage = event.response.usage
                                prompt_token_usage = usage.input_tokens
                                completion_token_usage = usage.output_tokens
                                payload["response"]["usage"] = usage.model_dump(
                                    mode="json"
                                )

                            if sources:
                                if "data" not in payload:
                                    payload["data"] = {}
                                payload["data"]["sources"] = [
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
                        "[responses] stream done request_id=%s prompt_tokens=%d completion_tokens=%d duration_ms=%.0f total_ms=%.0f",
                        request_id,
                        prompt_token_usage,
                        completion_token_usage,
                        (time.monotonic() - t_call) * 1000,
                        (time.monotonic() - t_start) * 1000,
                    )

                except Exception as e:
                    logger.error(
                        "[responses] stream error request_id=%s error=%s", request_id, e
                    )
                    log_ctx.set_error(error_code=500, error_message=str(e))
                    await log_ctx.commit()
                    yield f"data: {json.dumps({'error': 'stream_failed', 'message': str(e)})}\n\n"

            return StreamingResponse(
                response_stream_generator(), media_type="text/event-stream"
            )

        request_kwargs = {
            "model": req.model,
            "input": input_items,
            "instructions": instructions,
            "top_p": req.top_p,
            "temperature": req.temperature,
            "max_output_tokens": req.max_output_tokens,
        }
        if req.tools:
            request_kwargs["tools"] = req.tools
            request_kwargs["tool_choice"] = req.tool_choice

        logger.info(f"[responses] call start request_id={request_id}")
        t_call = time.monotonic()
        log_ctx.start_model_timing()
        response = await client.responses.create(**request_kwargs)
        log_ctx.end_model_timing()
        logger.info(
            f"[responses] call done request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
        )

        # Handle tool workflow
        log_ctx.start_tool_timing()
        (
            final_response,
            agg_prompt_tokens,
            agg_completion_tokens,
        ) = await handle_responses_tool_workflow(client, req, input_items, response)
        log_ctx.end_tool_timing()

        model_response = SignedResponse(
            **final_response.model_dump(),
            signature="",
            sources=sources,
        )

        if model_response.usage is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model response does not contain usage statistics",
            )

        if agg_prompt_tokens or agg_completion_tokens:
            model_response.usage.input_tokens += agg_prompt_tokens
            model_response.usage.output_tokens += agg_completion_tokens

        prompt_tokens = model_response.usage.input_tokens
        completion_tokens = model_response.usage.output_tokens

        meter.set_response(
            {
                "usage": LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    web_searches=len(sources) if sources else 0,
                )
            }
        )

        # Log query with context
        # Note: Response object structure for tools might differ from Chat,
        # but we'll assume basic usage logging is sufficient or adapt if needed.
        # For now, we don't count tool calls explicitly in log_ctx for responses unless we parse them.
        # Chat.py does: tool_calls_count = len(final_completion.choices[0].message.tool_calls)
        # Responses API structure is different. `final_response` is a Response object.
        # It might not have 'choices'. It has 'output'.

        log_ctx.set_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            web_search_calls=len(sources) if sources else 0,
        )
        background_tasks.add_task(log_ctx.commit)

        response_json = model_response.model_dump_json()
        signature = sign_message(state.private_key, response_json)
        model_response.signature = b64encode(signature).decode()

        logger.info(
            f"[responses] done request_id={request_id} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} total_ms={(time.monotonic() - t_start) * 1000:.0f}"
        )
        return model_response

    except HTTPException as e:
        error_code = e.status_code
        error_message = str(e.detail) if e.detail else str(e)
        logger.error(
            f"[responses] HTTPException request_id={request_id} user={auth_info.user.user_id} "
            f"model={model_name} error_code={error_code} error={error_message}",
            exc_info=True,
        )

        if error_code >= 500:
            if log_ctx.model is None:
                log_ctx.set_model(model_name)
            log_ctx.set_error(error_code=error_code, error_message=error_message)
            await log_ctx.commit()

        raise
    except Exception as e:
        error_message = str(e)
        logger.error(
            f"[responses] unexpected error request_id={request_id} user={auth_info.user.user_id} "
            f"model={model_name} error={error_message}",
            exc_info=True,
        )
        if log_ctx.model is None:
            log_ctx.set_model(model_name)
        log_ctx.set_error(error_code=500, error_message=error_message)
        await log_ctx.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {error_message}",
        )
