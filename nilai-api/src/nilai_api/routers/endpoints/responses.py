import json
import logging
import time
import uuid
from base64 import b64encode
from typing import AsyncGenerator, Optional, Union, List, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.config import CONFIG
from nilai_api.crypto import sign_message
from nilai_api.db.logs import QueryLogManager
from nilai_api.db.users import UserManager
from nilai_api.handlers.nildb.handler import get_prompt_from_nildb
# from nilai_api.handlers.nilrag import handle_nilrag_for_responses # Assumes an adapted handler
# from nilai_api.handlers.tools.responses_tool_router import handle_responses_tool_workflow # Assumes an adapted handler
# from nilai_api.handlers.web_search import handle_web_search_for_responses # Assumes an adapted handler
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state

# Import the new Response API models we created
from nilai_common import (
    ResponseRequest,
    InputItemAdapter,
    SignedResponse,
    Source,
)

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


@responses_router.post("/v1/responses", tags=["Responses"], response_model=None)
async def create_response(
    req: ResponseRequest = Body(
        ResponseRequest(
            model="openai/gpt-oss-20b",
            instructions="You are a helpful assistant.",
            input=[
                InputItemAdapter.new_message_item(role="user", content="What is your name?"),
            ],
            stream=False,
            web_search=False
        )
    ),
    _rate_limit=Depends(
        RateLimit(
            concurrent_extractor=responses_concurrent_rate_limit,
            web_search_extractor=responses_web_search_rate_limit,
        )
    ),
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> Union[SignedResponse, StreamingResponse]:
    """
    Generate a response from the AI model using the Responses API.
    
    This endpoint provides a more flexible and powerful way to interact with models,
    supporting complex inputs and a structured event stream.
    """
    if not req.input:
        raise HTTPException(
            status_code=400,
            detail="Request 'input' field cannot be empty.",
        )
    model_name = req.model
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

    endpoint = await state.get_model(model_name)
    if endpoint is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name {model_name}, check /v1/models for options",
        )

    if not endpoint.metadata.tool_support and req.tools:
        raise HTTPException(
            status_code=400,
            detail="Model does not support tool usage, remove tools from request",
        )

    has_multimodal = req.has_multimodal_content()
    if has_multimodal and (not endpoint.metadata.multimodal_support or req.web_search):
        raise HTTPException(
            status_code=400,
            detail="Model does not support multimodal content, remove image inputs from request",
        )

    model_url = endpoint.url + "/v1/"
    logger.info(
        f"[responses] start request_id={request_id} user={auth_info.user.userid} model={model_name} stream={req.stream} web_search={bool(req.web_search)} tools={bool(req.tools)} multimodal={has_multimodal} url={model_url}"
    )

    client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")
    if auth_info.prompt_document:
        try:
            nildb_prompt: str = await get_prompt_from_nildb(auth_info.prompt_document)
            req.ensure_instructions(nildb_prompt)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unable to extract prompt from nilDB: {str(e)}",
            )

    input_items = req.input
    sources: Optional[List[Source]] = None

    if req.web_search:
        logger.info(f"[responses] web_search start request_id={request_id}")
        t_ws = time.monotonic()
        # web_search_result = await handle_web_search_for_responses(req, model_name, client)
        # input_items = web_search_result.input_items
        # sources = web_search_result.sources
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
                request_kwargs = req.model_dump(exclude_unset=True, exclude={"web_search"})
                request_kwargs["input"] = input_items
                request_kwargs["stream"] = True

                stream = await client.responses.create(**request_kwargs)

                async for event in stream:
                    payload = event.model_dump(exclude_unset=True)

                    if event.event == "response.completed" and event.data.usage:
                        usage = event.data.usage
                        prompt_token_usage = usage.input_tokens
                        completion_token_usage = usage.output_tokens

                        if sources:
                            payload["data"]["sources"] = [s.model_dump(mode="json") for s in sources]
                    
                    yield f"data: {json.dumps(payload)}\n\n"

                await UserManager.update_token_usage(
                    auth_info.user.userid, prompt_tokens=prompt_token_usage, completion_tokens=completion_token_usage
                )
                await QueryLogManager.log_query(
                    auth_info.user.userid, model=req.model, prompt_tokens=prompt_token_usage,
                    completion_tokens=completion_token_usage, web_search_calls=len(sources) if sources else 0,
                )
                logger.info(
                    "[responses] stream done request_id=%s prompt_tokens=%d completion_tokens=%d duration_ms=%.0f total_ms=%.0f",
                    request_id, prompt_token_usage, completion_token_usage,
                    (time.monotonic() - t_call) * 1000, (time.monotonic() - t_start) * 1000,
                )

            except Exception as e:
                logger.error("[responses] stream error request_id=%s error=%s", request_id, e)
                yield f"data: {json.dumps({'error': 'stream_failed', 'message': str(e)})}\n\n"

        return StreamingResponse(response_stream_generator(), media_type="text/event-stream")

    # --- Non-Streaming Logic ---
    request_kwargs = req.model_dump(exclude_unset=True, exclude={"web_search", "stream"})
    request_kwargs["input"] = input_items

    logger.info(f"[responses] call start request_id={request_id}")
    t_call = time.monotonic()
    
    response = await client.responses.create(**request_kwargs)
    logger.info(
        f"[responses] call done request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
    )

    # FIX 2: Handle the case where tool workflow is disabled to prevent NameError.
    # Initialize variables for the non-tool-call path.
    final_response = response
    agg_prompt_tokens = 0
    agg_completion_tokens = 0

    # If you enable tool workflow, it will overwrite the variables above.
    # (
    #    final_response,
    #    agg_prompt_tokens,
    #    agg_completion_tokens,
    # ) = await handle_responses_tool_workflow(client, req, input_items, response)
    
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

    await UserManager.update_token_usage(
        auth_info.user.userid, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    await QueryLogManager.log_query(
        auth_info.user.userid, model=req.model, prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, web_search_calls=len(sources) if sources else 0,
    )

    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    logger.info(
        f"[responses] done request_id={request_id} prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} total_ms={(time.monotonic() - t_start) * 1000:.0f}"
    )
    return model_response