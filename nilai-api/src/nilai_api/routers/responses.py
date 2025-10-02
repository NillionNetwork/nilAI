import asyncio
import logging
import time
import uuid
from base64 import b64encode
from typing import AsyncGenerator, Optional, Union, List
from nilai_api.handlers.nilrag import handle_nilrag
from nilai_api.handlers.web_search import handle_web_search

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.crypto import sign_message
from nilai_api.db.logs import QueryLogManager
from nilai_api.db.users import UserManager
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state
from nilai_api.routers.common_rate_limit import (
    concurrent_rate_limit,
    web_search_rate_limit,
)

from nilai_api.handlers.nildb.handler import get_prompt_from_nildb

from nilai_common import (
    ChatRequest,
    ResponseRequest,
    MessageAdapter,
    SignedCompletion,
    Source,
)

from openai import AsyncOpenAI


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/responses", tags=["Responses"], response_model=None)
async def response_completion(
    req: ResponseRequest = Body(
        ResponseRequest(
            model="meta-llama/Llama-3.2-1B-Instruct",
            input="What is your name?",
            instructions="You are a helpful assistant.",
        )
    ),
    _rate_limit=Depends(
        RateLimit(
            concurrent_extractor=concurrent_rate_limit("response"),
            web_search_extractor=web_search_rate_limit("response"),
        )
    ),
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> Union[SignedCompletion, StreamingResponse]:
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info(f"[response] call start request_id={request_id}")

    model_name = req.model
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
    logger.info(f"[response] has_multimodal: {has_multimodal}")
    if has_multimodal and (not endpoint.metadata.multimodal_support or req.web_search):
        raise HTTPException(
            status_code=400,
            detail="Model does not support multimodal content, remove image inputs from request",
        )

    model_url = endpoint.url + "/v1/"

    logger.info(
        f"[response] start request_id={request_id} user={auth_info.user.userid} model={model_name} stream={req.stream} web_search={bool(req.web_search)} tools={bool(req.tools)} multimodal={has_multimodal} url={model_url}"
    )

    client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")

    chat_request = req.to_chat_request()

    if auth_info.prompt_document:
        try:
            nildb_prompt: str = await get_prompt_from_nildb(auth_info.prompt_document)
            chat_request.messages.insert(
                0, MessageAdapter.new_message(role="system", content=nildb_prompt)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Unable to extract prompt from nilDB: {str(e)}",
            )

    if chat_request.nilrag:
        logger.info(f"[response] nilrag start request_id={request_id}")
        t_nilrag = time.monotonic()
        await handle_nilrag(chat_request)
        logger.info(
            f"[response] nilrag done request_id={request_id} duration_ms={(time.monotonic() - t_nilrag) * 1000:.0f}"
        )

    messages = chat_request.messages
    sources: Optional[List[Source]] = None

    if chat_request.web_search:
        logger.info(f"[response] web_search start request_id={request_id}")
        t_ws = time.monotonic()
        web_search_result = await handle_web_search(chat_request, model_name, client)
        messages = web_search_result.messages
        sources = web_search_result.sources
        logger.info(
            f"[response] web_search done request_id={request_id} sources={len(sources) if sources else 0} duration_ms={(time.monotonic() - t_ws) * 1000:.0f}"
        )

    if req.stream:

        async def response_completion_stream_generator() -> AsyncGenerator[str, None]:
            try:
                logger.info(f"[response] stream start request_id={request_id}")
                t_call = time.monotonic()
                current_messages = messages
                request_kwargs = {
                    "model": req.model,
                    "messages": current_messages,
                    "stream": True,
                    "top_p": req.top_p,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "extra_body": {
                        "stream_options": {
                            "include_usage": True,
                            "continuous_usage_stats": True,
                        }
                    },
                }
                if req.tools:
                    request_kwargs["tools"] = req.tools

                response = await client.chat.completions.create(**request_kwargs)
                prompt_token_usage: int = 0
                completion_token_usage: int = 0
                async for chunk in response:
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                    await asyncio.sleep(0)

                    prompt_token_usage = (
                        chunk.usage.prompt_tokens if chunk.usage else prompt_token_usage
                    )
                    completion_token_usage = (
                        chunk.usage.completion_tokens
                        if chunk.usage
                        else completion_token_usage
                    )

                await UserManager.update_token_usage(
                    auth_info.user.userid,
                    prompt_tokens=prompt_token_usage,
                    completion_tokens=completion_token_usage,
                )
                await QueryLogManager.log_query(
                    auth_info.user.userid,
                    model=req.model,
                    prompt_tokens=prompt_token_usage,
                    completion_tokens=completion_token_usage,
                    web_search_calls=len(sources) if sources else 0,
                )
                logger.info(
                    f"[response] stream done request_id={request_id} prompt_tokens={prompt_token_usage} completion_tokens={completion_token_usage} duration_ms={(time.monotonic() - t_call) * 1000:.0f} total_ms={(time.monotonic() - t_start) * 1000:.0f}"
                )

            except Exception as e:
                logger.error(
                    f"[response] stream error request_id={request_id} error={e}"
                )
                return

        return StreamingResponse(
            response_completion_stream_generator(),
            media_type="text/event-stream",
        )

    current_messages = messages
    request_kwargs = {
        "model": req.model,
        "messages": current_messages,
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.tools:
        request_kwargs["tools"] = req.tools

    logger.info(f"[response] call start request_id={request_id}")
    logger.info(f"[response] call message: {current_messages}")
    t_call = time.monotonic()
    response = await client.chat.completions.create(**request_kwargs)
    logger.info(
        f"[response] call done request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
    )
    logger.info(f"[response] call response: {response}")

    model_response = SignedCompletion(
        **response.model_dump(),
        signature="",
        sources=sources,
    )
    logger.info(
        f"[response] model_response request_id={request_id} duration_ms={(time.monotonic() - t_call) * 1000:.0f}"
    )

    if model_response.usage is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model response does not contain usage statistics",
        )

    await UserManager.update_token_usage(
        auth_info.user.userid,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    await QueryLogManager.log_query(
        auth_info.user.userid,
        model=req.model,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
        web_search_calls=len(sources) if sources else 0,
    )

    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    logger.info(
        f"[response] done request_id={request_id} prompt_tokens={model_response.usage.prompt_tokens} completion_tokens={model_response.usage.completion_tokens} total_ms={(time.monotonic() - t_start) * 1000:.0f}"
    )
    return model_response
