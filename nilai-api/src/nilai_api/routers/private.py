# Fast API and serving
import asyncio
import logging
from base64 import b64encode
from typing import AsyncGenerator, Optional, Union, List, Tuple
from nilai_api.attestation import get_attestation_report
from nilai_api.handlers.nilrag import handle_nilrag
from nilai_api.handlers.web_search import handle_web_search

from fastapi import APIRouter, Body, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from nilai_api.auth import get_auth_info, AuthenticationInfo
from nilai_api.config import MODEL_CONCURRENT_RATE_LIMIT
from nilai_api.crypto import sign_message
from nilai_api.db.logs import QueryLogManager
from nilai_api.db.users import UserManager
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state

# Internal libraries
from nilai_common import (
    AttestationReport,
    ChatRequest,
    Message,
    ModelMetadata,
    SignedChatCompletion,
    Nonce,
    Source,
    Usage,
)
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(auth_info: AuthenticationInfo = Depends(get_auth_info)) -> Usage:
    """
    Retrieve the current token usage for the authenticated user.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Usage statistics for the user's token consumption

    ### Example
    ```python
    # Retrieves token usage for the logged-in user
    usage = await get_usage(user)
    ```
    """
    return Usage(
        prompt_tokens=auth_info.user.prompt_tokens,
        completion_tokens=auth_info.user.completion_tokens,
        total_tokens=auth_info.user.prompt_tokens + auth_info.user.completion_tokens,
        queries=auth_info.user.queries,  # type: ignore # FIXME this field is not part of Usage
    )


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(
    nonce: Optional[Nonce] = None,
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> AttestationReport:
    """
    Generate a cryptographic attestation report.

    - **attestation_request**: Attestation request containing a nonce
    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Attestation details for service verification

    ### Attestation Details
    - `verifying_key`: Public key for cryptographic verification
    - `cpu_attestation`: CPU environment verification
    - `gpu_attestation`: GPU environment verification

    ### Security Note
    Provides cryptographic proof of the service's integrity and environment.
    """

    attestation_report = await get_attestation_report(nonce)
    attestation_report.verifying_key = state.b64_public_key
    return attestation_report


@router.get("/v1/models", tags=["Model"])
async def get_models(
    auth_info: AuthenticationInfo = Depends(get_auth_info),
) -> List[ModelMetadata]:
    """
    List all available models in the system.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Dictionary of available models

    ### Example
    ```python
    # Retrieves list of available models
    models = await get_models(user)
    ```
    """
    return [endpoint.metadata for endpoint in (await state.models).values()]


async def chat_completion_concurrent_rate_limit(request: Request) -> Tuple[int, str]:
    body = await request.json()
    try:
        chat_request = ChatRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    key = f"chat:{chat_request.model}"
    limit = MODEL_CONCURRENT_RATE_LIMIT.get(
        chat_request.model, MODEL_CONCURRENT_RATE_LIMIT.get("default", 50)
    )
    return limit, key


async def chat_completion_web_search_rate_limit(request: Request) -> bool:
    """Extract web_search flag from request body for rate limiting."""
    body = await request.json()
    try:
        chat_request = ChatRequest(**body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request body")
    return getattr(chat_request, "web_search", False)


@router.post("/v1/chat/completions", tags=["Chat"], response_model=None)
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    ),
    _rate_limit=Depends(
        RateLimit(
            concurrent_extractor=chat_completion_concurrent_rate_limit,
            web_search_extractor=chat_completion_web_search_rate_limit,
        )
    ),
    auth_info: AuthenticationInfo = Depends(get_auth_info),
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
            {"role": "user", "content": "What's the latest news about AI?"}
        ],
    )
    response = await chat_completion(request, user)
    """

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
    model_url = endpoint.url + "/v1/"

    logger.info(
        f"Chat completion request for model {model_name} from user {auth_info.user.userid} on url: {model_url}"
    )

    client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")

    if req.nilrag:
        await handle_nilrag(req)

    messages = req.messages
    sources: Optional[List[Source]] = None
    if req.web_search:
        web_search_result = await handle_web_search(messages, model_name, client)
        messages = web_search_result.messages
        sources = web_search_result.sources

    if req.stream:
        # Forwarding Streamed Responses
        async def chat_completion_stream_generator() -> AsyncGenerator[str, None]:
            try:
                request_kwargs = {
                    "model": req.model,
                    "messages": messages,  # type: ignore
                    "stream": True,  # type: ignore
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
                    request_kwargs["tools"] = req.tools  # type: ignore

                response = await client.chat.completions.create(**request_kwargs)  # type: ignore
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
                )

            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                return

        # Return the streaming response
        return StreamingResponse(
            chat_completion_stream_generator(),
            media_type="text/event-stream",  # Ensure client interprets as Server-Sent Events
        )
    request_kwargs = {
        "model": req.model,
        "messages": messages,  # type: ignore
        "top_p": req.top_p,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.tools:
        request_kwargs["tools"] = req.tools  # type: ignore

    response = await client.chat.completions.create(**request_kwargs)  # type: ignore

    model_response = SignedChatCompletion(
        **response.model_dump(),
        signature="",
        sources=sources,
    )

    if model_response.usage is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model response does not contain usage statistics",
        )
    # Update token usage
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
    )

    # Sign the response
    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    return model_response
