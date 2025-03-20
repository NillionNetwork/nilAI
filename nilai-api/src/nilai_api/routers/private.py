# Fast API and serving
import logging
import os
import asyncio
from base64 import b64encode
from typing import AsyncGenerator, Union, List, Tuple
from nilai_api.handlers.nilrag import handle_nilrag

from fastapi import APIRouter, Body, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from nilai_api.auth import get_user
from nilai_api.config import MODEL_CONCURRENT_RATE_LIMIT
from nilai_api.crypto import sign_message
from nilai_api.db.users import UserManager, UserModel
from nilai_api.db.logs import QueryLogManager
from nilai_api.rate_limiting import RateLimit
from nilai_api.state import state
from openai import OpenAI, AsyncOpenAI

# Internal libraries
from nilai_common import (
    AttestationResponse,
    ChatRequest,
    SignedChatCompletion,
    Message,
    ModelMetadata,
    Usage,
)


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(user: UserModel = Depends(get_user)) -> Usage:
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
        prompt_tokens=user.prompt_tokens,
        completion_tokens=user.completion_tokens,
        total_tokens=user.prompt_tokens + user.completion_tokens,
        queries=user.queries,  # type: ignore # FIXME this field is not part of Usage
    )


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(user: UserModel = Depends(get_user)) -> AttestationResponse:
    """
    Generate a cryptographic attestation report.

    - **user**: Authenticated user information (through HTTP Bearer header)
    - **Returns**: Attestation details for service verification

    ### Attestation Details
    - `verifying_key`: Public key for cryptographic verification
    - `cpu_attestation`: CPU environment verification
    - `gpu_attestation`: GPU environment verification

    ### Security Note
    Provides cryptographic proof of the service's integrity and environment.
    """
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation=state.cpu_attestation,
        gpu_attestation=state.gpu_attestation,
    )


@router.get("/v1/models", tags=["Model"])
async def get_models(user: UserModel = Depends(get_user)) -> List[ModelMetadata]:
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
    logger.info(f"Retrieving models for user {user.userid} from pid {os.getpid()}")
    return [endpoint.metadata for endpoint in (await state.models).values()]
    # result = [Model(
    #     id = endpoint.metadata.id,
    #     created = 0,
    #     object = "model",
    #     owned_by = endpoint.metadata.author,
    #     data = endpoint.metadata.dict(),
    # ) for endpoint in (await state.models).values()]

    # return result[0]


async def chat_completion_concurrent_rate_limit(request: Request) -> Tuple[int, str]:
    body = await request.json()
    chat_request = ChatRequest(**body)
    key = f"chat:{chat_request.model}"
    try:
        limit = MODEL_CONCURRENT_RATE_LIMIT[chat_request.model]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid model name")
    return limit, key


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
    _=Depends(RateLimit(concurrent_extractor=chat_completion_concurrent_rate_limit)),
    user: UserModel = Depends(get_user),
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

    ### Response Components
    - Model-generated text completion
    - Token usage metrics
    - Cryptographically signed response for verification

    ### Processing Steps
    1. Validate input request parameters
    2. Prepare messages for model processing
    3. Generate AI model response
    4. Track and update token usage
    5. Cryptographically sign the response

    ### Potential HTTP Errors
    - **400 Bad Request**:
      - Missing messages list
      - No model specified
    - **500 Internal Server Error**:
      - Model fails to generate a response

    ### Example
    ```python
    # Generate a chat completion
    request = ChatRequest(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, who are you?"}
        ]
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
        f"Chat completion request for model {model_name} from user {user.userid} on url: {model_url}"
    )

    if req.nilrag:
        handle_nilrag(req)

    if req.stream:
        client = AsyncOpenAI(base_url=model_url, api_key="<not-needed>")

        # Forwarding Streamed Responses
        async def chat_completion_stream_generator() -> AsyncGenerator[str, None]:
            try:
                response = await client.chat.completions.create(
                    model=req.model,
                    messages=req.messages,  # type: ignore
                    stream=req.stream,  # type: ignore
                    top_p=req.top_p,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    tools=req.tools,  # type: ignore
                    extra_body={
                        "stream_options": {
                            "include_usage": True,
                            # "continuous_usage_stats": True,
                        }
                    },
                )  # type: ignore

                async for chunk in response:
                    if chunk.usage is not None:
                        await UserManager.update_token_usage(
                            user.userid,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                        await QueryLogManager.log_query(
                            user.userid,
                            model=req.model,
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                    else:
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                        await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                return

        # Return the streaming response
        return StreamingResponse(
            chat_completion_stream_generator(),
            media_type="text/event-stream",  # Ensure client interprets as Server-Sent Events
        )
    client = OpenAI(base_url=model_url, api_key="<not-needed>")
    response = client.chat.completions.create(
        model=req.model,
        messages=req.messages,  # type: ignore
        stream=req.stream,
        top_p=req.top_p,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        tools=req.tools,  # type: ignore
    )  # type: ignore

    model_response = SignedChatCompletion(
        **response.model_dump(),
        signature="",
    )
    if model_response.usage is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model response does not contain usage statistics",
        )
    # Update token usage
    await UserManager.update_token_usage(
        user.userid,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    await QueryLogManager.log_query(
        user.userid,
        model=req.model,
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    # Sign the response
    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    return model_response
