# Fast API and serving
import logging
import os
import asyncio
from base64 import b64encode
from typing import AsyncGenerator, Union

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from nilai_api.auth import get_user
from nilai_api.crypto import sign_message
from nilai_api.db import UserManager
from nilai_api.state import state

# Internal libraries
from nilai_common import (
    AttestationResponse,
    ChatRequest,
    ChatResponse,
    Message,
    ModelMetadata,
    Usage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(user: dict = Depends(get_user)) -> Usage:
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
    return Usage(**UserManager.get_token_usage(user["userid"]))  # type: ignore


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(user: dict = Depends(get_user)) -> AttestationResponse:
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
async def get_models(user: dict = Depends(get_user)) -> list[ModelMetadata]:
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
    logger.info(f"Retrieving models for user {user['userid']} from pid {os.getpid()}")
    return [endpoint.metadata for endpoint in (await state.models).values()]


@router.post("/v1/chat/completions", tags=["Chat"], response_model=None)
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model="Llama-3.2-1B-Instruct",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    ),
    user: dict = Depends(get_user),
) -> Union[ChatResponse, StreamingResponse]:
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
        model="Llama-3.2-1B-Instruct",
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
            status_code=400, detail="Invalid model name, check /v1/models for options"
        )

    model_url = endpoint.url

    if req.stream:
        # Forwarding Streamed Responses
        async def stream_response() -> AsyncGenerator[str, None]:
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{model_url}/v1/chat/completions",
                        json=req.model_dump(),
                        timeout=60.0,
                    ) as response:
                        response.raise_for_status()  # Raise an error for invalid status codes

                        # Process the streamed response chunks
                        async for chunk in response.aiter_lines():
                            if chunk:  # Skip empty lines
                                yield f"{chunk}\n"
                                await asyncio.sleep(
                                    0
                                )  # Add an await to return inmediately
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code,
                    detail=e.response.json().get("detail", str(e)),
                )
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Error connecting to model service: {str(e)}",
                )

        # Return the streaming response
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",  # Ensure client interprets as Server-Sent Events
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{model_url}/v1/chat/completions", json=req.model_dump(), timeout=60.0
            )
            response.raise_for_status()
            model_response = ChatResponse.model_validate_json(response.content)
    except httpx.HTTPStatusError as e:
        # Forward the original error from the model
        raise HTTPException(
            status_code=e.response.status_code,
            detail=e.response.json().get("detail", str(e)),
        )
    except httpx.RequestError as e:
        # Handle connection/timeout errors
        raise HTTPException(
            status_code=503, detail=f"Error connecting to model service: {str(e)}"
        )

    # Update token usage
    UserManager.update_token_usage(
        user["userid"],
        prompt_tokens=model_response.usage.prompt_tokens,
        completion_tokens=model_response.usage.completion_tokens,
    )

    # Sign the response
    response_json = model_response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    model_response.signature = b64encode(signature).decode()

    return model_response
