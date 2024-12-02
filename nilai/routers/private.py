# Fast API and serving
from base64 import b64encode

from fastapi import APIRouter, Body, Depends, HTTPException

from nilai.auth import get_user
from nilai.crypto import sign_message
from nilai.db import UserManager

# Internal libraries
from nilai.model import (
    AttestationResponse,
    ChatRequest,
    ChatResponse,
    Message,
    Model,
    Usage,
)
from nilai.state import state

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(user: dict = Depends(get_user)) -> Usage:
    """
    Retrieve the current token usage for the authenticated user.

    - **user**: Authenticated user information (through X-API-Key header)
    - **Returns**: Usage statistics for the user's token consumption

    ### Example
    ```python
    # Retrieves token usage for the logged-in user
    usage = await get_usage(user)
    ```
    """
    return Usage(**UserManager.get_token_usage(user["userid"]))


@router.get("/v1/model-info", tags=["Model"])
async def get_model_info(user: str = Depends(get_user)) -> dict:
    """
    Fetch detailed information about the current model.

    - **user**: Authenticated user (used for authorization through X-API-Key header)
    - **Returns**: Model metadata including name, version, and features

    ### Metadata Includes
    - `model_name`: Name of the current model
    - `version`: Model version
    - `supported_features`: List of model capabilities
    - `license`: Licensing information

    ### Example
    ```python
    # Retrieves current model information
    model_info = await get_model_info(user)
    ```
    """
    return {
        "model_name": state.models[0].name,
        "version": state.models[0].version,
        "supported_features": state.models[0].supported_features,
        "license": state.models[0].license,
    }


@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(user: dict = Depends(get_user)) -> AttestationResponse:
    """
    Generate a cryptographic attestation report.

    - **user**: Authenticated user information (through X-API-Key header)
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
        cpu_attestation="...",
        gpu_attestation="...",
    )


@router.get("/v1/models", tags=["Model"])
async def get_models(user: dict = Depends(get_user)) -> dict[str, list[Model]]:
    """
    List all available models in the system.

    - **user**: Authenticated user information (through X-API-Key header)
    - **Returns**: Dictionary of available models

    ### Example
    ```python
    # Retrieves list of available models
    models = await get_models(user)
    ```
    """
    return {"models": state.models}


@router.post("/v1/chat/completions", tags=["Chat"])
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model=state.models[0].name,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    ),
    user: dict = Depends(get_user),
) -> ChatResponse:
    """
    Generate a chat completion response from the AI model.

    - **req**: Chat completion request containing messages and model specifications
    - **user**: Authenticated user information
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
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="The 'messages' field is required.")
    if not req.model:
        raise HTTPException(status_code=400, detail="The 'model' field is required.")

    # Combine messages into a single prompt
    prompt = [
        {
            "role": msg.role,
            "content": msg.content,
        }
        for msg in req.messages
    ]

    # Generate response
    generated: dict = state.chat_pipeline.create_chat_completion(prompt)
    if not generated or len(generated) == 0:
        raise HTTPException(status_code=500, detail="The model returned no output.")

    response = ChatResponse(
        signature="",
        **generated,
    )

    response.model = req.model

    UserManager.update_token_usage(
        user["userid"],
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )
    # Sign the response
    response_json = response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()

    return response
