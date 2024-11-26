# Fast API and serving
import time
from base64 import b64encode
from typing import Any, List
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException

from nilai.auth import get_user
from nilai.crypto import sign_message

# Internal libraries
from nilai.model import (
    AttestationResponse,
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Model,
    Usage,
)
from nilai.state import state

router = APIRouter()


# Model Information Endpoint
@router.get("/v1/model-info", tags=["Model"])
async def get_model_info(user: str = Depends(get_user)) -> dict:
    return {
        "model_name": state.models[0].name,
        "version": state.models[0].version,
        "supported_features": state.models[0].supported_features,
        "license": state.models[0].license,
    }


# Attestation Report Endpoint
@router.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation(user: str = Depends(get_user)) -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="...",
    )


# Available Models Endpoint
@router.get("/v1/models", tags=["Model"])
async def get_models(user: str = Depends(get_user)) -> dict[str, list[Model]]:
    return {"models": state.models}


# Chat Completion Endpoint
@router.post("/v1/chat/completions", tags=["Chat"])
def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model=state.models[0].name,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    ),
    user: str = Depends(get_user),
) -> ChatResponse:
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="The 'messages' field is required.")

    if not req.model:
        raise HTTPException(status_code=400, detail="The 'model' field is required.")

    # Combine messages into a single prompt
    print(req)
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
    prompt = [
        {
            "role": msg.role,
            "content": msg.content,
        }
        for msg in req.messages
    ]

    # Generate response
    generated: List[Any] = state.chat_pipeline(
        prompt, max_length=1024, num_return_sequences=1, truncation=True
    )  # type: ignore
    print(type(generated))
    if not generated or len(generated) == 0:
        raise HTTPException(status_code=500, detail="The model returned no output.")

    response = generated[0]["generated_text"][-1]
    print(f"Prompt: {prompt}, Response: {response}")
    usage = Usage(
        prompt_tokens=sum(len(msg.content.split()) for msg in req.messages),
        completion_tokens=len(response["content"].split()),
        total_tokens=0,
    )
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    response = ChatResponse(
        id=f"chat-{uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(**response),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage=usage,
        signature="",  # Will be filled later
    )

    # Sign the response
    response_json = response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()

    return response
