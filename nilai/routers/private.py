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
    Message,
    Model,
    Usage,
)
from nilai.state import state
from nilai.db import UserManager

router = APIRouter()


@router.get("/v1/usage", tags=["Usage"])
async def get_usage(user: dict = Depends(get_user)) -> Usage:
    return Usage(**UserManager.get_token_usage(user["userid"]))


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
async def get_attestation(user: dict = Depends(get_user)) -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="...",
    )


# Available Models Endpoint
@router.get("/v1/models", tags=["Model"])
async def get_models(user: dict = Depends(get_user)) -> dict[str, list[Model]]:
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
    user: dict = Depends(get_user),
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
    generated: dict = state.chat_pipeline.create_chat_completion(prompt)
    if not generated or len(generated) == 0:
        raise HTTPException(status_code=500, detail="The model returned no output.")

    response = ChatResponse(
        signature="",
        **generated,
    )

    print(user)
    print(response.usage.prompt_tokens, response.usage.completion_tokens)
    UserManager.update_token_usage(
        user["userid"],
        input_tokens=response.usage.prompt_tokens,
        generated_tokens=response.usage.completion_tokens,
    )
    # Sign the response
    response_json = response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()

    return response
