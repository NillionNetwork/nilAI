# Fast API and serving
import time
from base64 import b64encode
from uuid import uuid4

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from nilai.crypto import sign_message
# Internal libraries
from nilai.model import (AttestationResponse, ChatRequest, ChatResponse,
                         Choice, HealthCheckResponse, Message, Model, Usage)

app = FastAPI(
    title="NilAI",
    description="An AI model serving platform based on TEE",
    version="0.1.0",
    terms_of_service="https://nillion.com",
    contact={
        "name": "Nillion AI Support",
        # "url": "https://nillion.com",
        "email": "jose.cabrero@nillion.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    openapi_tags=[
        {
            "name": "Attestation",
            "description": "Retrieve attestation information",
        },
        {
            "name": "Chat",
            "description": "Chat completion endpoint",
        },
        {
            "name": "Health",
            "description": "Health check endpoint",
        },
        {
            "name": "Model",
            "description": "Model information",
        },
    ],
)


from nilai.state import AppState

state = AppState()


# Health Check Endpoint
@app.get("/v1/health", tags=["Health"])
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", uptime=state.uptime)


# Model Information Endpoint
@app.get("/v1/model-info", tags=["Model"])
async def get_model_info():
    return {
        "model_name": state.models[0].name,
        "version": state.models[0].version,
        "supported_features": state.models[0].supported_features,
        "license": state.models[0].license,
    }


# Attestation Report Endpoint
@app.get("/v1/attestation/report", tags=["Attestation"])
async def get_attestation() -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="...",
    )


# Available Models Endpoint
@app.get("/v1/models", tags=["Model"])
async def get_models() -> dict[str, list[Model]]:
    return {"models": state.models}


# Chat Completion Endpoint
@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completion(
    req: ChatRequest = Body(
        ChatRequest(
            model=state.models[0].name,
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        )
    )
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
    generated = state.chat_pipeline(
        prompt, max_length=1024, num_return_sequences=1, truncation=True
    )
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
    response_json = response.json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()

    return JSONResponse(
        content=response.dict(), headers={"Content-Type": "application/json"}
    )


if __name__ == "__main__":
    import uvicorn

    # Path to your SSL certificate and key files
    # SSL_CERTFILE = "/path/to/certificate.pem"  # Replace with your certificate file path
    # SSL_KEYFILE = "/path/to/private-key.pem"  # Replace with your private key file path

    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces
        port=12345,  # Use port 8443 for HTTPS
        # ssl_certfile=SSL_CERTFILE,
        # ssl_keyfile=SSL_KEYFILE,
    )
