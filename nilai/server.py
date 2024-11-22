from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from uuid import uuid4
import time
from base64 import b64encode
from transformers import pipeline
import torch
from pydantic import BaseModel
from nilai.model import ChatRequest, ChatResponse, Message, Choice, Usage, AttestationResponse, Model, HealthCheckResponse
from nilai.cryptography import generate_key_pair, sign_message, verify_signature

app = FastAPI()

# Application State Initialization
class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        self.chat_pipeline = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.2-1B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.models = [
            Model(
                id="meta-llama/Llama-3.2-1B-Instruct",
                name="Llama-3.2-1B-Instruct",
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",
                license="MIT",
                source="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
            )
        ]
        self._uptime = time.time()

    @property
    def uptime(self):
        elapsed_time = time.time() - self._uptime
        days, remainder = divmod(elapsed_time, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{int(days)} days")
        if hours > 0:
            parts.append(f"{int(hours)} hours")
        if minutes > 0:
            parts.append(f"{int(minutes)} minutes")
        if seconds > 0:
            parts.append(f"{int(seconds)} seconds")
        
        return ", ".join(parts)

state = AppState()

# Health Check Endpoint
@app.get("/v1/health")
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok", uptime=state.uptime)

# Model Information Endpoint
@app.get("/v1/model-info")
async def get_model_info():
    return {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "version": "1.0",
        "supported_features": ["text-completion", "chat"],
        "license": "MIT",
    }

# Attestation Report Endpoint
@app.get("/v1/attestation/report")
async def get_attestation() -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="...",
    )

# Available Models Endpoint
@app.get("/v1/models")
async def get_models() -> dict[str, list[Model]]:
    return {"models": state.models}

# Chat Completion Endpoint
@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest) -> ChatResponse:
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="The 'messages' field is required.")

    if not req.model:
        raise HTTPException(status_code=400, detail="The 'model' field is required.")

    # Combine messages into a single prompt
    print(req)
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
    prompt = [{
        "role": msg.role,
        "content": msg.content,
        } for msg in req.messages]  

    # Generate response
    generated = state.chat_pipeline(prompt, max_length=1024, num_return_sequences=1, truncation=True)
    print(generated)
    if not generated or len(generated) == 0:
        raise HTTPException(status_code=500, detail="The model returned no output.")

    response = generated[0]["generated_text"][-1]

    print(f"Prompt: {prompt}, Response: {response}")
    usage = Usage(
            prompt_tokens=sum(len(msg.content.split()) for msg in req.messages),
            completion_tokens=len(response['content'].split()),
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
                logprobs=None
            )
        ],
        usage=usage,
        signature="",  # Will be filled later
    )

    # Sign the response
    response_json = response.json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()

    return JSONResponse(content=response.dict(), headers={"Content-Type": "application/json"})

# Example Optional Tokenize Endpoint
@app.post("/v1/tokenize")
async def tokenize(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="The 'text' field is required.")
    tokens = text.split()  # Example tokenization logic
    return {"tokens": tokens}

# Main Server Runner
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=12345)