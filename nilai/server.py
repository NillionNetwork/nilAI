from fastapi import FastAPI, Depends, HTTPException
from uuid import uuid4
import time
from base64 import b64encode
from transformers import pipeline
import torch 
from nilai.model import ChatRequest, ChatResponse, Message, Choice, Usage, AttestationResponse, Model
from nilai.cryptography import generate_key_pair, sign_message, verify_signature

app = FastAPI()

class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        # Initialize Hugging Face pipeline
        self.chat_pipeline = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

        self.models = [
            Model(
                id="meta-llama/Llama-3.2-1B-Instruct",
                name="Llama-3.2-1B-Instruct",
                description="Llama is a large language model trained on a mixture of supervised and unsupervised data.",
                author="Meta-Llama",
                license="MIT",
                source="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
            )
        ]
state = AppState()

@app.get("/v1/attestation/report")
async def get_attestation() -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="..."
    )

@app.get("/v1/models")
async def get_models() -> dict:
    return {
        "models": state.models
    }

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest) -> ChatResponse:
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="messages field is required")
    
    if not req.model:
        raise HTTPException(status_code=400, detail="model field is required")
    
    # Combine the conversation messages into a single prompt
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages]) + "\nassistant:"

    # Use Hugging Face pipeline to generate the response
    generated = state.chat_pipeline(prompt, max_length=1024, num_return_sequences=1)

    if not generated or len(generated) == 0:
        raise HTTPException(status_code=500, detail="Model returned no output")
    
    response_text = generated[0]['generated_text'].split("assistant:")[-1].strip()
    
    response = ChatResponse(
        id=f"chat-{uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(
            index=0,
            message=Message(
                role="assistant", 
                content=response_text
            ),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=len(prompt.split()),  # Approximate token count
            completion_tokens=len(response_text.split()),  # Approximate token count
            total_tokens=len(prompt.split()) + len(response_text.split())
        ),
        signature="",  # Will be filled later
    )
    
    # Sign the response
    response_json = response.json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()
    
    return response



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345)