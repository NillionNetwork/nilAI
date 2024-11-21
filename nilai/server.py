from fastapi import FastAPI, Depends, HTTPException
from uuid import uuid4
import time
from base64 import b64encode

from nilai.model import ChatRequest, ChatResponse, Message, Choice, Usage, AttestationResponse
from nilai.cryptography import generate_key_pair, sign_message, verify_signature


app = FastAPI()

class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()

state = AppState()

@app.get("/v1/attestation/report")
async def get_attestation() -> AttestationResponse:
    return AttestationResponse(
        verifying_key=state.verifying_key,
        cpu_attestation="...",
        gpu_attestation="..."
    )

@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest) -> ChatResponse:
    if not req.messages or len(req.messages) == 0:
        raise HTTPException(status_code=400, detail="messages field is required")
    
    if not req.model:
        raise HTTPException(status_code=400, detail="model field is required")
    

    response = ChatResponse(
        id=f"chat-{uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[Choice(
            index=0,
            message=Message(
                role="assistant", 
                content="Sample response"
            ),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=27,
            completion_tokens=66,
            total_tokens=93
        ),
        signature="", # Will be filled later
    )
    
    # Sign the response
    response_json = response.model_dump_json()
    signature = sign_message(state.private_key, response_json)
    response.signature = b64encode(signature).decode()
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=12345)