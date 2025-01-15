import uuid
import time
import torch
import logging
import json
import asyncio
from typing import AsyncGenerator
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import HTTPException
from transformers import AutoTokenizer
from vllm import SamplingParams, RequestOutput
from fastapi.responses import StreamingResponse
from nilai_common import (
    ChatRequest,
    ChatResponse,
    Message,
    ModelMetadata,
    Usage,
    Choice,
    ChatCompletionChunk,
    ChoiceChunk,
    ChoiceChunkContent,
)
from nilai_models.model import Model


class Llama1BGpu(Model):
    """
    A specific implementation of the Model base class for the Llama 8B GPU model.
    """

    def __init__(self, load=True) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Attempted to initialize GPU model on non-GPU machine")
        super().__init__(
            ModelMetadata(
                id="Llama-3.2-1B-Instruct",  # Unique identifier
                name="Llama-3.2-1B-Instruct",  # Human-readable name
                version="1.0",  # Model version
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",  # Model creators
                license="Apache 2.0",  # Usage license
                source="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",  # Model source
                supported_features=["chat_completion"],  # Capabilities
            ),
        )

    def load_models(self):
        """
        Load the model(s) required for the service.

        This method is called during model initialization to load the
        specific model(s) required for the service at service startup.
        """
        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.2-1B-Instruct",
            gpu_memory_utilization=0.3,
            max_model_len=60624,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        )

    async def chat_completion(
        self,
        req: ChatRequest = ChatRequest(
            # Default request with sample messages for documentation
            model="meta-llama/Llama-3.2-1B-Instruct",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="What is your name?"),
            ],
        ),
    ) -> StreamingResponse | ChatResponse:
        """
        Generate a chat completion using the Llama model, with optional streaming.

        Args:
            req (ChatRequest): The chat request containing conversation messages.
            stream (bool): Whether to return a streamed response.

        Returns:
            ChatResponse or StreamingResponse: Either a full response or a streaming response.
        """
        if not req.messages or len(req.messages) == 0:
            raise HTTPException(
                status_code=400, detail="The 'messages' field is required."
            )
        if not req.model:
            raise HTTPException(
                status_code=400, detail="The 'model' field is required."
            )

        # Transform incoming messages to llama_cpp-compatible format
        conversation = [
            {
                "role": msg.role,  # Preserve message role (system/user/assistant)
                "content": msg.content,  # Preserve message content
            }
            for msg in req.messages
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=req.temperature if req.temperature else 0.7,
            top_p=req.top_p if req.top_p else 0.95,
            max_tokens=req.max_tokens if req.max_tokens else 1024,
        )

        if req.stream:

            async def generate() -> AsyncGenerator[str, None]:
                try:
                    previous_generated_len = 0
                    async for chunk in self.llm_engine.generate(
                        prompt,
                        sampling_params=sampling_params,
                        request_id=str(uuid.uuid4()),
                    ):  # Generate chunks
                        current_text = chunk.outputs[0].text

                        # Get only new tokens by slicing from previous length
                        new_text = current_text[previous_generated_len:]
                        # print(new_tokens, end='', flush=True)
                        previous_generated_len = len(current_text)
                        chunk = ChoiceChunk(
                            index=0,
                            delta=ChoiceChunkContent(content=new_text),
                        )  # Create a ChoiceChunk
                        completion_chunk = ChatCompletionChunk(choices=[chunk])
                        yield f"data: {completion_chunk.model_dump_json()}\n\n"  # Stream the chunk
                        await asyncio.sleep(0)  # Add an await to return inmediately

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logging.error("An error occurred: %s", str(e))
                    yield f"data: {json.dumps({'error': 'Internal error occurred!'})}\n\n"

            # Return the streamed response with headers
            return StreamingResponse(generate(), media_type="text/event-stream")
        # Non-streaming (regular) chat completion
        try:
            request_output: RequestOutput = None  # type: ignore
            async for chunk in self.llm_engine.generate(
                prompt, sampling_params=sampling_params, request_id=str(uuid.uuid4())
            ):
                request_output = chunk
            generation = request_output.outputs[0].text
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="The prompt size exceeds the maximum limit of 2048 tokens.",
            )
        if not generation or len(generation) == 0:
            raise ValueError("The model returned no output.")

        response = ChatResponse(
            signature="",
            id="chatcmpl-" + str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=generation),
                    finish_reason="complete",
                    logprobs=None,
                )
            ],
            usage=Usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(generation.split()),
                total_tokens=len(prompt.split()) + len(generation.split()),
            ),
        )
        return response


# Create and expose the FastAPI app for this Llama model
# - Calls get_app() from the base Model class
# - Allows easy integration with ASGI servers like uvicorn
app = Llama1BGpu().get_app()
