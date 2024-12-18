import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from nilai_common import ChatRequest, ChatResponse, Message
from nilai_common.api_model import ChatCompletionChunk, ChoiceChunk, ChoiceChunkContent
from nilai_models.model import Model


class LlamaCppModel(Model):
    """
    A specific implementation of the Model base class for the Llama CPU model.

    This class provides:
    - Model initialization using llama_cpp
    - Chat completion functionality (with streaming support)
    - Metadata about the Llama model
    """

    def __init__(self, model, metadata, prefix="/models"):
        """
        Initialize the Llama 1B model:
        - Load pre-trained model using llama_cpp
        - Configured for CPU inference with 16 threads
        """
        self.model = model

        # Initialize model metadata
        super().__init__(metadata, prefix)

    async def chat_completion(
        self,
        req: ChatRequest = ChatRequest(
            model="bartowski/Llama-3.2-1B-Instruct-GGUF",
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
        prompt = [
            {
                "role": msg.role,
                "content": msg.content,
            }
            for msg in req.messages
        ]

        # Streaming response logic
        if req.stream:

            async def generate() -> AsyncGenerator[str, None]:
                try:
                    # Create a generator for the streamed output
                    loop = asyncio.get_event_loop()
                    output_generator = await loop.run_in_executor(
                        None,
                        lambda: self.model.create_chat_completion(
                            prompt,  # type: ignore
                            stream=True,
                            temperature=req.temperature if req.temperature else 0.2,
                            max_tokens=req.max_tokens,
                        ),
                    )
                    for output in output_generator:
                        # Extract delta content from output
                        choices = output.get("choices", [])  # type: ignore
                        if not choices or "delta" not in choices[0]:
                            continue  # Skip invalid chunks
                        # Extract delta
                        delta = choices[0]["delta"]
                        chunk = ChoiceChunk(
                            index=delta.get("index", 0),
                            delta=ChoiceChunkContent(content=delta.get("content", "")),
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
            generation: dict = self.model.create_chat_completion(prompt)  # type: ignore
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="The prompt size exceeds the maximum limit of 2048 tokens.",
            )
        if not generation or len(generation) == 0:
            raise ValueError("The model returned no output.")

        response = ChatResponse(
            signature="",
            **generation,
        )
        response.model = self.metadata.name
        return response
