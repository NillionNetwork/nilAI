import uuid
import time
import torch
from fastapi import HTTPException
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput
from fastapi.responses import StreamingResponse
from nilai_common import ChatRequest, ChatResponse, Message, ModelMetadata, Usage, Choice
from nilai_models.model import Model

class Llama8BGpu(Model):
    """
    A specific implementation of the Model base class for the Llama 8B GPU model.
    """

    def __init__(self, load=True) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Attempted to initialize GPU model on non-GPU machine")
        super().__init__(
            ModelMetadata(
                id="Llama-3.1-8B-Instruct",  # Unique identifier
                name="Llama-3.1-8B-Instruct",  # Human-readable name
                version="1.0",  # Model version
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",  # Model creators
                license="Apache 2.0",  # Usage license
                source="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",  # Model source
                supported_features=["chat_completion"],  # Capabilities
            ),
        )

    def load_models(self):
        """
        Load the model(s) required for the service.

        This method is called during model initialization to load the
        specific model(s) required for the service at service startup.
        """
        self.model = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_memory_utilization=0.6,
            max_model_len=60624,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct"
        )

    async def chat_completion(
        self,
        req: ChatRequest = ChatRequest(
            # Default request with sample messages for documentation
            model="meta-llama/Llama-3.1-8B-Instruct",
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

        # Streaming response logic
        if req.stream:
            raise HTTPException(
                status_code=400, detail="Streaming is not supported for this model."
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


        # Non-streaming (regular) chat completion
        try:
            generation: RequestOutput = self.model.generate(
                prompt,  # type: ignore
                sampling_params=sampling_params,
            )  # type: ignore
            print(generation)
            generation = generation[0].outputs[0].text
            print(generation)
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
app = Llama8BGpu().get_app()
