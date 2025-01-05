import torch
from fastapi import HTTPException
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from nilai_common import ChatRequest, ChatResponse, Message, ModelMetadata
from nilai_models.model import Model


class Llama8BGpu(Model):
    """
    A specific implementation of the Model base class for the Llama 8B GPU model.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise ValueError("Attempted to initialize GPU model on non-GPU machine")

        self.model = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
        )

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
    ) -> ChatResponse:
        """
        Generate a chat completion using the Llama model.

        Args:
            req (ChatRequest): The chat request containing conversation messages.

        Returns:
            ChatResponse: The model's generated response.

        Raises:
            ValueError: If the model fails to generate a response.
        """
        if not req.messages or len(req.messages) == 0:
            raise HTTPException(
                status_code=400, detail="The 'messages' field is required."
            )
        if not req.model:
            raise HTTPException(
                status_code=400, detail="The 'model' field is required."
            )

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

        # Generate chat completion using the Llama model
        # - Converts messages into a model-compatible prompt
        # - type: ignore suppresses type checking for external library
        outputs = self.model.generate(
            prompt,
            sampling_params=self.sampling_params,
        )
        generation: str = outputs[0].outputs[0].text

        # TODO: place generation into a ChatResponse using nilai-common

        return generation


# Create and expose the FastAPI app for this Llama model
# - Calls get_app() from the base Model class
# - Allows easy integration with ASGI servers like uvicorn
app = Llama8BGpu().get_app()
