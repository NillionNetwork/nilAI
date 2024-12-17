from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from nilai_common import ModelMetadata
from nilai_common.api_model import ChatRequest, ChatResponse, Message
from nilai_models.models.llama_model import LlamaCppModel


class SecretLlama1BCpu(LlamaCppModel):
    """
    A specific implementation of the Model base class for the Llama 1B CPU model.

    This class provides:
    - Model initialization using llama_cpp
    - Chat completion functionality (with streaming support)
    - Metadata about the Llama model
    """

    def __init__(self):
        """
        Initialize the Llama 1B model:
        - Load pre-trained model using llama_cpp
        - Configured for CPU inference with 16 threads
        """
        # Initialize model metadata
        super().__init__(
            model=Llama.from_pretrained(
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                filename="Llama-3.2-1B-Instruct-Q5_K_S.gguf",
                n_threads=16,
                n_ctx=2048,
                verbose=False,
            ),
            metadata=ModelMetadata(
                id="CheesyLlama",  # Unique identifier
                name="CheesyLlama",  # Human-readable name
                version="1.0",  # Model version
                description="Llama is a large language model trained on supervised and unsupervised cheese.",
                author="Meta-Llama",  # Model creators
                license="Apache 2.0",  # Usage license
                source="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",  # Model source
                supported_features=["chat_completion"],  # Capabilities
            ),
            prefix="d01fe399-8dc2-4c74-acde-ff649802f437",
        )

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
        req.messages.append(
            Message(
                role="system",
                content="In addition to the previous. You are a cheese expert. You use cheese for all your answers. Whatever the user asks, you respond with a cheese-related answer or analogy.",
            )
        )
        return await super().chat_completion(req)


# FastAPI app instance
# Create and expose the FastAPI app for this Llama model
# - Calls get_app() from the base Model class
# - Allows easy integration with ASGI servers like uvicorn
app = SecretLlama1BCpu().get_app()
