from fastapi import HTTPException
from llama_cpp import Llama
from nilai_common import ChatRequest, ChatResponse, Message, ModelMetadata
from nilai_models.model import Model


class Llama1BCpu(Model):
    """
    A specific implementation of the Model base class for the Llama 1B CPU model.

    This class provides:
    - Model initialization using llama_cpp
    - Chat completion functionality
    - Metadata about the Llama model
    """

    def __init__(self):
        """
        Initialize the Llama 1B model:
        1. Load the pre-trained model using llama_cpp
        2. Set up model metadata

        Configuration details:
        - Uses a specific quantized model from Hugging Face
        - Configured for CPU inference
        - Uses 16 threads for improved performance on CPU
        """
        # Load the pre-trained Llama model
        # - repo_id: Source of the model
        # - filename: Specific model file (quantized version)
        # - n_threads: Number of CPU threads for inference
        # - verbose: Disable detailed logging
        self.model = Llama.from_pretrained(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q5_K_S.gguf",
            n_threads=16,
            n_ctx=2048,
            verbose=False,
        )

        # Initialize the base Model class with model metadata
        # Provides comprehensive information about the model
        super().__init__(
            ModelMetadata(
                id="Llama-3.2-1B-Instruct",  # Unique identifier
                name="Llama-3.2-1B-Instruct",  # Human-readable name
                version="1.0",  # Model version
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",  # Model creators
                license="Apache 2.0",  # Usage license
                source="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",  # Model source
                supported_features=["chat_completion"],  # Capabilities
            ),
        )

    async def chat_completion(
        self,
        req: ChatRequest = ChatRequest(
            # Default request with sample messages for documentation
            model="bartowski/Llama-3.2-1B-Instruct-GGUF",
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
        # Transform incoming messages into a format compatible with llama_cpp
        # Extracts role and content from each message
        prompt = [
            {
                "role": msg.role,  # Preserve message role (system/user/assistant)
                "content": msg.content,  # Preserve message content
            }
            for msg in req.messages
        ]

        # Generate chat completion using the Llama model
        # - Converts messages into a model-compatible prompt
        # - type: ignore suppresses type checking for external library
        generation: dict = self.model.create_chat_completion(prompt)  # type: ignore

        # Validate model output
        if not generation or len(generation) == 0:
            raise ValueError("The model returned no output.")

        # Convert model generation to ChatResponse
        # - Uses dictionary unpacking to convert generation results
        # - Signature left empty (can be extended for tracking/verification)
        response = ChatResponse(
            signature="",
            **generation,
        )
        response.model = self.metadata.name  # Set model identifier
        return response


# Create and expose the FastAPI app for this Llama model
# - Calls get_app() from the base Model class
# - Allows easy integration with ASGI servers like uvicorn
app = Llama1BCpu().get_app()
