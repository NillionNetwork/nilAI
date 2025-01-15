from llama_cpp import Llama
from nilai_common import ModelMetadata
from nilai_models.models.llama_model import LlamaCppModel


class Llama1BCpu(LlamaCppModel):
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
                n_ctx=128 * 1024,
                verbose=False,
            ),
            metadata=ModelMetadata(
                id="Llama-3.2-1B-Instruct",
                name="Llama-3.2-1B-Instruct",
                version="1.0",
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",
                license="Apache 2.0",
                source="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
                supported_features=["chat_completion", "streaming"],  # Added streaming
            ),
        )

    def load_models(self):
        """
        Load the model(s) required for the service.

        This method is called during model initialization to load the
        specific model(s) required for the service at service startup.
        """
        pass


# FastAPI app instance
app = Llama1BCpu().get_app()
