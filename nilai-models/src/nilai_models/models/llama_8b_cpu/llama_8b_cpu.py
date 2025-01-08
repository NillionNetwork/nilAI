from llama_cpp import Llama
from nilai_common import ModelMetadata
from nilai_models.models.llama_model import LlamaCppModel


class Llama8BCpu(LlamaCppModel):
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

        # Initialize the base Model class with model metadata
        # Provides comprehensive information about the model
        super().__init__(
            model=Llama.from_pretrained(
                repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
                filename="Meta-Llama-3-8B-Instruct-Q5_K_M.gguf",
                n_threads=16,
                n_ctx=8 * 1024,
                verbose=False,
            ),
            metadata=ModelMetadata(
                id="Llama-3.1-8B-Instruct",  # Unique identifier
                name="Llama-3.1-8B-Instruct",  # Human-readable name
                version="1.0",  # Model version
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",  # Model creators
                license="Apache 2.0",  # Usage license
                source="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF",  # Model source
                supported_features=["chat_completion"],  # Capabilities
            ),
        )
    
    def load_models(self):
        """
        Load the model(s) required for the service.

        This method is called during model initialization to load the
        specific model(s) required for the service at service startup.
        """
        pass 

# Create and expose the FastAPI app for this Llama model
# - Calls get_app() from the base Model class
# - Allows easy integration with ASGI servers like uvicorn
app = Llama8BCpu().get_app()
