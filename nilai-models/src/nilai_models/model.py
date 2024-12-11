# nilai/models/model.py
import httpx # For making HTTP requests
from abc import ABC, abstractmethod  # Abstract Base Class to define interfaces
from contextlib import asynccontextmanager # For managing async context
import time  # For tracking uptime and time-related calculations
from fastapi import FastAPI  # Web framework for creating API endpoints
from nilai_common import (
    HealthCheckResponse,  # Custom response type for health checks
    ModelEndpoint,        # Endpoint information for model registration
    ModelMetadata,        # Metadata about the model
    ChatResponse,         # Response type for chat completions
    ChatRequest,          # Request type for chat interactions
)


class Model(ABC):
    """
    Abstract base class for AI models, providing a standardized interface 
    for model initialization, routing, and basic functionality.

    This class serves as a blueprint for creating different AI model 
    implementations with consistent API endpoints and behaviors.
    """

    def __init__(self, metadata: ModelMetadata):
        """
        Initialize the model with its metadata and tracking information.

        Args:
            metadata (ModelMetadata): Detailed information about the model.
        """
        # Store the model's metadata for later retrieval
        self.metadata = metadata
        
        # Record the start time for uptime tracking
        self._uptime = time.time()
        self.app = self.setup_app()


    def setup_app(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Load the model on the API
            async with httpx.AsyncClient() as client:
                url = "http://localhost:8000"
                data = ModelEndpoint(
                    url=url,
                    metadata=self.metadata
                )
                response = await client.post(f"http://localhost:8080/internal/endpoints", json=data.model_dump())
                print(response.text)
                # if response.status_code != 200:
                #     raise RuntimeError(f"Failed to connect to nilai-api: {response.text}")
            yield
            # Clean up resources if needed
            pass
        # Create a FastAPI application instance for the model
        self.app = FastAPI(lifespan=lifespan)
        self._setup_routes()
        return self.app
    
    def get_app(self) -> FastAPI:
        """
        Retrieve the FastAPI application instance for the model.

        Returns:
            FastAPI: The application instance associated with this model.
        """
        return self.app
    
    def _setup_routes(self):
        """
        Set up standard routes for the model's API.

        This method provides default implementations for common endpoints:
        - /chat: For initiating chat completions
        - /health: For checking the model's health status
        - /model: For retrieving model information

        Intended to be called during model initialization or overridden 
        by child classes to add custom routes.
        """
        # Chat completion endpoint
        @self.app.post("/chat")
        async def chat(req: ChatRequest) -> ChatResponse:
            return await self.chat_completion(req)

        # Health check endpoint
        @self.app.get("/health")
        async def health() -> HealthCheckResponse:
            return await self.health_check()

        # Model information endpoint
        @self.app.get("/model")
        async def model_info() -> ModelMetadata:
            print("model_info")
            return await self.model_info()
    
    @abstractmethod
    async def chat_completion(self, req: ChatRequest) -> ChatResponse:
        """
        Abstract method for generating chat completions.

        This method MUST be implemented by any child class.
        It defines the core interaction mechanism for the model.

        Args:
            req (ChatRequest): The chat request containing messages and parameters.

        Returns:
            ChatResponse: The generated response from the model.

        Raises:
            NotImplementedError: If not overridden by a child class.
        """
        pass

    async def model_info(self) -> ModelMetadata:
        """
        Retrieve the model's metadata.

        Returns:
            ModelMetadata: Detailed information about the model.
        """
        return self.metadata

    async def health_check(self) -> HealthCheckResponse:
        """
        Perform a basic health check for the model.

        Returns:
            HealthCheckResponse: Current health status and uptime.
        """
        return HealthCheckResponse(
            status="healthy",
            uptime=self.uptime,
        )

    @property
    def uptime(self):
        """
        Calculate and format the model's uptime in a human-readable format.

        Returns:
            str: A formatted string representing the model's uptime 
                 (e.g., "2 days, 3 hours, 45 minutes").
        """
        # Calculate total elapsed time since initialization
        elapsed_time = time.time() - self._uptime
        
        # Break down elapsed time into days, hours, minutes, seconds
        days, remainder = divmod(elapsed_time, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Prepare parts of the uptime string
        parts = []
        if days > 0:
            parts.append(f"{int(days)} days")
        if hours > 0:
            parts.append(f"{int(hours)} hours")
        if minutes > 0:
            parts.append(f"{int(minutes)} minutes")
        if seconds > 0:
            parts.append(f"{int(seconds)} seconds")

        # Join the parts into a readable string
        return ", ".join(parts)