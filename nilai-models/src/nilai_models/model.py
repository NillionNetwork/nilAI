# nilai/models/model.py
import asyncio  # For making HTTP requests
import logging
import time  # For tracking uptime and time-related calculations
from abc import ABC, abstractmethod  # Abstract Base Class to define interfaces
from contextlib import asynccontextmanager  # For managing async context
from typing import Union  # For type hinting

from fastapi import FastAPI  # Web framework for creating API endpoints
from fastapi.responses import StreamingResponse
from nilai_common import ChatRequest  # Request type for chat interactions
from nilai_common import ChatResponse  # Response type for chat completions
from nilai_common import HealthCheckResponse  # Custom response type for health checks
from nilai_common import ModelEndpoint  # Endpoint information for model registration
from nilai_common import ModelMetadata  # Metadata about the model
from nilai_common import (  # Model service discovery and host settings
    SETTINGS,
    ModelServiceDiscovery,
)

logger = logging.getLogger(__name__)


class Model(ABC):
    """
    Abstract base class for AI models, providing a standardized interface
    for model initialization, routing, and basic functionality.

    This class serves as a blueprint for creating different AI model
    implementations with consistent API endpoints and behaviors.
    """

    def __init__(self, metadata: ModelMetadata, prefix="/models"):
        """
        Initialize the model with its metadata and tracking information.

        Args:
            metadata (ModelMetadata): Detailed information about the model.
            prefix (str): Optional prefix for hiding the model behind a subpath.
        """
        # Store the model's metadata for later retrieval
        self.metadata = metadata
        self.url = f"http://{SETTINGS["host"]}:{SETTINGS["port"]}"
        self.endpoint = ModelEndpoint(url=self.url, metadata=self.metadata)
        # Record the start time for uptime tracking
        self._uptime = time.time()
        self.app = self.setup_app()
        self.prefix = prefix

    def setup_app(self):
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            discovery_service = None
            keep_alive_task = None

            try:
                # Initialize discovery service
                discovery_service = ModelServiceDiscovery(
                    host=SETTINGS["etcd_host"], port=SETTINGS["etcd_port"]
                )

                # Validate model metadata
                if not self.endpoint or not self.endpoint.metadata:
                    raise ValueError("Invalid model metadata")

                # Register model and start keepalive
                logger.info(f"Registering model: {self.endpoint.metadata.id}")
                lease = await discovery_service.register_model(
                    self.endpoint, self.prefix
                )
                keep_alive_task = asyncio.create_task(
                    discovery_service.keep_alive(lease),
                    name=f"keepalive_{self.endpoint.metadata.id}",
                )

                logger.info(f"Model registered successfully: {self.endpoint}")
                yield

            except Exception as e:
                logger.error(f"Failed to initialize model service: {e}")
                raise
            finally:
                # Cleanup
                if keep_alive_task and not keep_alive_task.done():
                    keep_alive_task.cancel()
                    try:
                        await keep_alive_task
                    except asyncio.CancelledError:
                        pass

                if discovery_service:
                    try:
                        await discovery_service.unregister_model(
                            self.endpoint.metadata.id
                        )
                        logger.info(f"Model unregistered: {self.endpoint.metadata.id}")
                    except Exception as e:
                        logger.error(f"Error unregistering model: {e}")

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

        @self.app.post("/v1/chat/completions", response_model=None)
        async def chat(req: ChatRequest) -> Union[ChatResponse, StreamingResponse]:
            return await self.chat_completion(req)

        # Health check endpoint
        @self.app.get("/v1/health")
        async def health() -> HealthCheckResponse:
            return await self.health_check()

        # Model information endpoint
        @self.app.get("/v1/models")
        async def model_info() -> ModelMetadata:
            return await self.model_info()

    @abstractmethod
    async def chat_completion(
        self, req: ChatRequest
    ) -> ChatResponse | StreamingResponse:
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
