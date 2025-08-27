from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.chat import ChatCompletionMessage

from nilai_common import (
    SignedChatCompletion,
    ModelEndpoint,
    ModelMetadata,
    Usage,
    Choice,
)

model_metadata: ModelMetadata = ModelMetadata(
    id="ABC",  # Unique identifier
    name="ABC",  # Human-readable name
    version="1.0",  # Model version
    description="Description",
    author="Author",  # Model creators
    license="License",  # Usage license
    source="http://test-model-url",  # Model source
    supported_features=["supported_feature"],  # Capabilities
    tool_support=False,  # Whether the model supports tools
)

model_endpoint: ModelEndpoint = ModelEndpoint(
    url="http://test-model-url", metadata=model_metadata
)

response: SignedChatCompletion = SignedChatCompletion(
    id="test-id",
    object="chat.completion",
    model="test-model",
    created=123456,
    choices=[
        Choice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content="test-content"),
            finish_reason="stop",
            logprobs=ChoiceLogprobs(),
        )
    ],  # type: ignore
    usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    signature="test-signature",
)
