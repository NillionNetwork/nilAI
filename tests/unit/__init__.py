from openai.types.chat.chat_completion import ChoiceLogprobs
from openai.types.responses import Response as OpenAIResponse, ResponseUsage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from nilai_common import (
    SignedChatCompletion,
    SignedResponse,
    ModelEndpoint,
    ModelMetadata,
    Usage,
    Choice,
    MessageAdapter,
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
            message=MessageAdapter.new_completion_message(content="test-content"),
            finish_reason="stop",
            logprobs=ChoiceLogprobs(),
        )
    ],  # type: ignore
    usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    signature="test-signature",
)

responses_usage: ResponseUsage = ResponseUsage(
    input_tokens=100,
    input_tokens_details=InputTokensDetails(cached_tokens=0),
    output_tokens=50,
    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    total_tokens=150,
)

RESPONSES_RESPONSE: OpenAIResponse = OpenAIResponse(
    id="test-response-id",
    object="response",
    model="test-model",
    created_at=123456.0,
    status="completed",
    output=[],
    parallel_tool_calls=False,
    tool_choice="auto",
    tools=[],
    usage=responses_usage,
)

SIGNED_RESPONSES_RESPONSE: SignedResponse = SignedResponse(
    **RESPONSES_RESPONSE.model_dump(),
    signature="test-signature",
)
