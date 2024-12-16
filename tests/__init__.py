from nilai_common import ChatResponse, Choice, Message, ModelEndpoint, ModelMetadata, Usage

model_metadata: ModelMetadata = ModelMetadata(
    id="ABC",  # Unique identifier
    name="ABC",  # Human-readable name
    version="1.0",  # Model version
    description="Description",
    author="Author",  # Model creators
    license="License",  # Usage license
    source="http://test-model-url",  # Model source
    supported_features=["supported_feature"],  # Capabilities
)

model_endpoint: ModelEndpoint = ModelEndpoint(url="http://test-model-url", metadata=model_metadata)

response: ChatResponse = ChatResponse(
    id="test-id",
    object="test-object",
    model="test-model",
    created=123456,
    choices=[
        Choice(
            index=0,
            message=Message(role="test-role", content="test-content"),
            finish_reason="test-finish-reason",
            logprobs={"test-logprobs": "test-value"},
        )
    ],
    usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    signature="test-signature",
)