import pytest
from nilai_common.api_model import (
    ModelMetadata,
    BaseCompletionRequest,
    ChatRequest,
    ResponseRequest,
    MessageAdapter,
)
from pydantic import ValidationError


def test_model_metadata_creation():
    """Test creating a ModelMetadata instance."""
    metadata = ModelMetadata(
        name="Test Model",
        version="1.0",
        description="A test model",
        author="Test Author",
        license="MIT",
        source="https://example.com",
        supported_features=["feature1", "feature2"],
        tool_support=False,
    )

    assert metadata.id is not None
    assert metadata.name == "Test Model"
    assert metadata.version == "1.0"
    assert metadata.description == "A test model"
    assert metadata.author == "Test Author"
    assert metadata.license == "MIT"
    assert metadata.source == "https://example.com"
    assert metadata.supported_features == ["feature1", "feature2"]


def test_model_metadata_default_id():
    """Test that ModelMetadata generates a default UUID for id."""
    metadata = ModelMetadata(
        name="Test Model",
        version="1.0",
        description="A test model",
        author="Test Author",
        license="MIT",
        source="https://example.com",
        supported_features=["feature1", "feature2"],
        tool_support=False,
    )

    assert metadata.id is not None
    assert len(metadata.id) == 36  # UUID length


def test_model_metadata_invalid_data():
    """Test creating ModelMetadata with invalid data."""
    with pytest.raises(ValidationError):
        ModelMetadata(
            name="",
            version="",
            description="",
            author="",
            license="",
            source="",
            tool_support=False,
        )  # type: ignore


def test_base_completion_request_defaults():
    """Test BaseCompletionRequest with default values."""
    req = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    
    assert req.model == "test-model"
    assert req.temperature is None
    assert req.top_p is None
    assert req.max_tokens is None
    assert req.stream == False
    assert req.tools is None
    assert req.nilrag == {}
    assert req.web_search == False


def test_chat_request_with_parameters():
    """Test ChatRequest with all parameters."""
    req = ChatRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        stream=True,
        web_search=True,
    )
    
    assert req.temperature == 0.7
    assert req.top_p == 0.9
    assert req.max_tokens == 100
    assert req.stream == True
    assert req.web_search == True


def test_chat_request_temperature_validation():
    """Test ChatRequest temperature validation."""
    with pytest.raises(ValidationError):
        ChatRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=6.0,
        )


def test_chat_request_empty_messages():
    """Test ChatRequest with empty messages."""
    with pytest.raises(ValidationError):
        ChatRequest(
            model="test-model",
            messages=[],
        )


def test_response_request_string_input():
    """Test ResponseRequest with string input."""
    req = ResponseRequest(
        model="test-model",
        input="What is AI?",
        instructions="You are a helpful assistant.",
        temperature=0.5,
    )
    
    assert req.model == "test-model"
    assert req.input == "What is AI?"
    assert req.instructions == "You are a helpful assistant."
    assert req.temperature == 0.5


def test_response_request_messages_input():
    """Test ResponseRequest with messages input."""
    req = ResponseRequest(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )
    
    assert isinstance(req.input, list)
    assert len(req.input) == 1


def test_response_request_to_chat_request_string():
    """Test converting ResponseRequest with string input to ChatRequest."""
    req = ResponseRequest(
        model="test-model",
        input="Hello world",
        instructions="Be helpful",
        temperature=0.7,
        max_tokens=50,
    )
    
    chat_req = req.to_chat_request()
    
    assert isinstance(chat_req, ChatRequest)
    assert chat_req.model == "test-model"
    assert len(chat_req.messages) == 2
    assert chat_req.messages[0]["role"] == "system"
    assert chat_req.messages[0]["content"] == "Be helpful"
    assert chat_req.messages[1]["role"] == "user"
    assert chat_req.messages[1]["content"] == "Hello world"
    assert chat_req.temperature == 0.7
    assert chat_req.max_tokens == 50


def test_response_request_to_chat_request_messages():
    """Test converting ResponseRequest with messages input to ChatRequest."""
    req = ResponseRequest(
        model="test-model",
        input=[
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
        ],
        instructions="You are an expert.",
    )
    
    chat_req = req.to_chat_request()
    
    assert len(chat_req.messages) == 4
    assert chat_req.messages[0]["role"] == "system"
    assert chat_req.messages[0]["content"] == "You are an expert."


def test_response_request_to_chat_request_no_instructions():
    """Test converting ResponseRequest without instructions."""
    req = ResponseRequest(
        model="test-model",
        input="Hello",
    )
    
    chat_req = req.to_chat_request()
    
    assert len(chat_req.messages) == 1
    assert chat_req.messages[0]["role"] == "user"


def test_response_request_max_output_tokens_alias():
    """Test that max_output_tokens alias works."""
    req = ResponseRequest(
        model="test-model",
        input="Hello",
        max_output_tokens=200,
    )
    
    assert req.max_tokens == 200


def test_response_request_modalities_default():
    """Test ResponseRequest modalities default value."""
    req = ResponseRequest(
        model="test-model",
        input="Hello",
    )
    
    assert req.modalities == ["text"]


def test_response_request_has_multimodal_content_string():
    """Test has_multimodal_content with string input."""
    req = ResponseRequest(
        model="test-model",
        input="Hello",
    )
    
    assert req.has_multimodal_content() == False


def test_response_request_has_multimodal_content_text_messages():
    """Test has_multimodal_content with text-only messages."""
    req = ResponseRequest(
        model="test-model",
        input=[{"role": "user", "content": "Hello"}],
    )
    
    assert req.has_multimodal_content() == False


def test_response_request_has_multimodal_content_image():
    """Test has_multimodal_content with image content."""
    req = ResponseRequest(
        model="test-model",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                ],
            }
        ],
    )
    
    assert req.has_multimodal_content() == True


def test_base_completion_request_nilrag_default_factory():
    """Test that nilrag uses default_factory for safe defaults."""
    req1 = ChatRequest(model="test-1", messages=[{"role": "user", "content": "Hi"}])
    req2 = ChatRequest(model="test-2", messages=[{"role": "user", "content": "Hi"}])
    
    req1.nilrag["key"] = "value1"
    
    assert "key" not in req2.nilrag


def test_chat_request_adapted_messages():
    """Test ChatRequest adapted_messages property."""
    req = ChatRequest(
        model="test-model",
        messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ],
    )
    
    adapted = req.adapted_messages
    assert len(adapted) == 2
    assert adapted[0].role == "system"
    assert adapted[1].role == "user"


def test_chat_request_get_last_user_query():
    """Test ChatRequest get_last_user_query method."""
    req = ChatRequest(
        model="test-model",
        messages=[
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Second question"},
        ],
    )
    
    last_query = req.get_last_user_query()
    assert last_query == "Second question"


def test_chat_request_has_multimodal_content():
    """Test ChatRequest has_multimodal_content method."""
    req = ChatRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                ],
            }
        ],
    )
    
    assert req.has_multimodal_content() == True


def test_inheritance_hierarchy():
    """Test that ChatRequest and ResponseRequest inherit from BaseCompletionRequest."""
    chat_req = ChatRequest(model="test", messages=[{"role": "user", "content": "Hi"}])
    response_req = ResponseRequest(model="test", input="Hi")
    
    assert isinstance(chat_req, BaseCompletionRequest)
    assert isinstance(response_req, BaseCompletionRequest)
