from nilai_common.api_model import Choice, Message, Usage
import pytest
from fastapi.testclient import TestClient
from nilai_models.model import Model
from nilai_common import ModelMetadata, ChatRequest, ChatResponse, HealthCheckResponse, ModelEndpoint
from tests import model_metadata, response, model_endpoint

class MyModel(Model):
    async def chat_completion(self, req: ChatRequest) -> ChatResponse:
        return response

@pytest.fixture
def model_instance():
    metadata = model_metadata
    return MyModel(metadata)

@pytest.fixture
def client(model_instance):
    return TestClient(model_instance.get_app())

def test_model_info(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "ABC"
    assert data["name"] == "ABC"
    assert data["description"] == "Description"

def test_health_check(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime" in data

def test_chat_completion(client):
    request = ChatRequest(
        model="ABC",
        messages=[
            Message(role="system", content="You are a helpful assistant"),
            Message(role="user", content="Hello, who are you?")
        ]
    )
    response = client.post("/v1/chat/completions", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["finish_reason"] == "test-finish-reason"
    assert data["usage"]["prompt_tokens"] == 100
    assert data["usage"]["completion_tokens"] == 50
    assert data["usage"]["total_tokens"] == 150
    assert data["signature"] == "test-signature"
