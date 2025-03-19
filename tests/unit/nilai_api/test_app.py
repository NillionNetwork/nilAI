from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def mock_client(mocker):
    from nilai_api.app import app

    client = TestClient(app)
    return client


def test_openapi_schema(mock_client):
    response = mock_client.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()
