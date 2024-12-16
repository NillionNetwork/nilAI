import pytest
from fastapi.testclient import TestClient
from nilai_api.routers.public import router
from nilai_api.state import state
from nilai_common import HealthCheckResponse
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_health_check():
  """Test the health check endpoint."""
  response = client.get("/v1/health")
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "ok"
  assert "uptime" in data
  assert isinstance(data["uptime"], str)