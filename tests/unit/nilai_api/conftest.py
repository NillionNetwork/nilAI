import pytest
from unittest.mock import patch, MagicMock
from testcontainers.redis import RedisContainer
from nilai_api.config import CONFIG


@pytest.fixture(scope="session", autouse=True)
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid downloading models during tests."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch(
        "nilai_api.handlers.nilrag.SentenceTransformer", return_value=mock_model
    ):
        yield mock_model


@pytest.fixture(scope="session", autouse=True)
def redis_server():
    container = RedisContainer()
    container.start()
    host_ip = container.get_container_host_ip()
    host_port = container.get_exposed_port(6379)
    CONFIG.redis.url = f"redis://{host_ip}:{host_port}"
    return container
