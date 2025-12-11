import pytest
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="session")
def redis_server():
    """Start a Redis container for testing."""
    container = RedisContainer()
    container.start()
    yield container
    container.stop()


@pytest.fixture
def redis_url(redis_server):
    """Get Redis host and port from the container."""
    host_ip = redis_server.get_container_host_ip()
    host_port = redis_server.get_exposed_port(6379)
    return f"redis://{host_ip}:{host_port}"
