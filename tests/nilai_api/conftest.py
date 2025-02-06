import os

import pytest
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="session", autouse=True)
def redis_server():
    container = RedisContainer()
    container.start()
    host_ip = container.get_container_host_ip()
    host_port = container.get_exposed_port(6379)
    os.environ["REDIS_URL"] = f"redis://{host_ip}:{host_port}"
    return container
