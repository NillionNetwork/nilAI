"""
Integration test configuration and fixtures.

This module provides shared fixtures for integration tests that use real database connections
via testcontainers.
"""

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from nilai_api.db import Base
from nilai_api.config import CONFIG


@pytest.fixture(scope="session")
def postgres_container():
    """Start a PostgreSQL container for the test session."""
    with PostgresContainer(
        image="postgres:15-alpine",
        username="testuser",
        password="testpass",
        dbname="testdb",
        port=5432,
    ) as postgres:
        yield postgres


@pytest.fixture(scope="function")
def database_config(postgres_container):
    """Configure the database connection for tests."""
    # Store original config values
    original_config = {
        "host": CONFIG.database.host,
        "port": CONFIG.database.port,
        "user": CONFIG.database.user,
        "password": CONFIG.database.password,
        "db": CONFIG.database.db,
    }

    # Update CONFIG to use test database
    CONFIG.database.host = postgres_container.get_container_host_ip()
    CONFIG.database.port = postgres_container.get_exposed_port(5432)
    CONFIG.database.user = "testuser"
    CONFIG.database.password = "testpass"
    CONFIG.database.db = "testdb"

    # Clear any existing engine/session maker to force re-creation with new config
    import nilai_api.db

    nilai_api.db._engine = None
    nilai_api.db._SessionLocal = None

    yield CONFIG

    # Restore original config
    CONFIG.database.host = original_config["host"]
    CONFIG.database.port = original_config["port"]
    CONFIG.database.user = original_config["user"]
    CONFIG.database.password = original_config["password"]
    CONFIG.database.db = original_config["db"]

    # Clear test engine/session maker
    nilai_api.db._engine = None
    nilai_api.db._SessionLocal = None


@pytest_asyncio.fixture(scope="function")
async def setup_database_schema(database_config):
    """Create database tables for each test function."""
    from nilai_api.db import get_engine

    engine = get_engine()

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Clean up tables after test
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception:
        # Ignore cleanup errors
        pass

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def clean_database(setup_database_schema):
    """Clean database data for tests."""
    # Setup is done by setup_database_schema
    yield

    # No cleanup needed - tables are dropped by setup_database_schema
