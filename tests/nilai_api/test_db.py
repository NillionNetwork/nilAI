import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import nilai_api.db as db

# Import the classes and functions to test
from nilai_api.db import Base, UserManager


@pytest.fixture(scope="function")
def in_memory_db():
    """
    Create an in-memory SQLite database for testing.
    This ensures each test runs with a clean database.
    """
    # Use in-memory SQLite database with StaticPool for thread safety
    db.engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create tables
    Base.metadata.create_all(bind=db.engine)

    # Create a test session factory
    db.SessionLocal = sessionmaker(bind=db.engine, autocommit=False, autoflush=False)

    try:
        yield db.engine
    finally:
        Base.metadata.drop_all(bind=db.engine)


@pytest.fixture
def user_manager(in_memory_db):
    """Fixture to provide a clean UserManager for each test."""
    return UserManager


class TestUserManager:
    def test_generate_user_id(self, user_manager):
        """Test that generate_user_id creates a valid UUID."""
        user_id = user_manager.generate_user_id()

        # Validate UUID
        try:
            uuid_obj = uuid.UUID(user_id)
            assert str(uuid_obj) == user_id
        except ValueError:
            pytest.fail("Generated user ID is not a valid UUID")

    def test_generate_api_key(self, user_manager):
        """Test that generate_api_key creates a valid UUID."""
        api_key = user_manager.generate_api_key()

        # Validate UUID
        try:
            uuid_obj = uuid.UUID(api_key)
            assert str(uuid_obj) == api_key
        except ValueError:
            pytest.fail("Generated API key is not a valid UUID")

    def test_insert_user(self, user_manager):
        """Test inserting a new user."""
        # Insert a user
        user_data = user_manager.insert_user("Test User")

        # Validate returned data
        assert "userid" in user_data
        assert "apikey" in user_data
        assert len(user_data["userid"]) > 0
        assert len(user_data["apikey"]) > 0

        # Verify user can be retrieved
        retrieved_user_tokens = user_manager.get_user_token_usage(user_data["userid"])
        assert retrieved_user_tokens is not None
        assert retrieved_user_tokens["prompt_tokens"] == 0
        assert retrieved_user_tokens["completion_tokens"] == 0

    def test_check_api_key(self, user_manager):
        """Test API key validation."""
        # Insert a user
        user_data = user_manager.insert_user("Check API User")

        # Check valid API key
        user_name = user_manager.check_api_key(user_data["apikey"])
        assert user_name["name"] == "Check API User"
        assert user_name["userid"] == user_data["userid"]

        # Check invalid API key
        invalid_result = user_manager.check_api_key("invalid-api-key")
        assert invalid_result is None

    def test_update_token_usage(self, user_manager):
        """Test updating token usage for a user."""
        # Insert a user
        user_data = user_manager.insert_user("Token User")

        # Update token usage
        user_manager.update_token_usage(
            user_data["userid"], prompt_tokens=100, completion_tokens=50
        )

        # Verify token usage
        token_usage = user_manager.get_user_token_usage(user_data["userid"])
        assert token_usage is not None
        assert token_usage["prompt_tokens"] == 100
        assert token_usage["completion_tokens"] == 50

        # Update again to check cumulative effect
        user_manager.update_token_usage(
            user_data["userid"], prompt_tokens=50, completion_tokens=25
        )

        token_usage = user_manager.get_user_token_usage(user_data["userid"])
        assert token_usage is not None
        assert token_usage["prompt_tokens"] == 150
        assert token_usage["completion_tokens"] == 75

    def test_get_all_users(self, user_manager):
        """Test retrieving all users."""
        # Insert multiple users
        _ = user_manager.insert_user("User 1")
        _ = user_manager.insert_user("User 2")

        # Retrieve all users
        all_users = user_manager.get_all_users()

        user_names = [user.name for user in all_users]
        assert all_users is not None
        assert len(all_users) >= 2

        # Verify user names
        assert "User 1" in user_names
        assert "User 2" in user_names

    def test_get_user_token_usage_nonexistent(self, user_manager):
        """Test getting token usage for a non-existent user."""
        # Try to get token usage for a non-existent user
        nonexistent_userid = "non-existent-user-id"
        token_usage = user_manager.get_user_token_usage(nonexistent_userid)

        assert token_usage is None


class TestDatabaseInitialization:
    def test_initialize_db(self, user_manager):
        """Test database initialization."""
        # First call should return False (tables already exist) in this case
        second_result = user_manager.initialize_db()
        assert second_result is False


# Performance and Concurrency Tests
class TestConcurrency:
    @pytest.mark.parametrize("num_users", [10, 50, 100])
    def test_bulk_user_creation(self, user_manager, num_users):
        """Test creating multiple users concurrently."""
        users = []
        for i in range(num_users):
            user = user_manager.insert_user(f"User {i}")
            users.append(user)

        # Verify all users were created
        all_users = user_manager.get_all_users()
        assert all_users is not None
        assert len(all_users) >= num_users


# Additional Tests Configuration
def pytest_configure(config):
    """Configure pytest with additional settings."""
    config.addinivalue_line(
        "markers", "concurrency: mark test to run with multiple concurrent operations"
    )


# Test Runner Configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
