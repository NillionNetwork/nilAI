import pytest
from ..nilai_api import MockUserDatabase


# Pytest fixture for MockUserDatabase
@pytest.fixture
def mock_db():
    """Fixture to create a fresh MockUserDatabase for each test."""
    return MockUserDatabase()


# Test functions using async/await
@pytest.mark.asyncio
async def test_insert_user(mock_db):
    """Test user insertion functionality."""
    user = await mock_db.insert_user("Test User", "test@example.com")

    assert "user_id" in user
    assert "apikey" in user
    assert len(mock_db.users) == 1


@pytest.mark.asyncio
async def test_check_api_key(mock_db):
    """Test API key validation."""
    user = await mock_db.insert_user("Test User", "test@example.com")

    valid_check = await mock_db.check_api_key(user["apikey"])
    assert valid_check is not None
    assert valid_check["name"] == "Test User"

    invalid_check = await mock_db.check_api_key("invalid-key")
    assert invalid_check is None


@pytest.mark.asyncio
async def test_token_usage(mock_db):
    """Test token usage tracking."""
    user = await mock_db.insert_user("Test User", "test@example.com")

    await mock_db.update_token_usage(user["user_id"], 50, 20)

    token_usage = await mock_db.get_token_usage(user["user_id"])
    assert token_usage["prompt_tokens"] == 50
    assert token_usage["completion_tokens"] == 20
    assert token_usage["queries"] == 1


@pytest.mark.asyncio
async def test_query_logging(mock_db):
    """Test query logging functionality."""
    user = await mock_db.insert_user("Test User", "test@example.com")

    await mock_db.log_query(user["user_id"], "test-model", 10, 15)

    assert len(mock_db.query_logs) == 1
    log_entry = list(mock_db.query_logs.values())[0]
    assert log_entry["user_id"] == user["user_id"]
    assert log_entry["model"] == "test-model"
