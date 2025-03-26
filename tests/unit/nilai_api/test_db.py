import pytest
from ..nilai_api import MockUserDatabase


# Pytest fixture for MockUserDatabase
@pytest.fixture
def mock_db():
    """Fixture to create a fresh MockUserDatabase for each test."""
    return MockUserDatabase()


# Test functions using sync wrappers for async methods
def test_insert_user(mock_db, event_loop):
    """Test user insertion functionality."""
    user = event_loop.run_until_complete(
        mock_db.insert_user("Test User", "test@example.com")
    )

    assert "userid" in user
    assert "apikey" in user
    assert len(mock_db.users) == 1


def test_check_api_key(mock_db, event_loop):
    """Test API key validation."""
    user = event_loop.run_until_complete(
        mock_db.insert_user("Test User", "test@example.com")
    )

    valid_check = event_loop.run_until_complete(mock_db.check_api_key(user["apikey"]))
    assert valid_check is not None
    assert valid_check["name"] == "Test User"

    invalid_check = event_loop.run_until_complete(mock_db.check_api_key("invalid-key"))
    assert invalid_check is None


def test_token_usage(mock_db, event_loop):
    """Test token usage tracking."""
    user = event_loop.run_until_complete(
        mock_db.insert_user("Test User", "test@example.com")
    )

    event_loop.run_until_complete(mock_db.update_token_usage(user["userid"], 50, 20))

    token_usage = event_loop.run_until_complete(mock_db.get_token_usage(user["userid"]))
    assert token_usage["prompt_tokens"] == 50
    assert token_usage["completion_tokens"] == 20
    assert token_usage["queries"] == 1


def test_query_logging(mock_db, event_loop):
    """Test query logging functionality."""
    user = event_loop.run_until_complete(
        mock_db.insert_user("Test User", "test@example.com")
    )

    event_loop.run_until_complete(
        mock_db.log_query(user["userid"], "test-model", 10, 15)
    )

    assert len(mock_db.query_logs) == 1
    log_entry = list(mock_db.query_logs.values())[0]
    assert log_entry["userid"] == user["userid"]
    assert log_entry["model"] == "test-model"
