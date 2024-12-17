import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from nilai_api.auth import get_user
from nilai_api.db import UserManager


@pytest.fixture
def mock_user_manager(mocker):
    """Fixture to mock UserManager methods."""
    mocker.patch.object(UserManager, "check_api_key")
    return UserManager


def test_get_user_valid_token(mock_user_manager):
    """Test get_user with a valid token."""
    mock_user_manager.check_api_key.return_value = {
        "name": "Test User",
        "userid": "test-user-id",
    }
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )
    user = get_user(credentials)
    assert user["name"] == "Test User"
    assert user["userid"] == "test-user-id"


def test_get_user_invalid_token(mock_user_manager):
    """Test get_user with an invalid token."""
    mock_user_manager.check_api_key.return_value = None
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )
    with pytest.raises(HTTPException) as exc_info:
        get_user(credentials)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing or invalid API key"
