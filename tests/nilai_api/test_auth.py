from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


@pytest.fixture
def mock_user_manager(mocker):
    from nilai_api.db.users import UserManager

    """Fixture to mock UserManager methods."""
    mocker.patch.object(UserManager, "check_api_key")
    mocker.patch.object(UserManager, "update_last_activity")
    return UserManager


@pytest.fixture
def mock_user_model():
    from nilai_api.db.users import UserModel

    mock = MagicMock(spec=UserModel)
    mock.name = "Test User"
    mock.userid = "test-user-id"
    return mock


@pytest.mark.asyncio
async def test_get_user_valid_token(mock_user_manager, mock_user_model):
    from nilai_api.auth import get_user

    """Test get_user with a valid token."""
    mock_user_manager.check_api_key.return_value = mock_user_model
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )
    user = await get_user(credentials)
    assert user.name == mock_user_model.name
    assert user.userid == mock_user_model.userid


@pytest.mark.asyncio
async def test_get_user_invalid_token(mock_user_manager):
    from nilai_api.auth import get_user

    """Test get_user with an invalid token."""
    mock_user_manager.check_api_key.return_value = None
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )
    with pytest.raises(HTTPException) as exc_info:
        await get_user(credentials)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing or invalid API key"
