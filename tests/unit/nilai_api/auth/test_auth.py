from datetime import datetime, timezone
import logging
from unittest.mock import MagicMock

from nilai_api.db.users import RateLimits
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from nilai_api.config import CONFIG as config

# For these tests, we will use the api_key strategy
config.auth.auth_strategy = "api_key"


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
    mock.apikey = "test-api-key"
    mock.prompt_tokens = 0
    mock.completion_tokens = 0
    mock.queries = 0
    mock.signup_date = datetime.now(timezone.utc)
    mock.last_activity = datetime.now(timezone.utc)
    mock.rate_limits = RateLimits().get_effective_limits().model_dump_json()
    mock.rate_limits_obj = RateLimits().get_effective_limits()
    return mock


@pytest.fixture
def mock_user_data(mock_user_model):
    from nilai_api.db.users import UserData

    logging.info(mock_user_model.rate_limits)
    return UserData.from_sqlalchemy(mock_user_model)


@pytest.fixture
def mock_auth_info():
    from nilai_api.auth import AuthenticationInfo

    mock = MagicMock(spec=AuthenticationInfo)
    mock.user = mock_user_data
    return mock


@pytest.mark.asyncio
async def test_get_auth_info_valid_token(
    mock_user_manager, mock_auth_info, mock_user_model
):
    from nilai_api.auth import get_auth_info

    """Test get_auth_info with a valid token."""
    mock_user_manager.check_api_key.return_value = mock_user_model
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )

    auth_info = await get_auth_info(credentials)
    print(auth_info)
    assert auth_info.user.name == "Test User", (
        f"Expected Test User but got {auth_info.user.name}"
    )
    assert auth_info.user.userid == "test-user-id", (
        f"Expected test-user-id but got {auth_info.user.userid}"
    )


@pytest.mark.asyncio
async def test_get_auth_info_invalid_token(mock_user_manager):
    from nilai_api.auth import get_auth_info

    """Test get_auth_info with an invalid token."""
    mock_user_manager.check_api_key.return_value = None
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )
    with pytest.raises(HTTPException) as exc_info:
        auth_infor = await get_auth_info(credentials)
        print(auth_infor)
    print(exc_info)
    assert exc_info.value.status_code == 401, (
        f"Expected status code 401 but got {exc_info.value.status_code}"
    )
    assert exc_info.value.detail == "Missing or invalid API key", (
        f"Expected Missing or invalid API key but got {exc_info.value.detail}"
    )
