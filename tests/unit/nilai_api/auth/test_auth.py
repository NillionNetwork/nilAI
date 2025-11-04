from datetime import datetime, timezone
import logging
from unittest.mock import MagicMock

from nilai_api.db.users import RateLimits
import pytest
from fastapi.security import HTTPAuthorizationCredentials

from nilai_api.config import CONFIG as config

# For these tests, we will use the api_key strategy
config.auth.auth_strategy = "api_key"


@pytest.fixture
def mock_validate_credential(mocker):
    """Fixture to mock validate_credential function."""
    return mocker.patch("nilai_api.auth.strategies.validate_credential")


@pytest.fixture
def mock_user_model():
    from nilai_api.db.users import UserModel

    mock = MagicMock(spec=UserModel)
    mock.name = "Test User"
    mock.user_id = "test-user-id"
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


@pytest.mark.asyncio
async def test_get_auth_info_valid_token(mock_validate_credential, mock_user_model):
    from nilai_api.auth import get_auth_info

    """Test get_auth_info with a valid token."""
    mock_validate_credential.return_value = mock_user_model
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="valid-token"
    )

    auth_info = await get_auth_info(credentials)
    print(auth_info)

    assert auth_info.user.user_id == "test-user-id", (
        f"Expected test-user-id but got {auth_info.user.user_id}"
    )


@pytest.mark.asyncio
async def test_get_auth_info_invalid_token(mock_validate_credential):
    from nilai_api.auth import get_auth_info
    from nilai_api.auth.common import AuthenticationError

    """Test get_auth_info with an invalid token."""
    mock_validate_credential.side_effect = AuthenticationError("Credential not found")
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer", credentials="invalid-token"
    )
    with pytest.raises(AuthenticationError) as exc_info:
        auth_infor = await get_auth_info(credentials)
        print(auth_infor)
    print(exc_info)
    assert "Credential not found" in str(exc_info.value.detail), (
        f"Expected 'Credential not found' but got {exc_info.value.detail}"
    )
