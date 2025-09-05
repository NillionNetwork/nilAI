import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException

from nilai_api.auth.strategies import api_key_strategy, jwt_strategy, nuc_strategy
from nilai_api.auth.common import AuthenticationInfo, PromptDocument
from nilai_api.db.users import UserModel


class TestAuthStrategies:
    """Test class for authentication strategies with nilDB integration"""

    @pytest.fixture
    def mock_user_model(self):
        """Mock UserModel fixture"""
        mock = MagicMock(spec=UserModel)
        mock.name = "Test User"
        mock.userid = "test-user-id"
        mock.apikey = "test-api-key"
        mock.prompt_tokens = 0
        mock.completion_tokens = 0
        mock.queries = 0
        mock.signup_date = datetime.now(timezone.utc)
        mock.last_activity = datetime.now(timezone.utc)
        mock.ratelimit_day = 1000
        mock.ratelimit_hour = 1000
        mock.ratelimit_minute = 1000
        mock.web_search_ratelimit_day = 100
        mock.web_search_ratelimit_hour = 50
        mock.web_search_ratelimit_minute = 10
        return mock

    @pytest.fixture
    def mock_prompt_document(self):
        """Mock PromptDocument fixture"""
        return PromptDocument(
            document_id="test-document-123", owner_did=f"did:nil:{'1' * 66}"
        )

    @pytest.mark.asyncio
    async def test_api_key_strategy_success(self, mock_user_model):
        """Test successful API key authentication"""
        with patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check:
            mock_check.return_value = mock_user_model

            result = await api_key_strategy("test-api-key")

            assert isinstance(result, AuthenticationInfo)
            assert result.user.name == "Test User"
            assert result.token_rate_limit is None
            assert result.prompt_document is None

    @pytest.mark.asyncio
    async def test_api_key_strategy_invalid_key(self):
        """Test API key authentication with invalid key"""
        with patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check:
            mock_check.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await api_key_strategy("invalid-key")

            assert exc_info.value.status_code == 401
            assert "Missing or invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_jwt_strategy_existing_user(self, mock_user_model):
        """Test JWT authentication with existing user"""
        with (
            patch("nilai_api.auth.strategies.validate_jwt") as mock_validate,
            patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check,
        ):
            mock_jwt_result = MagicMock()
            mock_jwt_result.user_address = "test-address"
            mock_jwt_result.pub_key = "test-pub-key"
            mock_validate.return_value = mock_jwt_result
            mock_check.return_value = mock_user_model

            result = await jwt_strategy("jwt-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.user.name == "Test User"
            assert result.token_rate_limit is None
            assert result.prompt_document is None

    @pytest.mark.asyncio
    async def test_jwt_strategy_new_user(self):
        """Test JWT authentication creating new user"""
        with (
            patch("nilai_api.auth.strategies.validate_jwt") as mock_validate,
            patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check,
            patch(
                "nilai_api.auth.strategies.UserManager.insert_user_model"
            ) as mock_insert,
        ):
            mock_jwt_result = MagicMock()
            mock_jwt_result.user_address = "new-user-address"
            mock_jwt_result.pub_key = "new-user-pub-key"
            mock_validate.return_value = mock_jwt_result
            mock_check.return_value = None
            mock_insert.return_value = None

            result = await jwt_strategy("jwt-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.token_rate_limit is None
            assert result.prompt_document is None
            mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_nuc_strategy_existing_user_with_prompt_document(
        self, mock_user_model, mock_prompt_document
    ):
        """Test NUC authentication with existing user and prompt document"""
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.UserManager.check_user"
            ) as mock_check_user,
        ):
            mock_validate.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = mock_prompt_document
            mock_check_user.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.user.name == "Test User"
            assert result.token_rate_limit is None
            assert result.prompt_document == mock_prompt_document

    @pytest.mark.asyncio
    async def test_nuc_strategy_new_user_with_token_limits(self, mock_prompt_document):
        """Test NUC authentication creating new user with token limits"""
        from nuc_helpers.usage import TokenRateLimits, TokenRateLimit

        mock_token_limits = TokenRateLimits(
            limits=[
                TokenRateLimit(
                    signature="test-signature",
                    expires_at=datetime.now(timezone.utc) + timedelta(days=1),
                    usage_limit=1,
                )
            ]
        )

        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.UserManager.check_user"
            ) as mock_check_user,
            patch(
                "nilai_api.auth.strategies.UserManager.insert_user_model"
            ) as mock_insert,
        ):
            mock_validate.return_value = ("subscription_holder", "new_user_id")
            mock_get_rate_limit.return_value = mock_token_limits
            mock_get_prompt_doc.return_value = mock_prompt_document
            mock_check_user.return_value = None
            mock_insert.return_value = None

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.token_rate_limit == mock_token_limits
            assert result.prompt_document == mock_prompt_document
            mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_nuc_strategy_no_prompt_document(self, mock_user_model):
        """Test NUC authentication when no prompt document is found"""
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.UserManager.check_user"
            ) as mock_check_user,
        ):
            mock_validate.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = None
            mock_check_user.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.user.name == "Test User"
            assert result.token_rate_limit is None
            assert result.prompt_document is None

    @pytest.mark.asyncio
    async def test_nuc_strategy_validation_error(self):
        """Test NUC authentication when validation fails"""
        with patch("nilai_api.auth.strategies.validate_nuc") as mock_validate:
            mock_validate.side_effect = Exception("Invalid NUC token")

            with pytest.raises(Exception, match="Invalid NUC token"):
                await nuc_strategy("invalid-nuc-token")

    @pytest.mark.asyncio
    async def test_nuc_strategy_get_prompt_document_error(self, mock_user_model):
        """Test NUC authentication when get_token_prompt_document fails"""
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.UserManager.check_user"
            ) as mock_check_user,
        ):
            mock_validate.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.side_effect = Exception(
                "Prompt document extraction failed"
            )
            mock_check_user.return_value = mock_user_model

            # The function should let the exception bubble up or handle it gracefully
            # Based on the diff, it looks like it doesn't catch exceptions from get_token_prompt_document
            with pytest.raises(Exception, match="Prompt document extraction failed"):
                await nuc_strategy("nuc-token")

    @pytest.mark.asyncio
    async def test_all_strategies_return_authentication_info_with_prompt_document_field(
        self,
    ):
        """Test that all strategies return AuthenticationInfo with prompt_document field"""
        mock_user_model = MagicMock(spec=UserModel)
        mock_user_model.name = "Test"
        mock_user_model.userid = "test"
        mock_user_model.apikey = "test"
        mock_user_model.prompt_tokens = 0
        mock_user_model.completion_tokens = 0
        mock_user_model.queries = 0
        mock_user_model.signup_date = datetime.now(timezone.utc)
        mock_user_model.last_activity = datetime.now(timezone.utc)
        mock_user_model.ratelimit_day = 1000
        mock_user_model.ratelimit_hour = 1000
        mock_user_model.ratelimit_minute = 1000
        mock_user_model.web_search_ratelimit_day = 100
        mock_user_model.web_search_ratelimit_hour = 50
        mock_user_model.web_search_ratelimit_minute = 10

        # Test API key strategy
        with patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check:
            mock_check.return_value = mock_user_model
            result = await api_key_strategy("test-key")
            assert hasattr(result, "prompt_document")
            assert result.prompt_document is None

        # Test JWT strategy
        with (
            patch("nilai_api.auth.strategies.validate_jwt") as mock_validate,
            patch("nilai_api.auth.strategies.UserManager.check_api_key") as mock_check,
        ):
            mock_jwt_result = MagicMock()
            mock_jwt_result.user_address = "test-address"
            mock_jwt_result.pub_key = "test-pub-key"
            mock_validate.return_value = mock_jwt_result
            mock_check.return_value = mock_user_model

            result = await jwt_strategy("jwt-token")
            assert hasattr(result, "prompt_document")
            assert result.prompt_document is None

        # Test NUC strategy
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.UserManager.check_user"
            ) as mock_check_user,
        ):
            mock_validate.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = None
            mock_check_user.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")
            assert hasattr(result, "prompt_document")
            assert result.prompt_document is None
