import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from nilai_api.auth.strategies import api_key_strategy, nuc_strategy
from nilai_api.auth.common import AuthenticationInfo, PromptDocument
from nilai_api.db.users import RateLimits, UserModel


class TestAuthStrategies:
    """Test class for authentication strategies with nilDB integration"""

    @pytest.fixture
    def mock_user_model(self):
        """Mock UserModel fixture"""
        mock = MagicMock(spec=UserModel)
        mock.user_id = "test-user-id"
        mock.rate_limits = RateLimits().get_effective_limits().model_dump_json()
        mock.rate_limits_obj = RateLimits().get_effective_limits()
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
        with patch("nilai_api.auth.strategies.validate_credential") as mock_validate:
            mock_validate.return_value = mock_user_model

            result = await api_key_strategy("test-api-key")

            assert isinstance(result, AuthenticationInfo)
            assert result.token_rate_limit is None
            assert result.prompt_document is None
            mock_validate.assert_called_once_with("test-api-key", is_public=False)

    @pytest.mark.asyncio
    async def test_api_key_strategy_invalid_key(self):
        """Test API key authentication with invalid key"""
        from nilai_api.auth.common import AuthenticationError

        with patch("nilai_api.auth.strategies.validate_credential") as mock_validate:
            mock_validate.side_effect = AuthenticationError("Credential not found")

            with pytest.raises(AuthenticationError, match="Credential not found"):
                await api_key_strategy("invalid-key")

    @pytest.mark.asyncio
    async def test_nuc_strategy_existing_user_with_prompt_document(
        self, mock_user_model, mock_prompt_document
    ):
        """Test NUC authentication with existing user and prompt document"""
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate_nuc,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.validate_credential"
            ) as mock_validate_credential,
        ):
            mock_validate_nuc.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = mock_prompt_document
            mock_validate_credential.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.token_rate_limit is None
            assert result.prompt_document == mock_prompt_document
            mock_validate_credential.assert_called_once_with(
                "subscription_holder", is_public=True
            )

    @pytest.mark.asyncio
    async def test_nuc_strategy_new_user_with_token_limits(
        self, mock_prompt_document, mock_user_model
    ):
        """Test NUC authentication creating new user with token limits"""
        from nilai_api.auth.nuc_helpers.usage import TokenRateLimits, TokenRateLimit

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
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate_nuc,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.validate_credential"
            ) as mock_validate_credential,
        ):
            mock_validate_nuc.return_value = ("subscription_holder", "new_user_id")
            mock_get_rate_limit.return_value = mock_token_limits
            mock_get_prompt_doc.return_value = mock_prompt_document
            mock_validate_credential.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
            assert result.token_rate_limit == mock_token_limits
            assert result.prompt_document == mock_prompt_document
            mock_validate_credential.assert_called_once_with(
                "subscription_holder", is_public=True
            )

    @pytest.mark.asyncio
    async def test_nuc_strategy_no_prompt_document(self, mock_user_model):
        """Test NUC authentication when no prompt document is found"""
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate_nuc,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.validate_credential"
            ) as mock_validate_credential,
        ):
            mock_validate_nuc.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = None
            mock_validate_credential.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")

            assert isinstance(result, AuthenticationInfo)
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
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate_nuc,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.validate_credential"
            ) as mock_validate_credential,
        ):
            mock_validate_nuc.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.side_effect = Exception(
                "Prompt document extraction failed"
            )
            mock_validate_credential.return_value = mock_user_model

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
        mock_user_model.user_id = "test"
        mock_user_model.rate_limits = (
            RateLimits().get_effective_limits().model_dump_json()
        )
        mock_user_model.rate_limits_obj = RateLimits().get_effective_limits()

        # Test API key strategy
        with patch("nilai_api.auth.strategies.validate_credential") as mock_validate:
            mock_validate.return_value = mock_user_model
            result = await api_key_strategy("test-key")
            assert hasattr(result, "prompt_document")
            assert result.prompt_document is None

        # Test NUC strategy
        with (
            patch("nilai_api.auth.strategies.validate_nuc") as mock_validate_nuc,
            patch(
                "nilai_api.auth.strategies.get_token_rate_limit"
            ) as mock_get_rate_limit,
            patch(
                "nilai_api.auth.strategies.get_token_prompt_document"
            ) as mock_get_prompt_doc,
            patch(
                "nilai_api.auth.strategies.validate_credential"
            ) as mock_validate_credential,
        ):
            mock_validate_nuc.return_value = ("subscription_holder", "user_id")
            mock_get_rate_limit.return_value = None
            mock_get_prompt_doc.return_value = None
            mock_validate_credential.return_value = mock_user_model

            result = await nuc_strategy("nuc-token")
            assert hasattr(result, "prompt_document")
            assert result.prompt_document is None
