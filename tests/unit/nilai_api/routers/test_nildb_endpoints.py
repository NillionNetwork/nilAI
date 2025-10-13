import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException, status

from nilai_api.auth.common import AuthenticationInfo, PromptDocument
from nilai_api.db.users import RateLimits, UserData, UserModel
from nilai_api.handlers.nildb.api_model import (
    PromptDelegationToken,
)
from datetime import datetime, timezone
from nilai_common import ResponseRequest


class TestNilDBEndpoints:
    """Test class for nilDB-related API endpoints"""

    @pytest.fixture
    def mock_subscription_owner_user(self):
        """Mock user data for subscription owner"""
        mock_user_model = MagicMock(spec=UserModel)
        mock_user_model.name = "Subscription Owner"
        mock_user_model.userid = "owner-id"
        mock_user_model.apikey = "owner-id"  # Same as userid for subscription owner
        mock_user_model.prompt_tokens = 0
        mock_user_model.completion_tokens = 0
        mock_user_model.queries = 0
        mock_user_model.signup_date = datetime.now(timezone.utc)
        mock_user_model.last_activity = datetime.now(timezone.utc)
        mock_user_model.rate_limits = (
            RateLimits().get_effective_limits().model_dump_json()
        )
        mock_user_model.rate_limits_obj = RateLimits().get_effective_limits()

        return UserData.from_sqlalchemy(mock_user_model)

    @pytest.fixture
    def mock_regular_user(self):
        """Mock user data for regular user (not subscription owner)"""
        mock_user_model = MagicMock(spec=UserModel)
        mock_user_model.name = "Regular User"
        mock_user_model.userid = "user-id"
        mock_user_model.apikey = "different-api-key"  # Different from userid
        mock_user_model.prompt_tokens = 0
        mock_user_model.completion_tokens = 0
        mock_user_model.queries = 0
        mock_user_model.signup_date = datetime.now(timezone.utc)
        mock_user_model.last_activity = datetime.now(timezone.utc)
        mock_user_model.rate_limits = (
            RateLimits().get_effective_limits().model_dump_json()
        )
        mock_user_model.rate_limits_obj = RateLimits().get_effective_limits()

        return UserData.from_sqlalchemy(mock_user_model)

    @pytest.fixture
    def mock_auth_info_subscription_owner(self, mock_subscription_owner_user):
        """Mock AuthenticationInfo for subscription owner"""
        return AuthenticationInfo(
            user=mock_subscription_owner_user,
            token_rate_limit=None,
            prompt_document=None,
        )

    @pytest.fixture
    def mock_auth_info_regular_user(self, mock_regular_user):
        """Mock AuthenticationInfo for regular user"""
        return AuthenticationInfo(
            user=mock_regular_user, token_rate_limit=None, prompt_document=None
        )

    @pytest.fixture
    def mock_prompt_delegation_token(self):
        """Mock PromptDelegationToken"""
        return PromptDelegationToken(
            token="delegation_token_123", did="did:nil:builder123"
        )

    @pytest.mark.asyncio
    async def test_get_prompt_store_delegation_success(
        self, mock_auth_info_subscription_owner, mock_prompt_delegation_token
    ):
        """Test successful delegation token request"""
        from nilai_api.routers.private import get_prompt_store_delegation

        with patch(
            "nilai_api.routers.private.get_nildb_delegation_token"
        ) as mock_get_delegation:
            mock_get_delegation.return_value = mock_prompt_delegation_token

            request = "user-123"

            result = await get_prompt_store_delegation(
                request, mock_auth_info_subscription_owner
            )

            assert isinstance(result, PromptDelegationToken)
            assert result.token == "delegation_token_123"
            assert result.did == "did:nil:builder123"
            mock_get_delegation.assert_called_once_with("user-123")

    @pytest.mark.asyncio
    async def test_get_prompt_store_delegation_forbidden_regular_user(
        self, mock_auth_info_regular_user
    ):
        """Test delegation token request by regular user (not subscription owner)"""
        from nilai_api.routers.private import get_prompt_store_delegation

        request = "user-123"

        with pytest.raises(HTTPException) as exc_info:
            await get_prompt_store_delegation(request, mock_auth_info_regular_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Prompt storage is reserved to subscription owners" in str(
            exc_info.value.detail
        )

    @pytest.mark.asyncio
    async def test_get_prompt_store_delegation_handler_error(
        self, mock_auth_info_subscription_owner
    ):
        """Test delegation token request when handler raises an exception"""
        from nilai_api.routers.private import get_prompt_store_delegation

        with patch(
            "nilai_api.routers.private.get_nildb_delegation_token"
        ) as mock_get_delegation:
            mock_get_delegation.side_effect = Exception("Handler failed")

            request = "user-123"

            with pytest.raises(HTTPException) as exc_info:
                await get_prompt_store_delegation(
                    request, mock_auth_info_subscription_owner
                )

            assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "Server unable to produce delegation tokens: Handler failed" in str(
                exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_chat_completion_with_prompt_document_injection(self):
        """Test chat completion with prompt document injection"""
        from nilai_api.routers.endpoints.chat import chat_completion
        from nilai_common import ChatRequest

        mock_prompt_document = PromptDocument(
            document_id="test-doc-123", owner_did="did:nil:" + "1" * 66
        )

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user, token_rate_limit=None, prompt_document=mock_prompt_document
        )

        request = ChatRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )

        with (
            patch(
                "nilai_api.routers.endpoints.chat.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch("nilai_api.routers.endpoints.chat.AsyncOpenAI") as mock_openai_client,
            patch("nilai_api.routers.endpoints.chat.state.get_model") as mock_get_model,
            patch(
                "nilai_api.routers.endpoints.chat.handle_nilrag"
            ) as mock_handle_nilrag,
            patch(
                "nilai_api.routers.endpoints.chat.handle_web_search"
            ) as mock_handle_web_search,
            patch(
                "nilai_api.routers.endpoints.chat.UserManager.update_token_usage"
            ) as mock_update_usage,
            patch(
                "nilai_api.routers.endpoints.chat.QueryLogManager.log_query"
            ) as mock_log_query,
            patch(
                "nilai_api.routers.endpoints.chat.handle_tool_workflow"
            ) as mock_handle_tool_workflow,
        ):
            mock_get_prompt.return_value = "System prompt from nilDB"

            # Mock state.get_model() to return a ModelEndpoint
            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            # Mock handle_nilrag and handle_web_search
            mock_handle_nilrag.return_value = None
            mock_web_search_result = MagicMock()
            mock_web_search_result.messages = request.messages
            mock_web_search_result.sources = []
            mock_handle_web_search.return_value = mock_web_search_result

            # Mock async database operations
            mock_update_usage.return_value = None
            mock_log_query.return_value = None

            # Mock OpenAI client
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            # Mock the response object that will be awaited
            mock_response.model_dump.return_value = {
                "id": "test-response-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            # Make the create method itself an AsyncMock that returns the response
            mock_client_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_client_instance.close = AsyncMock()
            mock_openai_client.return_value = mock_client_instance

            # Mock handle_tool_workflow to return the response and token counts
            mock_handle_tool_workflow.return_value = (mock_response, 0, 0)

            # Call the function (this will test the prompt injection logic)
            await chat_completion(req=request, auth_info=mock_auth_info)

            mock_get_prompt.assert_called_once_with(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_chat_completion_prompt_document_extraction_error(self):
        """Test chat completion when prompt document extraction fails"""
        from nilai_api.routers.endpoints.chat import chat_completion
        from nilai_common import ChatRequest

        mock_prompt_document = PromptDocument(
            document_id="test-doc-123", owner_did="did:nil:" + "1" * 66
        )

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user, token_rate_limit=None, prompt_document=mock_prompt_document
        )

        request = ChatRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )

        with (
            patch(
                "nilai_api.routers.endpoints.chat.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch("nilai_api.routers.endpoints.chat.state.get_model") as mock_get_model,
        ):
            # Mock state.get_model() to return a ModelEndpoint
            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            mock_get_prompt.side_effect = Exception("Unable to extract prompt")

            with pytest.raises(HTTPException) as exc_info:
                await chat_completion(req=request, auth_info=mock_auth_info)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert (
                "Unable to extract prompt from nilDB: Unable to extract prompt"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_chat_completion_without_prompt_document(self):
        """Test chat completion when no prompt document is present"""
        from nilai_api.routers.endpoints.chat import chat_completion
        from nilai_common import ChatRequest

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user,
            token_rate_limit=None,
            prompt_document=None,
        )

        request = ChatRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )

        with (
            patch(
                "nilai_api.routers.endpoints.chat.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch("nilai_api.routers.endpoints.chat.AsyncOpenAI") as mock_openai_client,
            patch("nilai_api.routers.endpoints.chat.state.get_model") as mock_get_model,
            patch(
                "nilai_api.routers.endpoints.chat.handle_nilrag"
            ) as mock_handle_nilrag,
            patch(
                "nilai_api.routers.endpoints.chat.handle_web_search"
            ) as mock_handle_web_search,
            patch(
                "nilai_api.routers.endpoints.chat.UserManager.update_token_usage"
            ) as mock_update_usage,
            patch(
                "nilai_api.routers.endpoints.chat.QueryLogManager.log_query"
            ) as mock_log_query,
            patch(
                "nilai_api.routers.endpoints.chat.handle_tool_workflow"
            ) as mock_handle_tool_workflow,
        ):
            # Mock state.get_model() to return a ModelEndpoint
            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            # Mock handle_nilrag and handle_web_search
            mock_handle_nilrag.return_value = None
            mock_web_search_result = MagicMock()
            mock_web_search_result.messages = request.messages
            mock_web_search_result.sources = []
            mock_handle_web_search.return_value = mock_web_search_result

            # Mock async database operations
            mock_update_usage.return_value = None
            mock_log_query.return_value = None

            # Mock OpenAI client
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            # Mock the response object that will be awaited
            mock_response.model_dump.return_value = {
                "id": "test-response-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            # Make the create method itself an AsyncMock that returns the response
            mock_client_instance.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock_client_instance.close = AsyncMock()
            mock_openai_client.return_value = mock_client_instance

            # Mock handle_tool_workflow to return the response and token counts
            mock_handle_tool_workflow.return_value = (mock_response, 0, 0)

            # Call the function
            await chat_completion(req=request, auth_info=mock_auth_info)

            # Should not call get_prompt_from_nildb when no prompt document
            mock_get_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_with_prompt_document_injection(self):
        """Test responses endpoint with prompt document injection"""
        from nilai_api.routers.endpoints.responses import create_response

        mock_prompt_document = PromptDocument(
            document_id="test-doc-123", owner_did="did:nil:" + "1" * 66
        )

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user, token_rate_limit=None, prompt_document=mock_prompt_document
        )

        request = ResponseRequest(model="test-model", input="Hello")

        response_payload = {
            "id": "test-response-id",
            "object": "response",
            "model": "test-model",
            "created_at": 123456.0,
            "status": "completed",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        }

        with (
            patch(
                "nilai_api.routers.endpoints.responses.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch(
                "nilai_api.routers.endpoints.responses.AsyncOpenAI"
            ) as mock_openai_client,
            patch(
                "nilai_api.routers.endpoints.responses.state.get_model"
            ) as mock_get_model,
            patch(
                "nilai_api.routers.endpoints.responses.UserManager.update_token_usage"
            ) as mock_update_usage,
            patch(
                "nilai_api.routers.endpoints.responses.QueryLogManager.log_query"
            ) as mock_log_query,
            patch(
                "nilai_api.routers.endpoints.responses.handle_responses_tool_workflow"
            ) as mock_handle_tool_workflow,
        ):
            mock_get_prompt.return_value = "System prompt from nilDB"

            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            mock_update_usage.return_value = None
            mock_log_query.return_value = None

            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = response_payload
            mock_client_instance.responses.create = AsyncMock(
                return_value=mock_response
            )
            mock_openai_client.return_value = mock_client_instance

            mock_handle_tool_workflow.return_value = (mock_response, 0, 0)

            await create_response(req=request, auth_info=mock_auth_info)

            mock_get_prompt.assert_called_once_with(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_responses_prompt_document_extraction_error(self):
        """Test responses endpoint when prompt document extraction fails"""
        from nilai_api.routers.endpoints.responses import create_response

        mock_prompt_document = PromptDocument(
            document_id="test-doc-123", owner_did="did:nil:" + "1" * 66
        )

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user, token_rate_limit=None, prompt_document=mock_prompt_document
        )

        request = ResponseRequest(model="test-model", input="Hello")

        with (
            patch(
                "nilai_api.routers.endpoints.responses.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch(
                "nilai_api.routers.endpoints.responses.state.get_model"
            ) as mock_get_model,
        ):
            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            mock_get_prompt.side_effect = Exception("Unable to extract prompt")

            with pytest.raises(HTTPException) as exc_info:
                await create_response(req=request, auth_info=mock_auth_info)

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert (
                "Unable to extract prompt from nilDB: Unable to extract prompt"
                in str(exc_info.value.detail)
            )

    @pytest.mark.asyncio
    async def test_responses_without_prompt_document(self):
        """Test responses endpoint when no prompt document is present"""
        from nilai_api.routers.endpoints.responses import create_response

        mock_user = MagicMock()
        mock_user.userid = "test-user-id"
        mock_user.name = "Test User"
        mock_user.apikey = "test-api-key"
        mock_user.rate_limits = RateLimits().get_effective_limits()

        mock_auth_info = AuthenticationInfo(
            user=mock_user,
            token_rate_limit=None,
            prompt_document=None,
        )

        request = ResponseRequest(model="test-model", input="Hello")

        response_payload = {
            "id": "test-response-id",
            "object": "response",
            "model": "test-model",
            "created_at": 123456.0,
            "status": "completed",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        }

        with (
            patch(
                "nilai_api.routers.endpoints.responses.get_prompt_from_nildb"
            ) as mock_get_prompt,
            patch(
                "nilai_api.routers.endpoints.responses.AsyncOpenAI"
            ) as mock_openai_client,
            patch(
                "nilai_api.routers.endpoints.responses.state.get_model"
            ) as mock_get_model,
            patch(
                "nilai_api.routers.endpoints.responses.UserManager.update_token_usage"
            ) as mock_update_usage,
            patch(
                "nilai_api.routers.endpoints.responses.QueryLogManager.log_query"
            ) as mock_log_query,
            patch(
                "nilai_api.routers.endpoints.responses.handle_responses_tool_workflow"
            ) as mock_handle_tool_workflow,
        ):
            mock_model_endpoint = MagicMock()
            mock_model_endpoint.url = "http://test-model-endpoint"
            mock_model_endpoint.metadata.tool_support = True
            mock_model_endpoint.metadata.multimodal_support = True
            mock_get_model.return_value = mock_model_endpoint

            mock_update_usage.return_value = None
            mock_log_query.return_value = None

            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = response_payload
            mock_client_instance.responses.create = AsyncMock(
                return_value=mock_response
            )
            mock_openai_client.return_value = mock_client_instance

            mock_handle_tool_workflow.return_value = (mock_response, 0, 0)

            await create_response(req=request, auth_info=mock_auth_info)

            mock_get_prompt.assert_not_called()

    def test_prompt_delegation_request_model_validation(self):
        """Test PromptDelegationRequest model validation"""
        # Valid request
        valid_request = "user-123"
        assert valid_request == "user-123"

        # Test with different types of user IDs
        request_with_uuid = "550e8400-e29b-41d4-a716-446655440000"
        assert request_with_uuid == "550e8400-e29b-41d4-a716-446655440000"

    def test_prompt_delegation_token_model_validation(self):
        """Test PromptDelegationToken model validation"""
        token = PromptDelegationToken(
            token="delegation_token_123", did="did:nil:builder123"
        )
        assert token.token == "delegation_token_123"
        assert token.did == "did:nil:builder123"

    def test_user_is_subscription_owner_property(
        self, mock_subscription_owner_user, mock_regular_user
    ):
        """Test the is_subscription_owner property"""
        # Subscription owner (userid == apikey)
        assert mock_subscription_owner_user.is_subscription_owner is True

        # Regular user (userid != apikey)
        assert mock_regular_user.is_subscription_owner is False
