import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from nilai_api.handlers.nildb.handler import (
    get_nildb_delegation_token,
    get_prompt_from_nildb,
    create_builder_client,
    create_user_client,
)
from nilai_api.handlers.nildb.api_model import PromptDelegationToken
from nilai_api.auth.common import PromptDocument
from secretvaults.common.types import Uuid


class TestNilDBHandler:
    """Test class for nilDB handler functions"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for tests"""
        with patch("nilai_api.handlers.nildb.handler.CONFIG") as mock_config:
            # Mock the nested nildb config structure
            mock_nildb = MagicMock()
            mock_nildb.nilchain_url = "http://test-nilchain.com"
            mock_nildb.nilauth_url = "http://test-nilauth.com"
            mock_nildb.nodes = ["http://node1.com", "http://node2.com"]
            mock_nildb.builder_private_key = "0x1234567890abcdef"
            mock_nildb.collection = Uuid("12345678-1234-1234-1234-123456789012")
            mock_config.nildb = mock_nildb
            yield mock_config

    @pytest.fixture
    def mock_prompt_document(self):
        """Mock PromptDocument for tests"""
        return PromptDocument(
            document_id="test-document-123", owner_did="did:nil:" + "1" * 66
        )

    @pytest.fixture
    def mock_keypair(self):
        """Mock keypair for tests"""
        mock_keypair = MagicMock()
        mock_keypair.private_key.return_value = "mock_private_key"
        mock_keypair.to_did_string.return_value = "did:nil:builder123"
        return mock_keypair

    @pytest.fixture
    def mock_builder_client(self, mock_keypair):
        """Mock builder client for tests"""
        client = MagicMock()
        client.keypair = mock_keypair

        # Mock the root_token to be a proper envelope-like object
        mock_envelope = MagicMock()
        mock_token = MagicMock()
        mock_envelope.token.token = mock_token
        client.root_token = mock_envelope

        client.refresh_root_token = AsyncMock()
        return client

    @pytest.fixture
    def mock_user_client(self):
        """Mock user client for tests"""
        client = MagicMock()
        client.read_data = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_create_builder_client(self, mock_config):
        """Test creating builder client"""
        with (
            patch(
                "secretvaults.common.keypair.Keypair.from_hex"
            ) as mock_keypair_from_hex,
            patch(
                "nilai_api.handlers.nildb.handler.SecretVaultBuilderClient.from_options"
            ) as mock_from_options,
        ):
            mock_keypair = MagicMock()
            mock_keypair_from_hex.return_value = mock_keypair

            mock_client = MagicMock()
            mock_client.refresh_root_token = AsyncMock()
            mock_from_options.return_value = mock_client

            # Clear the cache first

            result = await create_builder_client()

            mock_keypair_from_hex.assert_called_once_with(
                mock_config.nildb.builder_private_key
            )
            mock_from_options.assert_called_once()
            mock_client.refresh_root_token.assert_called_once()
            assert result == mock_client

    @pytest.mark.asyncio
    async def test_create_user_client(self, mock_config):
        """Test creating user client"""
        with (
            patch(
                "secretvaults.common.keypair.Keypair.from_hex"
            ) as mock_keypair_from_hex,
            patch(
                "nilai_api.handlers.nildb.handler.SecretVaultUserClient.from_options"
            ) as mock_from_options,
        ):
            mock_keypair = MagicMock()
            mock_keypair_from_hex.return_value = mock_keypair

            mock_client = MagicMock()
            mock_from_options.return_value = mock_client

            # Clear the cache first

            result = await create_user_client()

            mock_keypair_from_hex.assert_called_once_with(
                mock_config.nildb.builder_private_key
            )
            mock_from_options.assert_called_once()
            assert result == mock_client

    @pytest.mark.asyncio
    async def test_get_nildb_delegation_token_success(
        self, mock_config, mock_builder_client
    ):
        """Test successful delegation token generation"""
        user_did = f"did:nil:{'1' * 66}"

        with (
            patch(
                "nilai_api.handlers.nildb.handler.create_builder_client",
                new_callable=AsyncMock,
            ) as mock_create_builder,
            patch(
                "nilai_api.handlers.nildb.handler.into_seconds_from_now"
            ) as mock_into_seconds,
        ):
            mock_create_builder.return_value = mock_builder_client
            mock_into_seconds.return_value = 1234567890

            # Mock the entire NucTokenBuilder class to return a string for the chain
            with patch(
                "nilai_api.handlers.nildb.handler.NucTokenBuilder"
            ) as mock_token_builder:
                mock_builder_chain = MagicMock()
                mock_builder_chain.command.return_value = mock_builder_chain
                mock_builder_chain.audience.return_value = mock_builder_chain
                mock_builder_chain.expires_at.return_value = mock_builder_chain
                mock_builder_chain.build.return_value = "delegation_token"

                mock_token_builder.extending.return_value = mock_builder_chain

                result = await get_nildb_delegation_token(user_did)

                assert isinstance(result, PromptDelegationToken)
                assert result.token == "delegation_token"
                assert result.did == "did:nil:builder123"

                mock_token_builder.extending.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_nildb_delegation_token_no_root_token(self, mock_config):
        """Test delegation token generation when no root token is available"""
        user_did = f"did:nil:{'1' * 66}"

        with patch(
            "nilai_api.handlers.nildb.handler.create_builder_client",
            new_callable=AsyncMock,
        ) as mock_create_builder:
            mock_builder = MagicMock()
            mock_builder.root_token = None
            mock_create_builder.return_value = mock_builder

            with pytest.raises(
                ValueError, match="Couldn't extract root NUC token from nilDB profile"
            ):
                await get_nildb_delegation_token(user_did)

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_success(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test successful prompt retrieval from nilDB"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Mock successful document response
            class MockData:
                def __init__(self):
                    self.owner = "did:nil:" + "1" * 66
                    self.prompt = "This is a test prompt"

            class MockResponse:
                def __init__(self):
                    self.data = MockData()

            mock_response = MockResponse()

            mock_user_client.read_data.return_value = mock_response

            result = await get_prompt_from_nildb(mock_prompt_document)

            assert result == "This is a test prompt"
            mock_user_client.read_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_no_response(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval when no response is received"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client
            mock_user_client.read_data.return_value = None

            with pytest.raises(
                ValueError, match="Couldn't get document response from nilDB nodes"
            ):
                await get_prompt_from_nildb(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_wrong_owner(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval when document owner doesn't match"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Mock response with different owner
            class MockDataWrongOwner:
                def __init__(self):
                    self.owner = "did:nil:" + "2" * 66
                    self.prompt = "This is a test prompt"

            class MockResponseWrongOwner:
                def __init__(self):
                    self.data = MockDataWrongOwner()

            mock_response = MockResponseWrongOwner()

            mock_user_client.read_data.return_value = mock_response

            with pytest.raises(
                ValueError,
                match="Non-owning entity trying to invoke access to a document resource",
            ):
                await get_prompt_from_nildb(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_no_prompt_field(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval when prompt field is missing"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Mock response without prompt field
            class MockDataNoPrompt:
                def __init__(self):
                    self.owner = "did:nil:" + "1" * 66
                    # No prompt attribute

            class MockResponseNoPrompt:
                def __init__(self):
                    self.data = MockDataNoPrompt()

            mock_response = MockResponseNoPrompt()

            mock_user_client.read_data.return_value = mock_response

            with pytest.raises(
                ValueError,
                match="Couldn't find prompt field in document response from nilDB",
            ):
                await get_prompt_from_nildb(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_null_prompt(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval when prompt field is None"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Mock response with None prompt
            class MockDataNullPrompt:
                def __init__(self):
                    self.owner = "did:nil:" + "1" * 66
                    self.prompt = None

            class MockResponseNullPrompt:
                def __init__(self):
                    self.data = MockDataNullPrompt()

            mock_response = MockResponseNullPrompt()

            mock_user_client.read_data.return_value = mock_response

            with pytest.raises(
                ValueError, match="Prompt field is None in document response from nilDB"
            ):
                await get_prompt_from_nildb(mock_prompt_document)

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_with_model_dump(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval using model_dump method"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Mock response with model_dump method
            class MockDataModelDump:
                def model_dump(self):
                    return {
                        "owner": "did:nil:" + "1" * 66,
                        "prompt": "Test prompt from model_dump",
                    }

                # Don't provide __dict__ by overriding __getattribute__
                def __getattribute__(self, name):
                    if name == "__dict__":
                        raise AttributeError("__dict__")
                    return super().__getattribute__(name)

            class MockResponseModelDump:
                def __init__(self):
                    self.data = MockDataModelDump()

            mock_response = MockResponseModelDump()
            mock_user_client.read_data.return_value = mock_response

            result = await get_prompt_from_nildb(mock_prompt_document)

            assert result == "Test prompt from model_dump"

    @pytest.mark.asyncio
    async def test_get_prompt_from_nildb_direct_response(
        self, mock_config, mock_prompt_document, mock_user_client
    ):
        """Test prompt retrieval with direct response (no data attribute)"""
        with patch(
            "nilai_api.handlers.nildb.handler.create_user_client",
            new_callable=AsyncMock,
        ) as mock_create_user:
            mock_create_user.return_value = mock_user_client

            # Create a simple object to act as direct response
            class MockDirectResponse:
                def __init__(self):
                    self.owner = "did:nil:" + "1" * 66
                    self.prompt = "Direct response prompt"
                    self.data = None

            mock_response = MockDirectResponse()

            mock_user_client.read_data.return_value = mock_response

            result = await get_prompt_from_nildb(mock_prompt_document)

            assert result == "Direct response prompt"
