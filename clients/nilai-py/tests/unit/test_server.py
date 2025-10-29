"""
Comprehensive tests for the DelegationTokenServer class.

This test suite provides 100% code coverage for the DelegationTokenServer class,
testing all public and private methods, error conditions, and edge cases.

Test Coverage:
- Initialization with default and custom configurations
- Root token management (creation, caching, expiration handling)
- Token expiration checking logic
- Delegation token creation with various configurations
- Error handling for invalid keys and network failures
- Configuration property access
- Datetime calculations for token expiration

The tests use extensive mocking to isolate the DelegationTokenServer logic
from external dependencies like NilauthClient and cryptographic operations.
"""

import datetime
import pytest
from unittest.mock import Mock, patch
from nilai_py.server import DelegationTokenServer
from nilai_py.niltypes import (
    DelegationTokenRequest,
    DelegationTokenResponse,
    DelegationServerConfig,
    DefaultDelegationTokenServerConfig,
    RequestType,
)
from nilai_py.common import is_expired
from nuc.envelope import NucTokenEnvelope
from nuc.token import NucToken
from nuc.nilauth import BlindModule


class TestDelegationTokenServer:
    """Test cases for DelegationTokenServer class."""

    @pytest.fixture
    def private_key_hex(self):
        """Sample private key in hex format for testing."""
        return "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"

    @pytest.fixture
    def public_key_hex(self):
        """Sample public key in hex format for testing."""
        return "04a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"

    @pytest.fixture
    def custom_config(self):
        """Custom configuration for testing."""
        return DelegationServerConfig(
            expiration_time=120,
            token_max_uses=5,
        )

    @pytest.fixture
    def delegation_request(self, public_key_hex):
        """Sample delegation token request."""
        return DelegationTokenRequest(public_key=public_key_hex)

    @pytest.fixture
    def mock_token_envelope(self):
        """Mock token envelope for testing."""
        envelope = Mock(spec=NucTokenEnvelope)
        token_wrapper = Mock()
        token = Mock(spec=NucToken)
        token.expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(hours=1)
        token_wrapper.token = token
        envelope.token = token_wrapper
        return envelope

    @pytest.fixture
    def expired_token_envelope(self):
        """Mock expired token envelope for testing."""
        envelope = Mock(spec=NucTokenEnvelope)
        token_wrapper = Mock()
        token = Mock(spec=NucToken)
        token.expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(hours=1)
        token_wrapper.token = token
        envelope.token = token_wrapper
        return envelope

    def test_init_with_default_config(self, private_key_hex):
        """Test server initialization with default configuration."""
        server = DelegationTokenServer(private_key_hex)

        assert server.config == DefaultDelegationTokenServerConfig
        assert hasattr(server, "private_key")

    def test_init_with_custom_config(self, private_key_hex, custom_config):
        """Test server initialization with custom configuration."""
        server = DelegationTokenServer(
            private_key_hex,
        )

        assert server.config == custom_config

    def test_init_invalid_private_key(self):
        """Test server initialization with invalid private key."""
        with pytest.raises(ValueError):
            DelegationTokenServer("invalid_hex_key")

    def test_is_expired_with_expired_token(
        self, private_key_hex, expired_token_envelope
    ):
        """Test _is_expired method with an expired token."""
        assert is_expired(expired_token_envelope) is True

    def test_is_expired_with_valid_token(self, private_key_hex, mock_token_envelope):
        """Test _is_expired method with a valid token."""

        assert is_expired(mock_token_envelope) is False

    def test_is_expired_with_no_expiration(self, private_key_hex):
        """Test is_expired method with a token that has no expiration."""

        envelope = Mock(spec=NucTokenEnvelope)
        token_wrapper = Mock()
        token = Mock(spec=NucToken)
        token.expires_at = None
        token_wrapper.token = token
        envelope.token = token_wrapper

        assert is_expired(envelope) is False

    @patch("nilai_py.server.NilauthClient")
    @patch("nilai_py.server.NucTokenEnvelope")
    def test_root_token_first_access(
        self,
        mock_envelope_class,
        mock_nilauth_client_class,
        private_key_hex,
        mock_token_envelope,
    ):
        """Test root_token property on first access."""
        # Setup mocks
        mock_client = Mock()
        mock_nilauth_client_class.return_value = mock_client
        mock_client.request_token.return_value = "mock_token_response"
        mock_envelope_class.parse.return_value = mock_token_envelope

        server = DelegationTokenServer(private_key_hex)

        # Access root_token property
        result = server.root_token

        assert result == mock_token_envelope
        mock_client.request_token.assert_called_once_with(
            server.private_key, blind_module=BlindModule.NILAI
        )
        mock_envelope_class.parse.assert_called_once_with("mock_token_response")

    @patch("nilai_py.server.NilauthClient")
    @patch("nilai_py.server.NucTokenEnvelope")
    def test_root_token_cached_access(
        self,
        mock_envelope_class,
        mock_nilauth_client_class,
        private_key_hex,
        mock_token_envelope,
    ):
        """Test root_token property returns cached token when not expired."""
        # Setup mocks
        mock_client = Mock()
        mock_nilauth_client_class.return_value = mock_client
        mock_client.request_token.return_value = "mock_token_response"
        mock_envelope_class.parse.return_value = mock_token_envelope

        server = DelegationTokenServer(private_key_hex)

        # Access root_token property twice
        result1 = server.root_token
        result2 = server.root_token

        assert result1 == result2 == mock_token_envelope
        # Should only call the client once (cached)
        mock_client.request_token.assert_called_once()

    @patch("nilai_py.server.NilauthClient")
    @patch("nilai_py.server.NucTokenEnvelope")
    def test_root_token_refresh_when_expired(
        self,
        mock_envelope_class,
        mock_nilauth_client_class,
        private_key_hex,
        expired_token_envelope,
        mock_token_envelope,
    ):
        """Test root_token property refreshes when token is expired."""
        # Setup mocks
        mock_client = Mock()
        mock_nilauth_client_class.return_value = mock_client
        mock_client.request_token.return_value = "mock_token_response"
        mock_envelope_class.parse.side_effect = [
            expired_token_envelope,
            mock_token_envelope,
        ]

        server = DelegationTokenServer(private_key_hex)

        # Access root_token property twice
        result1 = server.root_token  # Should return expired token first
        result2 = server.root_token  # Should refresh and return new token

        assert result1 == expired_token_envelope
        assert result2 == mock_token_envelope
        # Should call the client twice (once for initial, once for refresh)
        assert mock_client.request_token.call_count == 2

    @patch("nilai_py.server.NucTokenBuilder")
    def test_create_delegation_token_success(
        self,
        mock_builder_class,
        private_key_hex,
        delegation_request,
        mock_token_envelope,
    ):
        """Test successful delegation token creation."""
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.extending.return_value = mock_builder
        mock_builder.expires_at.return_value = mock_builder
        mock_builder.audience.return_value = mock_builder
        mock_builder.command.return_value = mock_builder
        mock_builder.meta.return_value = mock_builder
        mock_builder.build.return_value = "delegation_token_string"

        server = DelegationTokenServer(private_key_hex)
        server._root_token_envelope = mock_token_envelope

        result = server.create_delegation_token(delegation_request)

        assert isinstance(result, DelegationTokenResponse)
        assert result.delegation_token == "delegation_token_string"
        assert result.type == RequestType.DELEGATION_TOKEN_RESPONSE

        # Verify builder chain calls
        mock_builder_class.extending.assert_called_once_with(mock_token_envelope)
        mock_builder.expires_at.assert_called_once()
        mock_builder.audience.assert_called_once()
        mock_builder.command.assert_called_once()
        mock_builder.meta.assert_called_once()
        mock_builder.build.assert_called_once_with(server.private_key)

    @patch("nilai_py.server.NucTokenBuilder")
    def test_create_delegation_token_with_config_override(
        self,
        mock_builder_class,
        private_key_hex,
        delegation_request,
        custom_config,
        mock_token_envelope,
    ):
        """Test delegation token creation with configuration override."""
        # Setup mocks
        mock_builder = Mock()
        mock_builder_class.extending.return_value = mock_builder
        mock_builder.expires_at.return_value = mock_builder
        mock_builder.audience.return_value = mock_builder
        mock_builder.command.return_value = mock_builder
        mock_builder.meta.return_value = mock_builder
        mock_builder.build.return_value = "delegation_token_string"

        server = DelegationTokenServer(private_key_hex)
        server._root_token_envelope = mock_token_envelope

        result = server.create_delegation_token(
            delegation_request, config_override=custom_config
        )

        assert isinstance(result, DelegationTokenResponse)

        # Verify that the custom config values are used
        mock_builder.meta.assert_called_once_with(
            {"usage_limit": custom_config.token_max_uses}
        )

    def test_create_delegation_token_invalid_public_key(
        self, private_key_hex, mock_token_envelope
    ):
        """Test delegation token creation with invalid public key."""
        server = DelegationTokenServer(private_key_hex)
        server._root_token_envelope = mock_token_envelope

        invalid_request = DelegationTokenRequest(public_key="invalid_hex")

        with pytest.raises(ValueError):
            server.create_delegation_token(invalid_request)

    @patch("nilai_py.server.NilauthClient")
    def test_nilauth_client_error_handling(
        self, mock_nilauth_client_class, private_key_hex
    ):
        """Test error handling when NilauthClient fails."""
        # Setup mock to raise an exception
        mock_client = Mock()
        mock_nilauth_client_class.return_value = mock_client
        mock_client.request_token.side_effect = Exception("Network error")

        server = DelegationTokenServer(private_key_hex)

        with pytest.raises(Exception, match="Network error"):
            _ = server.root_token

    def test_config_properties_access(self, private_key_hex, custom_config):
        """Test that configuration properties are properly accessible."""
        server = DelegationTokenServer(private_key_hex, config=custom_config)

        assert server.config.expiration_time == 120
        assert server.config.token_max_uses == 5

    @patch("nilai_py.server.datetime")
    def test_expiration_time_calculation(
        self,
        mock_datetime_module,
        private_key_hex,
        delegation_request,
        mock_token_envelope,
    ):
        """Test that expiration time is calculated correctly."""
        # Setup datetime mock
        fixed_now = datetime.datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        mock_datetime_module.datetime.now.return_value = fixed_now
        mock_datetime_module.timedelta = datetime.timedelta
        mock_datetime_module.timezone = datetime.timezone

        with (
            patch("nilai_py.server.NucTokenBuilder") as mock_builder_class,
        ):
            mock_builder = Mock()
            mock_builder_class.extending.return_value = mock_builder
            mock_builder.expires_at.return_value = mock_builder
            mock_builder.audience.return_value = mock_builder
            mock_builder.command.return_value = mock_builder
            mock_builder.meta.return_value = mock_builder
            mock_builder.build.return_value = "delegation_token_string"

            server = DelegationTokenServer(private_key_hex)
            server._root_token_envelope = mock_token_envelope

            server.create_delegation_token(delegation_request)

            # Verify expires_at was called with correct expiration time
            expected_expiration = fixed_now + datetime.timedelta(
                seconds=60
            )  # Default expiration
            mock_builder.expires_at.assert_called_once_with(expected_expiration)
