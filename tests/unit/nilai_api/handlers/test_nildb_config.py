import pytest
from unittest.mock import patch
from nilai_api.handlers.nildb.config import get_required_env_var, NilDBConfig
from secretvaults.common.types import Uuid


class TestNilDBConfig:
    """Test class for nilDB configuration"""

    def test_get_required_env_var_success(self):
        """Test successful environment variable retrieval"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "test_value"

            result = get_required_env_var("TEST_VAR")

            assert result == "test_value"
            mock_getenv.assert_called_once_with("TEST_VAR", None)

    def test_get_required_env_var_not_set(self):
        """Test environment variable retrieval when variable is not set"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = None

            with pytest.raises(
                ValueError, match="Required environment variable TEST_VAR is not set"
            ):
                get_required_env_var("TEST_VAR")

    def test_get_required_env_var_empty_string(self):
        """Test environment variable retrieval when variable is empty string"""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = ""

            # Empty string should be returned as is (not treated as None)
            result = get_required_env_var("TEST_VAR")
            assert result == ""

    def test_nildb_config_model_validation(self):
        """Test NilDBConfig model validation with valid data"""
        config = NilDBConfig(
            NILCHAIN_URL="http://test-nilchain.com",
            NILAUTH_URL="http://test-nilauth.com",
            NODES=["http://node1.com", "http://node2.com", "http://node3.com"],
            BUILDER_PRIVATE_KEY="0x1234567890abcdef1234567890abcdef12345678",
            COLLECTION=Uuid("12345678-1234-1234-1234-123456789012"),
        )

        assert config.NILCHAIN_URL == "http://test-nilchain.com"
        assert config.NILAUTH_URL == "http://test-nilauth.com"
        assert config.NODES == [
            "http://node1.com",
            "http://node2.com",
            "http://node3.com",
        ]
        assert (
            config.BUILDER_PRIVATE_KEY == "0x1234567890abcdef1234567890abcdef12345678"
        )
        assert str(config.COLLECTION) == "12345678-1234-1234-1234-123456789012"

    def test_nildb_config_with_single_node(self):
        """Test NilDBConfig with single node in list"""
        config = NilDBConfig(
            NILCHAIN_URL="http://test-nilchain.com",
            NILAUTH_URL="http://test-nilauth.com",
            NODES=["http://single-node.com"],
            BUILDER_PRIVATE_KEY="0x1234567890abcdef1234567890abcdef12345678",
            COLLECTION=Uuid("12345678-1234-1234-1234-123456789012"),
        )

        assert len(config.NODES) == 1
        assert config.NODES[0] == "http://single-node.com"

    def test_config_initialization(self):
        """Test CONFIG initialization process"""
        with patch("os.getenv") as mock_getenv:
            # Mock environment variable values - need to handle all possible getenv calls
            def mock_getenv_impl(name, default=None):
                env_values = {
                    "NILDB_NILCHAIN_URL": "http://test-nilchain.com",
                    "NILDB_NILAUTH_URL": "http://test-nilauth.com",
                    "NILDB_NODES": "http://node1.com,http://node2.com,http://node3.com",
                    "NILDB_BUILDER_PRIVATE_KEY": "0x1234567890abcdef1234567890abcdef12345678",
                    "NILDB_COLLECTION": "12345678-1234-1234-1234-123456789012",
                }
                return env_values.get(name, default)

            mock_getenv.side_effect = mock_getenv_impl

            # Reimport the config module to trigger initialization
            import importlib
            import nilai_api.handlers.nildb.config

            importlib.reload(nilai_api.handlers.nildb.config)

            # Check that the CONFIG has been properly initialized
            config = nilai_api.handlers.nildb.config.CONFIG
            assert config.NILCHAIN_URL == "http://test-nilchain.com"
            assert config.NILAUTH_URL == "http://test-nilauth.com"
            assert config.NODES == [
                "http://node1.com",
                "http://node2.com",
                "http://node3.com",
            ]
            assert (
                config.BUILDER_PRIVATE_KEY
                == "0x1234567890abcdef1234567890abcdef12345678"
            )

    def test_nodes_string_splitting(self):
        """Test that NODES string is properly split by commas"""
        nodes_string = "http://node1.com,http://node2.com,http://node3.com"
        nodes_list = nodes_string.split(",")

        assert len(nodes_list) == 3
        assert nodes_list == [
            "http://node1.com",
            "http://node2.com",
            "http://node3.com",
        ]

    def test_nodes_string_splitting_with_spaces(self):
        """Test NODES string splitting with spaces around commas"""
        nodes_string = "http://node1.com, http://node2.com ,http://node3.com"
        nodes_list = [node.strip() for node in nodes_string.split(",")]

        assert len(nodes_list) == 3
        assert nodes_list == [
            "http://node1.com",
            "http://node2.com",
            "http://node3.com",
        ]

    def test_uuid_validation(self):
        """Test UUID validation in NilDBConfig"""
        valid_uuid_str = "12345678-1234-1234-1234-123456789012"
        uuid_obj = Uuid(valid_uuid_str)

        config = NilDBConfig(
            NILCHAIN_URL="http://test-nilchain.com",
            NILAUTH_URL="http://test-nilauth.com",
            NODES=["http://node1.com"],
            BUILDER_PRIVATE_KEY="0x1234567890abcdef1234567890abcdef12345678",
            COLLECTION=uuid_obj,
        )

        assert config.COLLECTION == uuid_obj
        assert str(config.COLLECTION) == valid_uuid_str

    def test_invalid_uuid_raises_error(self):
        """Test that invalid UUID raises validation error"""
        import uuid

        with pytest.raises(ValueError):
            uuid.UUID("invalid-uuid-string")
