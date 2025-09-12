import pytest
from unittest.mock import patch
from nilai_api.auth.nuc import get_token_prompt_document
from nilai_api.auth.common import PromptDocument


class TestNucAuthFunctions:
    """Test class for NUC authentication functions"""

    @patch("nilai_api.auth.nuc.PromptDocument.from_token")
    def test_get_token_prompt_document_success(self, mock_from_token):
        """Test successful prompt document extraction"""
        mock_prompt_doc = PromptDocument(
            document_id="test-doc-123", owner_did=f"did:nil:{'1' * 66}"
        )
        mock_from_token.return_value = mock_prompt_doc

        result = get_token_prompt_document("test_token")

        assert result == mock_prompt_doc
        mock_from_token.assert_called_once_with("test_token")

    @patch("nilai_api.auth.nuc.PromptDocument.from_token")
    def test_get_token_prompt_document_none(self, mock_from_token):
        """Test when no prompt document is found"""
        mock_from_token.return_value = None

        result = get_token_prompt_document("test_token")

        assert result is None
        mock_from_token.assert_called_once_with("test_token")

    @patch("nilai_api.auth.nuc.PromptDocument.from_token")
    def test_get_token_prompt_document_exception(self, mock_from_token):
        """Test when PromptDocument.from_token raises an exception"""
        mock_from_token.side_effect = Exception("Token parsing failed")

        # The function should let the exception bubble up
        with pytest.raises(Exception, match="Token parsing failed"):
            get_token_prompt_document("invalid_token")
