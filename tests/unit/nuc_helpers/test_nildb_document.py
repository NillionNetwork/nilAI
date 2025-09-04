import unittest
from unittest.mock import patch, MagicMock
from nuc_helpers.nildb_document import PromptDocument
from nuc.token import Did
from ..nuc_helpers import DummyDecodedNucToken, DummyNucTokenEnvelope


class TestPromptDocument(unittest.TestCase):
    def setUp(self):
        """Clear the cache before each test"""
        PromptDocument.from_token.cache_clear()

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_from_token_no_document_id_returns_none(self, mock_parse):
        """Test that from_token returns None when no document_id is found"""
        proofs = [
            DummyDecodedNucToken({}),
            DummyDecodedNucToken({"other_field": "value"}),
        ]
        envelope = DummyNucTokenEnvelope(proofs)
        mock_parse.return_value = envelope

        result = PromptDocument.from_token("dummy_token")
        self.assertIsNone(result)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_from_token_with_document_id_returns_prompt_document(self, mock_parse):
        """Test that from_token returns PromptDocument when document_id is found"""
        issuer_did = Did.parse(f"did:nil:{'1' * 66}")
        document_id = "test-document-123"

        proofs = [
            DummyDecodedNucToken({"document_id": document_id}, issuer_did),
            DummyDecodedNucToken({}),
        ]
        envelope = DummyNucTokenEnvelope(proofs)
        mock_parse.return_value = envelope

        result = PromptDocument.from_token(
            "dummy_token"
        )  # will return the envelope above

        self.assertIsNotNone(result)
        self.assertEqual(result.document_id, document_id)  # type: ignore
        self.assertEqual(result.owner_did, issuer_did)  # type: ignore

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_from_token_multiple_document_ids_returns_first(self, mock_parse):
        """Test that from_token returns the first document_id found (uppermost in chain)"""
        issuer_did_1 = Did.parse(f"did:nil:{'1' * 66}")
        issuer_did_2 = Did.parse(f"did:nil:{'2' * 66}")
        document_id_1 = "first-document"
        document_id_2 = "second-document"

        # Note: proofs are processed in reverse order, so the last one is "uppermost"
        proofs = [
            DummyDecodedNucToken({"document_id": document_id_2}, issuer_did_2),
            DummyDecodedNucToken({"document_id": document_id_1}, issuer_did_1),
        ]
        envelope = DummyNucTokenEnvelope(proofs)
        mock_parse.return_value = envelope

        result = PromptDocument.from_token(
            "dummy_token"
        )  # will return the envelope above

        self.assertIsNotNone(result)
        self.assertEqual(result.document_id, document_id_1)  # type: ignore
        self.assertEqual(result.owner_did, issuer_did_1)  # type: ignore

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_from_token_with_none_document_id_skips(self, mock_parse):
        """Test that from_token skips proofs with None document_id"""
        issuer_did = Did.parse(f"did:nil:{'1' * 66}")
        document_id = "valid-document"

        proofs = [
            DummyDecodedNucToken({"document_id": None}),
            DummyDecodedNucToken({"document_id": document_id}, issuer_did),
        ]
        envelope = DummyNucTokenEnvelope(proofs)
        mock_parse.return_value = envelope

        result = PromptDocument.from_token(
            "dummy_token"
        )  # will return the envelope above

        self.assertIsNotNone(result)
        self.assertEqual(result.document_id, document_id)  # type: ignore
        self.assertEqual(result.owner_did, issuer_did)  # type: ignore

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_from_token_with_null_token_meta_skips(self, mock_parse):
        """Test that from_token skips proofs with null token meta"""
        issuer_did = Did.parse(f"did:nil:{'1' * 66}")
        document_id = "valid-document"

        # Create a proof with null token
        proof_with_null_token = MagicMock()
        proof_with_null_token.token = None

        proofs = [
            proof_with_null_token,
            DummyDecodedNucToken({"document_id": document_id}, issuer_did),
        ]
        envelope = DummyNucTokenEnvelope(proofs)
        mock_parse.return_value = envelope

        result = PromptDocument.from_token(
            "dummy_token"
        )  # will return the envelope above

        self.assertIsNotNone(result)
        self.assertEqual(result.document_id, document_id)  # type: ignore
        self.assertEqual(result.owner_did, issuer_did)  # type: ignore

    def test_prompt_document_model_validation(self):
        """Test that PromptDocument model validates correctly"""
        issuer_did = Did.parse(f"did:nil:{'1' * 66}")
        document_id = "test-document-123"

        prompt_doc = PromptDocument(document_id=document_id, owner_did=issuer_did)

        self.assertEqual(prompt_doc.document_id, document_id)  # type: ignore
        self.assertEqual(prompt_doc.owner_did, issuer_did)  # type: ignore

    def test_cache_functionality(self):
        """Test that the cache works correctly"""
        with patch("nuc.envelope.NucTokenEnvelope.parse") as mock_parse:
            issuer_did = Did.parse(f"did:nil:{'1' * 66}")
            document_id = "cached-document"

            proofs = [DummyDecodedNucToken({"document_id": document_id}, issuer_did)]
            envelope = DummyNucTokenEnvelope(proofs)
            mock_parse.return_value = envelope

            token = "test_token"

            # First call
            result1 = PromptDocument.from_token(token)  # will return the envelope above
            # Second call - should use cache
            result2 = PromptDocument.from_token(token)

            # Should only parse once due to caching
            mock_parse.assert_called_once()

            # Both results should be identical
            self.assertEqual(result1.document_id, result2.document_id)  # type: ignore
            self.assertEqual(result1.owner_did, result2.owner_did)  # type: ignore


if __name__ == "__main__":
    unittest.main()
