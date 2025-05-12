import unittest
from unittest.mock import patch
from nuc_helpers.usage import get_usage_limit, UsageLimitError, UsageLimitKind

from datetime import datetime, timedelta


# Dummy token envelope structure to simulate nuc.envelope
class DummyNucToken:
    def __init__(self, meta=None, expires_at=datetime.now() + timedelta(days=1)):
        self.meta = meta or {}
        self.expires_at = expires_at


class DummyDecodedNucToken:
    def __init__(self, meta=None):
        self.token = DummyNucToken(meta)
        self.signature = b"\x01\x02"


class DummyNucTokenEnvelope:
    def __init__(self, proofs, invocation_meta=None):
        self.proofs = proofs
        self.token = DummyDecodedNucToken(invocation_meta)


class GetUsageLimitTests(unittest.TestCase):
    def setUp(self):
        """Clear the cache before each test, because the cache is global and we use the same dummy token for all tests."""
        get_usage_limit.cache_clear()

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_no_usage_limit_returns_none(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken(), DummyDecodedNucToken()]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, None)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_single_usage_limit_returns_value(self, mock_parse):
        env = DummyNucTokenEnvelope(proofs=[DummyDecodedNucToken({"usage_limit": 10})])
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, 10)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken(
                    {"usage_limit": 25}
                ),  # This is a second reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": 50}
                ),  # This is a first reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": 100}
                ),  # This is the base usage limit
            ]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, 25)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits_with_none(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken(
                    {"usage_limit": 25}
                ),  # This is a second reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": None}
                ),  # This is a first reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": 100}
                ),  # This is the base usage limit
            ]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, 25)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits_with_none_2(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken(
                    {"usage_limit": 25}
                ),  # This is a second reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": 100}
                ),  # This is a first reduction of the base usage limit
                DummyDecodedNucToken(
                    {"usage_limit": None}
                ),  # This is the base usage limit
            ]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, 25)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_inconsistent_usage_limits_raises_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": 110}),
                DummyDecodedNucToken({"usage_limit": 90}),
                DummyDecodedNucToken({"usage_limit": 100}),
            ]
        )
        mock_parse.return_value = env

        with self.assertRaises(UsageLimitError) as cm:
            get_usage_limit("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INCONSISTENT)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_inconsistent_usage_limits_with_none_raises_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": 110}),
                DummyDecodedNucToken({"usage_limit": None}),
                DummyDecodedNucToken({"usage_limit": 100}),
            ]
        )
        mock_parse.return_value = env

        with self.assertRaises(UsageLimitError) as cm:
            get_usage_limit("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INCONSISTENT)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_invalid_type_usage_limit_raises_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": "not-an-int"}),
            ]
        )
        mock_parse.return_value = env

        with self.assertRaises(UsageLimitError) as cm:
            get_usage_limit("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INVALID_TYPE)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_none_type_usage_doesnt_raise_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": None}),
            ]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, None)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_invocation_usage_limit_ignored(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": 5})],
            invocation_meta={"usage_limit": 999},  # Should be ignored
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")
        self.assertEqual(limit, 5)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_caching_behavior(self, mock_parse):
        env = DummyNucTokenEnvelope(proofs=[DummyDecodedNucToken({"usage_limit": 10})])
        mock_parse.return_value = env

        get_usage_limit("dummy_token")
        get_usage_limit("dummy_token")

        # NucTokenEnvelope.parse should only be called once due to caching
        mock_parse.assert_called_once()

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_expires_at_returns_correct_value(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": 10, "expires_at": 1715702400})]
        )
        mock_parse.return_value = env

        sig, limit, expires_at = get_usage_limit("dummy_token")

        # Check expires_at is less than 1 day from now
        self.assertLess(expires_at, datetime.now() + timedelta(days=1))  # type: ignore


if __name__ == "__main__":
    unittest.main()
