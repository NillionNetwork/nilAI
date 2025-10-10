import unittest
from unittest.mock import patch
from nilai_api.auth.nuc_helpers.usage import (
    TokenRateLimits,
    UsageLimitError,
    UsageLimitKind,
)
from ..nuc_helpers import DummyDecodedNucToken, DummyNucTokenEnvelope

from datetime import datetime, timedelta, timezone


class GetUsageLimitTests(unittest.TestCase):
    def setUp(self):
        """Clear the cache before each test, because the cache is global and we use the same dummy token for all tests."""
        TokenRateLimits.from_token.cache_clear()

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_no_usage_limit_returns_none(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken(), DummyDecodedNucToken()]
        )
        mock_parse.return_value = env

        limits = TokenRateLimits.from_token("dummy_token")
        self.assertIsNone(limits)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_single_usage_limit_returns_value(self, mock_parse):
        env = DummyNucTokenEnvelope(proofs=[DummyDecodedNucToken({"usage_limit": 10})])
        mock_parse.return_value = env

        limits = TokenRateLimits.from_token("dummy_token")
        if limits is None:
            self.fail("Limits should not be None")
        self.assertEqual(limits.last.usage_limit, 10)

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

        limits = TokenRateLimits.from_token("dummy_token")
        if limits is None:
            self.fail("Limits should not be None")
        self.assertEqual(limits.last.usage_limit, 25)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits_with_none(self, mock_parse):
        limits = [25, None, 100]
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

        limits = TokenRateLimits.from_token("dummy_token")
        if limits is None:
            self.fail("Limits should not be None")
        self.assertEqual(limits.last.usage_limit, 25)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits_with_none_2(self, mock_parse):
        limits = [25, 100, None]
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": limit}) for limit in limits]
        )
        mock_parse.return_value = env

        token_rate_limits = TokenRateLimits.from_token("dummy_token")
        if token_rate_limits is None:
            self.fail("Limits should not be None")
        limits_reversed_without_none = [
            limit for limit in limits[::-1] if limit is not None
        ]
        for effective_limit, expected_limit in zip(
            token_rate_limits.limits, limits_reversed_without_none
        ):
            self.assertEqual(effective_limit.usage_limit, expected_limit)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_multiple_consistent_limits_long(self, mock_parse):
        limits = [25, 32, None, 50, None, None, 75, 100, None]
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": limit}) for limit in limits]
        )
        mock_parse.return_value = env

        token_rate_limits = TokenRateLimits.from_token("dummy_token")
        if token_rate_limits is None:
            self.fail("Limits should not be None")
        limits_reversed_without_none = [
            limit for limit in limits[::-1] if limit is not None
        ]
        for effective_limit, expected_limit in zip(
            token_rate_limits.limits, limits_reversed_without_none
        ):
            self.assertEqual(effective_limit.usage_limit, expected_limit)

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
            TokenRateLimits.from_token("dummy_token")
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
            TokenRateLimits.from_token("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INCONSISTENT)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_inconsistent_usage_limits_with_negative_raises_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": 80}),
                DummyDecodedNucToken({"usage_limit": -90}),
                DummyDecodedNucToken({"usage_limit": 100}),
            ]
        )
        mock_parse.return_value = env

        with self.assertRaises(UsageLimitError) as cm:
            TokenRateLimits.from_token("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INCONSISTENT)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_inconsistent_usage_limits_with_long_chain(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": 50}),
                DummyDecodedNucToken({"usage_limit": 74}),
                DummyDecodedNucToken({"usage_limit": 85}),
                DummyDecodedNucToken({"usage_limit": 88}),
                DummyDecodedNucToken({"usage_limit": None}),
                DummyDecodedNucToken({"usage_limit": -89}),
                DummyDecodedNucToken({"usage_limit": 99}),
                DummyDecodedNucToken({"usage_limit": None}),
                DummyDecodedNucToken({"usage_limit": 100}),
            ]
        )
        mock_parse.return_value = env

        with self.assertRaises(UsageLimitError) as cm:
            TokenRateLimits.from_token("dummy_token")
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
            TokenRateLimits.from_token("dummy_token")
        self.assertEqual(cm.exception.kind, UsageLimitKind.INVALID_TYPE)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_none_type_usage_doesnt_raise_error(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[
                DummyDecodedNucToken({"usage_limit": None}),
            ]
        )
        mock_parse.return_value = env

        limits = TokenRateLimits.from_token("dummy_token")
        self.assertIsNone(limits)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_invocation_usage_limit_ignored(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": 5})],
            invocation_meta={"usage_limit": 999},  # Should be ignored
        )
        mock_parse.return_value = env

        limits = TokenRateLimits.from_token("dummy_token")
        if limits is None:
            self.fail("Limits should not be None")
        self.assertEqual(limits.last.usage_limit, 5)

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_caching_behavior(self, mock_parse):
        env = DummyNucTokenEnvelope(proofs=[DummyDecodedNucToken({"usage_limit": 10})])
        mock_parse.return_value = env

        TokenRateLimits.from_token("dummy_token")
        TokenRateLimits.from_token("dummy_token")

        # NucTokenEnvelope.parse should only be called once due to caching
        mock_parse.assert_called_once()

    @patch("nuc.envelope.NucTokenEnvelope.parse")
    def test_expires_at_returns_correct_value(self, mock_parse):
        env = DummyNucTokenEnvelope(
            proofs=[DummyDecodedNucToken({"usage_limit": 10, "expires_at": 1715702400})]
        )
        mock_parse.return_value = env

        limits = TokenRateLimits.from_token("dummy_token")
        if limits is None:
            self.fail("Limits should not be None")
        expires_at = limits.last.expires_at

        # Check expires_at is less than 1 day from now
        self.assertLess(expires_at, datetime.now(timezone.utc) + timedelta(days=1))  # type: ignore


if __name__ == "__main__":
    unittest.main()
