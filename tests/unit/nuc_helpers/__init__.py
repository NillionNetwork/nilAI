"""
Shared test utilities for nuc-helpers tests.

This module contains common dummy classes and utilities used across
multiple test files to avoid code duplication.
"""

from datetime import datetime, timedelta, timezone
from nuc.token import Did


class DummyNucToken:
    """Dummy NUC token for testing purposes."""

    def __init__(self, meta=None, issuer=None, expires_at=None):
        self.meta = meta or {}
        self.issuer = issuer or Did.parse(f"did:nil:{'1' * 66}")
        self.expires_at = expires_at or (datetime.now(timezone.utc) + timedelta(days=1))


class DummyDecodedNucToken:
    """Dummy decoded NUC token for testing purposes."""

    def __init__(self, meta=None, issuer=None, expires_at=None):
        self.token = DummyNucToken(meta, issuer, expires_at)
        self.signature = b"\x01\x02"


class DummyNucTokenEnvelope:
    """Dummy NUC token envelope for testing purposes."""

    def __init__(self, proofs, invocation_meta=None):
        self.proofs = proofs
        self.token = DummyDecodedNucToken(invocation_meta)
