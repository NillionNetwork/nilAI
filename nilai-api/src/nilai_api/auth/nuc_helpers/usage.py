from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional, List
from nuc.envelope import NucTokenEnvelope
from enum import Enum
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TokenRateLimit(BaseModel):
    signature: str
    expires_at: datetime
    usage_limit: Optional[int]

    @property
    def ms_remaining(self) -> int:
        if self.expires_at is None:
            return 0  # Or handle as infinite, e.g., float('inf'), or raise error
        return int(
            (self.expires_at - datetime.now(timezone.utc)).total_seconds() * 1000
        )


class UsageLimitKind(Enum):
    INCONSISTENT = "Inconsistent usage limit across proofs"
    INVALID_TYPE = "Invalid usage limit type. Usage limit must be an integer."


class UsageLimitError(Exception):
    """
    Usage limit error.
    """

    def __init__(self, kind: UsageLimitKind, message: str) -> None:
        super().__init__(self, f"validation failed: {kind}: {message}")
        self.kind = kind


def is_reduction_of(base: int, reduced: int) -> bool:
    """Check if `reduced` is a valid reduction of `base`."""
    return 0 < reduced <= base


class TokenRateLimits(BaseModel):
    limits: List[TokenRateLimit] = Field(default_factory=list, min_length=1)

    @property
    def last(self) -> TokenRateLimit:
        if len(self.limits) == 0:
            raise ValueError("No limits found")
        return self.limits[-1]

    def get_limit(self, signature: str) -> Optional[TokenRateLimit]:
        for limit in self.limits:
            if limit.signature == signature:
                return limit
        return None

    @staticmethod
    @lru_cache(maxsize=128)
    def from_token(token: str) -> Optional["TokenRateLimits"]:
        """
        Extracts the effective usage limits from a valid NUC delegation token proof chain, ensuring consistency across proofs.

        The token is expected to be a valid NUC delegation token. If the token is invalid,
        the function may not raise an error, but the result can be incorrect.

        This function parses the provided token and inspects all associated proofs and the invocation
        token (if present) to determine the applicable usage limits. The behavior is as follows:

        - If multiple proofs include a `usage_limit` in their metadata, they must all be reductions of
        the same base usage limit. Inconsistencies will raise an error.
        - If the invocation token includes a `usage_limit`, it is ignored.
        - If no usage limits are found in either proofs or invocation, the function returns `None`.

        The function is cached based on the token string to avoid redundant parsing and validation.

        Note: This function is cached, so it will return the same result for the same token string.
        If you need to invalidate the cache, call `get_usage_limit.cache_clear()`.


        Args:
            token (str): The serialized delegation token.

        Returns:
            TokenRateLimit: The signature, the effective usage limit, and the expiration date, or `None` if no usage limit is found.

        Raises:
            UsageLimitInconsistencyError: If usage limits across proofs or invocation are inconsistent.
        """
        token_envelope = NucTokenEnvelope.parse(token)

        usage_limits = []

        # Iterate over proofs and collect usage limits from the root token -> last delegation token
        for i, proof in enumerate(token_envelope.proofs[::-1]):
            meta = proof.token.meta if proof.token else None
            logger.info(f"Proof {i} meta: {meta}")
            if meta and "usage_limit" in meta and meta["usage_limit"] is not None:
                token_usage_limit = meta["usage_limit"]
                logger.info(f"Proof {i} usage limit: {token_usage_limit}")
                if not isinstance(token_usage_limit, int):
                    logger.error(
                        f"Proof {i} has invalid usage limit type: {type(token_usage_limit)} and value: {token_usage_limit}."
                    )
                    raise UsageLimitError(
                        UsageLimitKind.INVALID_TYPE,
                        f"Proof {i} has invalid usage limit type: {type(token_usage_limit)} and value: {token_usage_limit}.",
                    )
                # We have a usage limit, we need to check if it is a reduction of the previous usage limit
                if len(usage_limits) > 0 and not is_reduction_of(
                    usage_limits[-1].usage_limit, token_usage_limit
                ):
                    error_message = f"Inconsistent usage limit: {token_usage_limit} is not a reduction of {usage_limits[-1].usage_limit}"
                    logger.error(error_message)
                    raise UsageLimitError(
                        UsageLimitKind.INCONSISTENT,
                        error_message,
                    )
                logger.info(f"Usage limit updated to: {token_usage_limit}")
                sig = proof.signature.hex()
                expires_at = (
                    proof.token.expires_at
                    if proof.token.expires_at is not None
                    else datetime.now(timezone.utc) - timedelta(days=1)
                )  # Set to a past date to indicate that the token is expired and invalid
                usage_limits.append(
                    TokenRateLimit(
                        signature=sig,
                        expires_at=expires_at,
                        usage_limit=token_usage_limit,
                    )
                )

        if len(usage_limits) == 0:
            return None
        return TokenRateLimits(limits=usage_limits)
