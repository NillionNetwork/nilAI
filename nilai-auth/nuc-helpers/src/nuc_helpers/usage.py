from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple
from nuc.envelope import NucTokenEnvelope
from enum import Enum
import logging

logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=128)
def get_usage_limit(token: str) -> Tuple[str, Optional[int], Optional[datetime]]:
    """
    Extracts the effective usage limit from a valid NUC delegation token, ensuring consistency across proofs.

    The token is expected to be a valid NUC delegation token. If the token is invalid,
    the function may not raise an error, but the result can be incorrect.

    This function parses the provided token and inspects all associated proofs and the invocation
    token (if present) to determine the applicable usage limit. The behavior is as follows:

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
        Tuple[str, str, Optional[int]]: The signature, the effective usage limit, and the expiration date, or `None` if no usage limit is found.

    Raises:
        UsageLimitInconsistencyError: If usage limits across proofs or invocation are inconsistent.
    """
    token_envelope = NucTokenEnvelope.parse(token)

    usage_limit = None

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
            if usage_limit is not None and not is_reduction_of(
                usage_limit, token_usage_limit
            ):
                logger.error(
                    f"Inconsistent usage limit: {token_usage_limit} is not a reduction of {usage_limit}"
                )
                raise UsageLimitError(
                    UsageLimitKind.INCONSISTENT,
                    f"Inconsistent usage limit: {token_usage_limit} is not a reduction of {usage_limit}",
                )
            logger.info(
                f"Usage limit updated to: {token_usage_limit} from {usage_limit}"
            )
            usage_limit = token_usage_limit

    # Convert the signature to a string for the last delegation token
    signature = token_envelope.proofs[0].signature.hex()
    expires_at = token_envelope.proofs[0].token.expires_at
    return signature, usage_limit, expires_at
