from functools import lru_cache
from typing import Optional
from nuc.envelope import NucTokenEnvelope
import logging
from pydantic import BaseModel

from nuc.token import Did

logger = logging.getLogger(__name__)


class PromptDocument(BaseModel):
    document_id: str
    owner_did: Did

    @staticmethod
    @lru_cache(maxsize=128)
    def from_token(token: str) -> Optional["PromptDocument"]:
        """
        Extracts the prompt_document_id from the NUC token if there is one.

        This serves to determine which document if there is one to extract from nilDB to be used as a prompt


        This function parses the provided token and inspects all associated proofs from upwards down to the invocation
        token (if present) to determine the applicable document. The behavior is as follows:

        - The invocation token is never considered as it is created by the user.
        - The uppermost token containing a `document_id` in their metadata is the one considered.
        - If two `document_id` are present, only the uppermost in the chain is considered.

        The function is cached based on the token string to avoid redundant parsing and validation.

        Note: This function is cached, so it will return the same result for the same token string.
        If you need to invalidate the cache, call `get_usage_limit.cache_clear()`.


        Args:
            token (str): The serialized delegation token.

        Returns:
            PromptDocumentId: The document_id and the issuer did to be matched to the database
        """
        token_envelope = NucTokenEnvelope.parse(token)

        # Iterate over proofs and collect the first document_id found together with issuer.
        for i, proof in enumerate(token_envelope.proofs[::-1]):
            meta = proof.token.meta if proof.token else None
            logger.debug(f"Proof {i} meta: {meta}")
            if meta is not None and meta.get("document_id", None) is not None:
                return PromptDocument(
                    document_id=meta["document_id"], owner_did=proof.token.issuer
                )

        return None
