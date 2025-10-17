from datetime import datetime, timezone
from typing import Optional, Tuple
from nuc.validate import NucTokenValidator, ValidationParameters, InvocationRequirement
from nuc.envelope import NucTokenEnvelope
from nuc.token import Did, NucToken, Command
from functools import lru_cache
from nilai_api.state import state
from nilai_api.auth.common import AuthenticationError


from nilai_api.auth.nuc_helpers.usage import TokenRateLimits
from nilai_api.auth.nuc_helpers.nildb_document import PromptDocument

import logging

logger = logging.getLogger(__name__)

NILAI_BASE_COMMAND: Command = Command.parse("/nil/ai")


@lru_cache(maxsize=1)
def get_validator() -> NucTokenValidator:
    """
    Get the public key of the Nilauth service

    Returns:
        A NucTokenValidator that can be used to validate NUC tokens
        The validator is cached to avoid re-initializing the validator for each request
    """

    # From now on, we don't need to trust any root issuer
    # The users can issue their own tokens.
    # The tokens are charged to the root issuer.
    # Here we just validate that the token is valid and not expired.
    # And that the chain is valid.
    return NucTokenValidator([])


@lru_cache(maxsize=1)
def get_validation_parameters() -> ValidationParameters:
    """
    Get the validation parameters for the NUC token

    Returns:
        The validation parameters for the NUC token
    """
    default_parameters = ValidationParameters.default()
    default_parameters.token_requirements = InvocationRequirement(
        audience=Did(state.public_key.serialize())
    )
    return default_parameters


def check_is_nilai_subcommand(nuc_token_envelope: NucTokenEnvelope) -> bool:
    """
    Check if the NUC token is a Nilai subcommand
    """
    command: Command = nuc_token_envelope.token.token.command
    return command.is_attenuation_of(NILAI_BASE_COMMAND)


def validate_nuc(nuc_token: str) -> Tuple[str, str]:
    """
    Validate a NUC token

    Args:
        nuc_token: The NUC token to validate

    Returns:
        The subscription holder and the user that the NUC token is for in hex format (str, str)
    """
    nuc_token_envelope = NucTokenEnvelope.parse(nuc_token)
    logger.info(f"Validating NUC token: {nuc_token_envelope.token.token}")
    logger.info(f"Validation parameters: {get_validation_parameters()}")
    logger.info(f"Public key: {state.public_key.serialize()}")
    if not check_is_nilai_subcommand(nuc_token_envelope):
        logger.error(
            f"NUC token namespace is not a /nil/ai attenuation: {nuc_token_envelope.token.token.command}"
        )
        raise AuthenticationError("NUC token namespace is not a /nil/ai attenuation")

    get_validator().validate(
        nuc_token_envelope, context={}, parameters=get_validation_parameters()
    )
    token: NucToken = nuc_token_envelope.token.token

    # Validate the
    # Return the subject of the token, the subscription holder
    subscription_holder = token.subject.public_key.hex()
    user = token.issuer.public_key.hex()
    logger.info(f"Subscription holder: {subscription_holder}")
    logger.info(f"User: {user}")
    return subscription_holder, user


def get_token_rate_limit(nuc_token: str) -> Optional[TokenRateLimits]:
    """
    Get the rate limit for the NUC token

    Args:
        nuc_token: The NUC token to get the rate limit for

    Returns:
        The rate limit for the NUC token

    Raises:
        UsageLimitError: If the usage limit is not found or is invalid
    """
    token_rate_limits = TokenRateLimits.from_token(nuc_token)
    if not token_rate_limits:
        return None
    for limit in token_rate_limits.limits:
        if limit.usage_limit is None:
            raise AuthenticationError("Token has no usage limit")
        if limit.expires_at < datetime.now(timezone.utc):
            raise AuthenticationError("Token has expired")

    return token_rate_limits


def get_token_prompt_document(nuc_token: str) -> Optional[PromptDocument]:
    prompt_document = PromptDocument.from_token(nuc_token)
    return prompt_document
